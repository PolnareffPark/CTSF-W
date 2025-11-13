# ============================================================
# experiments/w3_experiment.py
#   - W3: 데이터 구조 원인 검증 (교란 주입 vs 클린 평가)
#   - 학습 단계에서만 교란을 적용하고, 평가 단계는 항상 클린
#   - 결과 스키마는 W1/W2와 동일 (results_W3.csv + summary/detail)
# ============================================================

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_experiment import BaseExperiment
from data.dataset import build_test_tod_vector
from models.ctsf_model import HybridTS
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics
from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    compute_and_save_w3_gate_distribution,
    compute_and_save_w3_gate_tod_heatmap,
    rebuild_w3_forest_summary,
    rebuild_w3_perturb_delta_bar,
    save_w3_forest_detail_row,
)

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"
MODES = ("none", "tod_shift", "smooth")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _infer_tod_bins(dataset: str) -> int:
    """데이터셋 이름을 기반으로 하루 TOD bin 수 추정."""
    ds = (dataset or "").lower()
    if "ettm" in ds:
        return 96  # 15-min
    if "etth" in ds:
        return 24  # 1-hour
    if "weather" in ds:
        return 144  # 10-min
    return 24  # 안전 기본값


def _default_perturb_cfg(dataset: str) -> Dict[str, Any]:
    """데이터셋별 기본 교란 강도 설정."""
    ds = (dataset or "").lower()
    if "ettm2" in ds:
        return {
            "tod_shift": {"max_shift": 8},
            "smooth": {"kernel": 3, "alpha": 0.25},
        }
    if "etth2" in ds:
        return {
            "tod_shift": {"max_shift": 2},
            "smooth": {"kernel": 5, "alpha": 0.35},
        }
    return {
        "tod_shift": {"max_shift": max(1, _infer_tod_bins(dataset) // 48)},
        "smooth": {"kernel": 3, "alpha": 0.25},
    }


def _detect_layout(x: torch.Tensor) -> Tuple[int, int]:
    """
    입력 텐서의 시간축/채널축을 추정.
    허용 형태: (B, L, C) 혹은 (B, C, L)
    반환: (time_axis, channel_axis)
    """
    if x.dim() != 3:
        raise ValueError("W3 wrapper expects 3D input (B, L, C) or (B, C, L).")
    _, a1, a2 = x.shape
    if a1 >= a2:
        return 1, 2  # (B, L, C)
    return 2, 1  # (B, C, L)


def _moving_avg_1d(x: torch.Tensor, k: int, t_axis: int) -> torch.Tensor:
    """시간축 기준 이동평균을 적용."""
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1  # 홀수 강제

    if t_axis == 1:  # (B, L, C)
        x_nc_t = x.permute(0, 2, 1)  # (B, C, L)
    else:
        x_nc_t = x  # (B, C, L)

    b, c, L = x_nc_t.shape
    weight = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
    x_flat = x_nc_t.reshape(b * c, 1, L)
    pad = k // 2
    x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
    y = F.conv1d(x_pad, weight).reshape(b, c, L)

    if t_axis == 1:
        return y.permute(0, 2, 1)
    return y


def _append_or_update_results(csv_path: Path | str, row: Dict[str, Any], subset: List[str]) -> None:
    """results_W3.csv에 행을 추가하거나 동일 키(subset)로 갱신."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])

    if path.exists():
        old_df = pd.read_csv(path)
        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = np.nan
        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged.drop_duplicates(subset=subset, keep="last", inplace=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        merged.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    else:
        new_df.to_csv(path, index=False)


def _merge_perturb_cfg(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _extract_user_perturb_cfg(cfg: Dict[str, Any], dataset: str) -> Optional[Dict[str, Any]]:
    user_cfg = cfg.get("perturb_cfg")
    if not isinstance(user_cfg, dict):
        return None
    ds_lower = (dataset or "").lower()
    for key, value in user_cfg.items():
        if isinstance(key, str) and key.lower() == ds_lower and isinstance(value, dict):
            return value
    keys_lower = {k.lower() for k in user_cfg.keys() if isinstance(k, str)}
    if "tod_shift" in keys_lower or "smooth" in keys_lower:
        return user_cfg
    return None


# ---------------------------------------------------------------------------
# Perturbation Wrapper
# ---------------------------------------------------------------------------
class _W3PerturbWrapper(nn.Module):
    """
    학습 단계에서만 입력 x에 교란을 가하고, 평가 단계에서는 원본 입력을 그대로 사용.
    """

    def __init__(self, base: nn.Module, mode: str, dataset: str, perturb_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.base = base
        self.mode = str(mode or "none")
        self.dataset = str(dataset or "")
        self.cfg = perturb_cfg.copy() if isinstance(perturb_cfg, dict) else _default_perturb_cfg(dataset)

        defaults = _default_perturb_cfg(dataset)
        for key, value in defaults.items():
            self.cfg.setdefault(key, value)

        self.tod_bins = _infer_tod_bins(self.dataset)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if (not self.training) or self.mode == "none":
            return self.base(x, *args, **kwargs)

        if self.mode == "tod_shift":
            x = self._apply_tod_shift(x)
        elif self.mode == "smooth":
            x = self._apply_smooth(x)

        return self.base(x, *args, **kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            base = object.__getattribute__(self, "base")
            if hasattr(base, name):
                return getattr(base, name)
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @torch.no_grad()
    def _apply_tod_shift(self, x: torch.Tensor) -> torch.Tensor:
        time_axis, _ = _detect_layout(x)
        max_shift = int(self.cfg.get("tod_shift", {}).get("max_shift", 1))
        if max_shift <= 0:
            return x

        shift = torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
        if shift == 0 and max_shift > 0:
            shift = 1
        if shift == 0:
            return x
        return torch.roll(x, shifts=shift, dims=time_axis)

    @torch.no_grad()
    def _apply_smooth(self, x: torch.Tensor) -> torch.Tensor:
        time_axis, _ = _detect_layout(x)
        kernel = int(self.cfg.get("smooth", {}).get("kernel", 3))
        alpha = float(self.cfg.get("smooth", {}).get("alpha", 0.25))
        if kernel <= 1 or alpha <= 0:
            return x
        smoothed = _moving_avg_1d(x, kernel, time_axis)
        return (1.0 - alpha) * x + alpha * smoothed


# ---------------------------------------------------------------------------
# W3 Experiment
# ---------------------------------------------------------------------------
class W3Experiment(BaseExperiment):
    """
    BaseExperiment의 학습/평가 루틴을 그대로 사용하되,
    학습 단계에서만 교란을 주고 평가 단계는 항상 클린으로 수행한다.
    """

    def __init__(self, cfg: Dict[str, Any]):
        cfg_local = copy.deepcopy(cfg or {})
        cfg_local.setdefault("experiment_type", EXP_TAG)
        cfg_local.setdefault("mode", "none")
        super().__init__(cfg_local)

        dataset = str(self.cfg.get("dataset") or "")
        mode = str(self.cfg.get("mode") or "none")
        defaults = _default_perturb_cfg(dataset)
        overrides = _extract_user_perturb_cfg(self.cfg, dataset)
        perturb_cfg = _merge_perturb_cfg(defaults, overrides)
        self.model = _W3PerturbWrapper(self.model, mode=mode, dataset=dataset, perturb_cfg=perturb_cfg)

        self._last_hooks_data: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}

    def _create_model(self):
        """BaseExperiment 요구사항: 모델 인스턴스 생성."""
        return HybridTS(self.cfg, self.n_vars)

    def _ctx(self, dataset: str, horizon: int, seed: int, mode: str,
             model_tag: str, run_tag: str, plot_type: str = "") -> Dict[str, Any]:
        return {
            "experiment_type": EXP_TAG,
            "dataset": dataset,
            "plot_type": plot_type,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": str(mode),
            "model_tag": str(model_tag),
            "run_tag": str(run_tag),
        }

    def evaluate_test(self) -> Dict[str, Any]:
        try:
            tod_vec = build_test_tod_vector(self.cfg)
        except Exception:
            tod_vec = None

        direct = evaluate_with_direct_evidence(
            self.model,
            self.test_loader,
            self.mu,
            self.std,
            tod_vec=tod_vec,
            device=self.device,
            collect_gate_outputs=True,
        )

        hooks = direct.pop("hooks_data", None) or {}

        try:
            exp_specific = compute_all_experiment_metrics(
                experiment_type=EXP_TAG,
                model=self.model,
                hooks_data=hooks if hooks else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=self.cfg.get("mode", "none"),
            )
            if exp_specific:
                direct.update(exp_specific)
        except Exception:
            pass

        self._last_hooks_data = copy.deepcopy(hooks)
        self._last_direct = copy.deepcopy(direct)
        return direct

    def _after_eval_save(
        self,
        dataset: str,
        horizon: int,
        seed: int,
        mode: str,
        direct: Dict[str, Any],
        hooks: Dict[str, Any],
        model_tag: str,
        run_tag: str,
    ) -> None:
        ctx = self._ctx(dataset, horizon, seed, mode, model_tag, run_tag)

        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        heatmap_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
        dist_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
        forest_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        dist_dir.mkdir(parents=True, exist_ok=True)

        mse_real = direct.get("mse_real")
        rmse_direct = direct.get("rmse")
        try:
            rmse_real = float(rmse_direct) if rmse_direct is not None else float(np.sqrt(float(mse_real)))
        except Exception:
            rmse_real = float("nan")

        mae_direct = direct.get("mae", direct.get("mae_real"))
        try:
            mae_real = float(mae_direct) if mae_direct is not None else float("nan")
        except Exception:
            mae_real = float("nan")

        save_w3_forest_detail_row(
            {**ctx, "plot_type": "forest_plot"},
            rmse_real=rmse_real,
            mae_real=mae_real,
            detail_csv=forest_dir / "forest_plot_detail.csv",
        )

        try:
            tod_vec = build_test_tod_vector(self.cfg)
        except Exception:
            tod_vec = None

        if tod_vec is not None:
            compute_and_save_w3_gate_tod_heatmap(
                {**ctx, "plot_type": "gate_tod_heatmap"},
                hooks_data=hooks,
                tod_vec=tod_vec,
                dataset=dataset,
                out_dir=heatmap_dir,
                amp_method="range",
            )
        compute_and_save_w3_gate_distribution(
            {**ctx, "plot_type": "gate_distribution"},
            hooks_data=hooks,
            out_dir=dist_dir,
        )

        results_row: Dict[str, Any] = {
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "mode": mode,
            "model_tag": model_tag,
            "experiment_type": EXP_TAG,
            "mse_real": float(mse_real) if mse_real is not None else np.nan,
            "rmse": rmse_real,
            "mae": mae_real,
        }

        for key, value in direct.items():
            if key in results_row:
                continue
            try:
                results_row[key] = float(value)
            except Exception:
                results_row[key] = value

        _append_or_update_results(
            RESULTS_ROOT / f"results_{EXP_TAG}.csv",
            results_row,
            subset=["dataset", "horizon", "seed", "mode", "model_tag", "experiment_type"],
        )

        perturb_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar"
        perturb_dir.mkdir(parents=True, exist_ok=True)

        rebuild_w3_forest_summary(
            detail_csv=forest_dir / "forest_plot_detail.csv",
            summary_csv=forest_dir / "forest_plot_summary.csv",
            gate_tod_summary_csv=heatmap_dir / "gate_tod_heatmap_summary.csv",
            group_keys=("experiment_type", "dataset", "horizon"),
        )
        rebuild_w3_perturb_delta_bar(
            results_csv=RESULTS_ROOT / f"results_{EXP_TAG}.csv",
            detail_csv=perturb_dir / "perturb_delta_bar_detail.csv",
            summary_csv=perturb_dir / "perturb_delta_bar_summary.csv",
        )

    def run(self):
        dataset = str(self.cfg.get("dataset") or getattr(self, "dataset_tag", ""))
        horizon = int(self.cfg.get("horizon", 96))
        seed = int(self.cfg.get("seed", 42))
        mode = str(self.cfg.get("mode", "none"))
        model_tag = getattr(self.model, "model_tag", "HyperConv")
        run_tag = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}"

        try:
            self.train()
            direct = self.evaluate_test()
            hooks = self._last_hooks_data
            self._after_eval_save(dataset, horizon, seed, mode, direct, hooks, model_tag, run_tag)
        finally:
            self.cleanup()

