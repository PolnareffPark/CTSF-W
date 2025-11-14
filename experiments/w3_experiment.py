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
            "tod_shift": {"max_shift": 16, "seg_perm_k": 4, "per_channel_jitter": 2, "sdiff_beta": 0.15},
            "smooth":    {"kernel": 7,  "alpha": 0.5},  # ETTm2는 smooth 가 보조 역할
        }
    if "etth2" in ds:
        return {
            "tod_shift": {"max_shift": 4, "seg_perm_k": 3, "per_channel_jitter": 1}, # ETTh2 는 tod_shift 가 보조 역할할
            "smooth":    {"kernel": 7,  "alpha": 0.6},  # 피크 완화 강화
        }
    # 기타 데이터셋 (ETTm1, ETTh1, weather 등)
    tod_bins = _infer_tod_bins(dataset)
    return {
        "tod_shift": {"max_shift": max(1, tod_bins // 48), "seg_perm_k": 0, "per_channel_jitter": 0, "sdiff_beta": 0.0},
        "smooth":    {"kernel": 5, "alpha": 0.3},
    }


def _detect_layout(x: torch.Tensor) -> Tuple[int, int]:
    """
    입력 텐서의 시간축/채널축을 추정.
    허용 형태: (B, L, C) 혹은 (B, C, L)
    반환: (time_axis, channel_axis)
    
    일반적으로 시계열 길이(L)가 채널 수(C)보다 크다고 가정.
    L == C인 경우 (B, L, C)로 해석합니다.
    """
    if x.dim() != 3:
        raise ValueError("W3 wrapper expects 3D input (B, L, C) or (B, C, L).")
    _, a1, a2 = x.shape
    if a1 >= a2:
        return 1, 2  # (B, L, C)
    return 2, 1  # (B, C, L)


# def _moving_avg_1d(x: torch.Tensor, k: int, t_axis: int) -> torch.Tensor:
#     """시간축 기준 이동평균을 적용. (현재 미사용)"""
#     if k <= 1:
#         return x
#     if k % 2 == 0:
#         k += 1  # 홀수 강제
#
#     if t_axis == 1:  # (B, L, C)
#         x_nc_t = x.permute(0, 2, 1)  # (B, C, L)
#     else:
#         x_nc_t = x  # (B, C, L)
#
#     b, c, L = x_nc_t.shape
#     weight = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
#     x_flat = x_nc_t.reshape(b * c, 1, L)
#     pad = k // 2
#     x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
#     y = F.conv1d(x_pad, weight).reshape(b, c, L)
#
#     if t_axis == 1:
#         return y.permute(0, 2, 1)
#     return y


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
        self.add_module("base", base)
        self.mode = str(mode or "none")
        self.dataset = str(dataset or "")
        
        # 기본값 가져오기
        defaults = _default_perturb_cfg(dataset)
        
        # 사용자 제공 값이 있으면 복사 후 기본값으로 채우기, 없으면 기본값 사용
        if isinstance(perturb_cfg, dict):
            self.cfg = perturb_cfg.copy()
            for key, value in defaults.items():
                self.cfg.setdefault(key, value)
        else:
            self.cfg = defaults

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
        """
        래퍼에 없는 속성은 항상 내부 base 모듈로 위임.
        단, 'base' 자체를 요청하면 base 모듈을 직접 반환.
        """
        modules = object.__getattribute__(self, "_modules")
        base = modules.get("base")
        if base is None:
            raise AttributeError(
                f"{self.__class__.__name__} has no registered 'base' submodule"
            )
        # 'base' 속성 자체를 요청하면 base 모듈을 반환
        if name == "base":
            return base
        # 그 외 속성은 base에 위임
        return getattr(base, name)

    def unwrap(self) -> nn.Module:
        """내부 원본 모델을 반환."""
        return self.base  # __getattr__을 통해 안전하게 접근

    @torch.no_grad()
    def _apply_tod_shift(self, x: torch.Tensor) -> torch.Tensor:
        # ---- 강화된 정합 붕괴형 tod_shift ----
        t_axis, c_axis = _detect_layout(x)               # (B,L,C) 또는 (B,C,L)
        L = x.shape[t_axis]
        if L < 8:                                        # 너무 짧으면 skip
            return x

        cfg = self.cfg.get("tod_shift", {}) or {}
        max_shift = int(cfg.get("max_shift", 8))         # ETTm2: 8→12~24 권장
        seg_k = int(cfg.get("seg_perm_k", 4))            # 세그먼트 수(4 권장)
        ch_jitter = int(cfg.get("per_channel_jitter", 2))# 채널별 ±jitter(0~2)
        mix_beta = float(cfg.get("sdiff_beta", 0.0))     # 0.0~0.25 권장

        # 1) 기본 원형 roll(전구간)
        if max_shift > 0:
            shift = torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=t_axis)

        # 2) 구간 퍼뮤테이션(세그먼트 순열)
        if seg_k > 1 and L >= seg_k:
            seg_len = L // seg_k
            remainder = L % seg_k
            
            # 세그먼트 경계 계산
            segments = []
            start = 0
            for i in range(seg_k):
                end = start + seg_len + (1 if i < remainder else 0)
                segments.append((start, end))
                start = end
            
            # 무작위 순열
            order = torch.randperm(seg_k, device=x.device).tolist()
            
            # 세그먼트 재배열
            if t_axis == 1:  # (B, L, C)
                chunks = [x[:, segments[i][0]:segments[i][1], :] for i in order]
                x = torch.cat(chunks, dim=1)
            else:  # (B, C, L)
                chunks = [x[:, :, segments[i][0]:segments[i][1]] for i in order]
                x = torch.cat(chunks, dim=2)

        # 3) 채널별 미세 시프트(동조성 약화)
        if ch_jitter > 0:
            # 각 채널 별로 독립 시프트
            if t_axis == 1:   # (B,L,C) → (B,C,L)
                x_nc_t = x.permute(0, 2, 1)
            else:
                x_nc_t = x  # (B,C,L)
            B, C, LL = x_nc_t.shape
            if C > 1:  # 채널이 1개 이상일 때만 적용
                j = torch.randint(-ch_jitter, ch_jitter + 1, (C,), device=x.device)
                for c in range(C):
                    s = int(j[c].item())
                    if s != 0:
                        x_nc_t[:, c, :] = torch.roll(x_nc_t[:, c, :], shifts=s, dims=-1)
            x = x_nc_t.permute(0, 2, 1) if t_axis == 1 else x_nc_t

        # 4) 약한 seasonal differencing 혼합(선택)
        if mix_beta > 1e-8 and L > self.tod_bins:
            # mix_beta 클램핑 (0.0 ~ 0.3 권장)
            mix_beta = min(mix_beta, 0.3)
            P = self.tod_bins  # ETTm=96 / ETTh=24
            x = (1.0 - mix_beta) * x + mix_beta * (x - torch.roll(x, shifts=P, dims=t_axis))
        return x


    @torch.no_grad()
    def _apply_smooth(self, x: torch.Tensor) -> torch.Tensor:
        # ---- 피크/급변 완화 강화형 smooth ----
        t_axis, _ = _detect_layout(x)
        cfg = self.cfg.get("smooth", {}) or {}
        k = int(cfg.get("kernel", 7))                    # 7~9 권장
        alpha = float(cfg.get("alpha", 0.6))             # 0.5~0.7 권장
        if k <= 1 or alpha <= 0:
            return x
        if k % 2 == 0: k += 1

        # (A) 삼각 커널로 부드럽게
        half = k // 2
        # float32로 커널 생성 후 정규화
        tri = torch.arange(1, half + 2, device=x.device, dtype=torch.float32)
        weight = torch.cat([tri, tri.flip(0)[1:]]).view(1, 1, -1)  # [1,2,3,2,1] 형태
        weight = weight / weight.sum()  # 정규화

        # conv1d 적용을 위해 (B,C,L) 형태로 맞춤
        if t_axis == 1:  # (B,L,C) → (B,C,L)
            x_nc_t = x.permute(0, 2, 1)
        else:
            x_nc_t = x
        B, C, L = x_nc_t.shape
        x_flat = x_nc_t.reshape(B*C, 1, L)
        pad = k // 2
        x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
        y = F.conv1d(x_pad, weight).reshape(B, C, L)

        # (B) 블렌딩
        y = (1.0 - alpha) * x_nc_t + alpha * y
        return y.permute(0, 2, 1) if t_axis == 1 else y

def _unwrap_model(m: nn.Module) -> nn.Module:
    """
    _W3PerturbWrapper / DataParallel / DDP 등 래핑을 벗겨 순수 베이스 모델을 얻는다.
    """
    cur = m
    while True:
        if isinstance(cur, _W3PerturbWrapper):
            cur = cur.unwrap()
            continue
        if hasattr(cur, "module"):  # DP/DDP 호환
            cur = cur.module
            continue
        break
    return cur

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
        self._last_tod_vec: Optional[np.ndarray] = None

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
            self._last_tod_vec = tod_vec  # 저장
        except Exception:
            tod_vec = None
            self._last_tod_vec = None

        # 언랩된 모델
        model_for_analysis = _unwrap_model(self.model)

        direct = evaluate_with_direct_evidence(
            model_for_analysis,        # <-- 언랩된 모델 전달
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
                model=model_for_analysis,   # <-- 언랩된 모델 전달
                hooks_data=hooks if hooks else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=self.cfg.get("mode", "none"),
            )
            if exp_specific:
                direct.update(exp_specific)
        except Exception as e:
            import warnings
            warnings.warn(f"W3 compute_all_experiment_metrics failed: {e}")

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

        # 디렉터리 준비
        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        heatmap_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
        dist_dir   = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
        forest_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        dist_dir.mkdir(parents=True, exist_ok=True)

        # 스칼라 성능치 정리 (rmse_real/mae_real)
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

        # forest detail (각 run의 실측 RMSE/MAE 기록)
        save_w3_forest_detail_row(
            {**ctx, "plot_type": "forest_plot"},
            rmse_real=rmse_real,
            mae_real=mae_real,
            detail_csv=forest_dir / "forest_plot_detail.csv",
        )

        # Gate‑TOD Heatmap / Gate 분포
        tod_vec = self._last_tod_vec  # 재사용

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

        # results_W3.csv에 넣을 기본 행
        results_row: Dict[str, Any] = {
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "mode": mode,
            "model_tag": model_tag,
            "experiment_type": EXP_TAG,
            "mse_real": float(mse_real) if mse_real is not None else np.nan,
            "rmse": rmse_real,
            "mae":  mae_real,
        }

        # --- win_errors: summary만 results에 기록, 원본 분포는 별도 detail로 저장 ---
        we = direct.get("win_errors", None)
        if isinstance(we, (list, tuple, np.ndarray)) and len(we) > 0:
            w = np.asarray(we, float)
            w = w[np.isfinite(w)]
            if w.size > 0:
                results_row["win_rmse_mean"] = float(np.nanmean(w))
                results_row["win_rmse_std"]  = float(np.nanstd(w))
                results_row["win_rmse_p95"]  = float(np.nanpercentile(w, 95))
                results_row["win_rmse_max"]  = float(np.nanmax(w))
                # run 별 원본 분포 저장 (append 모드)
                win_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "window_errors"
                win_dir.mkdir(parents=True, exist_ok=True)
                win_csv = win_dir / "window_errors_detail.csv"
                
                dfw = pd.DataFrame({
                    "dataset": [dataset]*w.size,
                    "horizon": [horizon]*w.size,
                    "seed": [seed]*w.size,
                    "mode": [mode]*w.size,
                    "model_tag": [model_tag]*w.size,
                    "idx": np.arange(w.size),
                    "rmse": w
                })
                
                # append 또는 create
                if win_csv.exists():
                    old = pd.read_csv(win_csv)
                    merged = pd.concat([old, dfw], ignore_index=True)
                    # 같은 설정의 이전 run 제거
                    subset = ["dataset", "horizon", "seed", "mode", "model_tag", "idx"]
                    merged.drop_duplicates(subset=subset, keep="last", inplace=True)
                    merged.to_csv(win_csv, index=False)
                else:
                    dfw.to_csv(win_csv, index=False)
            # CSV에는 raw vector를 넣지 않음
            direct.pop("win_errors", None)

        # direct 나머지 스칼라/문자열 합류
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

        # 요약/Δ 리빌드 (데이터셋 폴더에 자신의 Δ만 쓰게 하려면 filter_dataset 인자 사용)
        perturb_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar"
        perturb_dir.mkdir(parents=True, exist_ok=True)

        rebuild_w3_forest_summary(
            detail_csv=forest_dir / "forest_plot_detail.csv",
            summary_csv=forest_dir / "forest_plot_summary.csv",
            gate_tod_summary_csv=heatmap_dir / "gate_tod_heatmap_summary.csv",
            group_keys=("experiment_type", "dataset", "horizon"),
        )

        # w3_plotting_metrics.rebuild_w3_perturb_delta_bar가 filter_dataset 인자를 받도록
        rebuild_w3_perturb_delta_bar(
            results_csv=RESULTS_ROOT / f"results_{EXP_TAG}.csv",
            detail_csv=perturb_dir / "perturb_delta_bar_detail.csv",
            summary_csv=perturb_dir / "perturb_delta_bar_summary.csv",
            filter_dataset=dataset,
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

