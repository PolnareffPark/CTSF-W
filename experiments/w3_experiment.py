# ============================================================
# experiments/w3_experiment.py
#   - W3: 데이터 구조 원인 확인 (학습-교란 / 평가-클린)
#   - none | tod_shift | smooth
#   - 결과 저장: results/results_W3.csv (+ 3 그림 summary/detail)
# ============================================================

from __future__ import annotations
import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .base_experiment import BaseExperiment
from data.dataset import build_test_tod_vector
from models.ctsf_model import HybridTS
from utils.direct_evidence import evaluate_with_direct_evidence

from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    compute_and_save_w3_gate_tod_heatmap,
    compute_and_save_w3_gate_distribution,
    save_w3_forest_detail_row,
    rebuild_w3_forest_summary,
    rebuild_w3_perturb_delta_bar,
)

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"
PERTURB_MODES = ("none", "tod_shift", "smooth")


# ---------------------------
# 내부 유틸
# ---------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _append_or_update_results(csv_path: str | Path, row: Dict[str, Any], subset: List[str]) -> None:
    import pandas as pd
    path = _ensure_dir(csv_path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # 컬럼 union
        for c in new.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in new.columns:
                new[c] = np.nan
        df = pd.concat([old, new], ignore_index=True)
        df.drop_duplicates(subset=subset, keep="last", inplace=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)
    else:
        new.drop_duplicates(subset=subset, keep="last", inplace=True)
        new.to_csv(path, index=False)


def _infer_tod_bins(dataset: str) -> int:
    ds = (dataset or "").lower()
    if "ettm" in ds:
        return 96
    if "etth" in ds:
        return 24
    if "weather" in ds:
        return 144
    return 24


# ---------------------------
# 학습-시행시 입력 교란 래퍼
# ---------------------------
class _W3PerturbWrapper(nn.Module):
    """
    학습시에만 입력(첫 번째 텐서)을 교란하고, 평가/테스트에선 그대로 통과.
    모델 시그니처는 (*args, **kwargs) 그대로 유지. 입력 텐서 시간축을 자동 감지.
    """
    def __init__(self, base: nn.Module, cfg: Dict[str, Any]):
        super().__init__()
        self.base = base
        self.cfg = cfg
        self.mode = str(cfg.get("perturbation", "none"))

        # 교란 강도(CTSF 012 분석 요지 반영)
        ds = str(cfg.get("dataset", "")).lower()
        if self.mode == "tod_shift":
            # ETTm2: 15min → 1~2h = 4~8 스텝; 약간 강화해서 4~12
            self.roll_min, self.roll_max = (4, 12) if "ettm2" in ds else (1, 4)
        elif self.mode == "smooth":
            # ETTh2: 피크 완화(5-tap, 부드럽게, 피크 감소 ~15~25%)
            self.smooth_kernel = torch.tensor([1, 4, 6, 4, 1], dtype=torch.float32)
            self.smooth_kernel = self.smooth_kernel / self.smooth_kernel.sum()
        else:
            self.roll_min, self.roll_max = (0, 0)
            self.smooth_kernel = None

        self.seq_len = int(cfg.get("seq_len", 0))

    def _find_time_axis(self, x: torch.Tensor) -> int:
        # seq_len 힌트가 있으면 거기에 맞춤. 없으면 가장 긴 축(배치축 제외) 추정.
        if self.seq_len > 0:
            for ax, sz in enumerate(x.shape):
                if sz == self.seq_len:
                    return ax
        # B, T, C  or B, C, T 가 일반적
        # 배치축(0)을 제외한 가장 큰 축을 시간축으로 가정
        dims = list(range(x.ndim))
        if x.ndim >= 3:
            candidates = [ax for ax in dims[1:] if x.shape[ax] >= 12]
            if len(candidates) > 0:
                # 가장 긴 축
                return max(candidates, key=lambda a: x.shape[a])
        return x.ndim - 1  # 마지막 축 fall-back

    @torch.no_grad()
    def _apply_tod_shift(self, x: torch.Tensor) -> torch.Tensor:
        if self.roll_max <= 0:
            return x
        t_ax = self._find_time_axis(x)
        # (B,...,T,...) 형태로 bring
        perm = list(range(x.ndim))
        if t_ax != -1 and t_ax != x.ndim - 1:
            perm[t_ax], perm[-1] = perm[-1], perm[t_ax]
            x = x.permute(*perm)
        B, *rest, T = x.shape
        if B <= 0 or T <= 2:
            # 되돌려 놓고 반환
            if t_ax != -1 and t_ax != x.ndim - 1:
                inv = [0]*len(perm)
                for i,p in enumerate(perm): inv[p]=i
                x = x.permute(*inv)
            return x

        # 배치별로 다른 roll
        shifts = torch.randint(low=self.roll_min, high=self.roll_max+1, size=(B,), device=x.device)
        signs  = torch.randint(low=0, high=2, size=(B,), device=x.device) * 2 - 1  # {-1, +1}
        svec   = (shifts * signs).tolist()
        xs = []
        for b in range(B):
            xs.append(torch.roll(x[b], shifts=int(svec[b]), dims=-1))
        x2 = torch.stack(xs, dim=0)

        # 원래 축으로 복귀
        if t_ax != -1 and t_ax != x.ndim - 1:
            inv = [0]*len(perm)
            for i,p in enumerate(perm): inv[p]=i
            x2 = x2.permute(*inv)
        return x2

    @torch.no_grad()
    def _apply_smooth(self, x: torch.Tensor) -> torch.Tensor:
        if self.smooth_kernel is None or x.ndim < 3:
            return x
        t_ax = self._find_time_axis(x)
        perm = list(range(x.ndim))
        if t_ax != -1 and t_ax != x.ndim - 1:
            perm[t_ax], perm[-1] = perm[-1], perm[t_ax]
            x = x.permute(*perm)
        # 이제 (..., T) 가 마지막 축
        # Conv1d를 쓰려면 (B*C,1,T) 형식으로
        B = x.shape[0]
        T = x.shape[-1]
        C = int(np.prod(x.shape[1:-1])) if x.ndim > 3 else x.shape[-2]
        x_ = x.reshape(B, C, T)

        k = self.smooth_kernel.to(x_.device, dtype=x_.dtype).view(1, 1, -1)
        pad = k.shape[-1] // 2
        x_pad = torch.nn.functional.pad(x_, (pad, pad), mode="reflect")
        # depthwise(채널별)로 동일 커널 적용
        w = k.expand(C, 1, -1)
        x_sm = torch.nn.functional.conv1d(x_pad, w, bias=None, stride=1, padding=0, groups=C)
        x_sm = x_sm.reshape(*x.shape)

        if t_ax != -1 and t_ax != x.ndim - 1:
            inv = [0]*len(perm)
            for i,p in enumerate(perm): inv[p]=i
            x_sm = x_sm.permute(*inv)
        return x_sm

    def forward(self, *args, **kwargs):
        if not self.training or self.mode == "none":
            return self.base(*args, **kwargs)

        # 첫 번째 텐서형 인수 찾아서 변환
        args = list(args)
        for i, a in enumerate(args):
            if torch.is_tensor(a) and a.ndim >= 3:
                x = a
                if self.mode == "tod_shift":
                    x = self._apply_tod_shift(x)
                elif self.mode == "smooth":
                    x = self._apply_smooth(x)
                args[i] = x
                break
        return self.base(*args, **kwargs)

    def __getattr__(self, name):
        """base 모델의 속성에 투명하게 접근 (무한 재귀 방지)"""
        # object.__getattribute__를 사용하여 _modules['base']에 직접 접근
        # 이는 __getattr__를 우회하므로 무한 재귀를 방지함
        try:
            # nn.Module은 서브모듈을 _modules dict에 저장
            modules = object.__getattribute__(self, '_modules')
            base = modules['base']
        except (AttributeError, KeyError):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # base 모델의 속성으로 위임
        try:
            return getattr(base, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


# ---------------------------
# 실험 본체
# ---------------------------
class W3Experiment(BaseExperiment):
    """
    W3 = 데이터 구조 원인 검증 (학습 교란 / 평가 클린)
    mode ∈ {none, tod_shift, smooth}
    """

    def __init__(self, cfg: Dict[str, Any]):
        cfg_local = copy.deepcopy(cfg or {})
        cfg_local.setdefault("experiment_type", EXP_TAG)
        if cfg_local.get("mode") not in PERTURB_MODES:
            cfg_local["mode"] = "none"
        # _W3PerturbWrapper가 사용하는 perturbation도 설정
        if "perturbation" not in cfg_local:
            cfg_local["perturbation"] = cfg_local.get("mode", "none")
        super().__init__(cfg_local)
        self._last_hooks_data: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}

        # 학습 시 forward 교란 패치(평가 시에는 꺼짐)
        self.model = _W3PerturbWrapper(self.model, self.cfg)

    # BaseExperiment 가 요구하는 모델 생성
    def _create_model(self):
        # W1/W2와 동일한 백본(수정 금지 원칙)
        return HybridTS(self.cfg, self.n_vars)

    # ------------- 평가 1회 -------------
    def evaluate_test(self) -> Dict[str, Any]:
        # (평가는 깨끗한 데이터; 교란은 wrapper가 training시에만 적용)
        tod_vec = build_test_tod_vector(self.cfg)
        direct = evaluate_with_direct_evidence(
            self.model,
            self.test_loader,
            self.mu,
            self.std,
            tod_vec=tod_vec,
            device=self.device,
            collect_gate_outputs=True,
        )

        hooks_data = {}
        w3_hooks = direct.pop("hooks_data", None)
        if w3_hooks:
            hooks_data.update(w3_hooks or {})

        self._last_hooks_data = copy.deepcopy(hooks_data) if hooks_data else {}
        self._last_direct = copy.deepcopy(direct) if direct else {}
        return direct

    # ------------- 저장/그림 -------------
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
        ctx = {
            "experiment_type": EXP_TAG,
            "dataset": dataset,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": str(mode),
            "model_tag": str(model_tag),
            "run_tag": str(run_tag),
        }

        # 1) forest raw(detail)
        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))
        fdir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        fdir.mkdir(parents=True, exist_ok=True)
        save_w3_forest_detail_row(
            {**ctx, "plot_type": "forest_plot"},
            rmse_real=rmse_real, mae_real=mae_real,
            detail_csv=fdir / "forest_plot_detail.csv",
        )

        # 2) gate heatmap / distribution (hooks + tod_vec 기반)
        tod_vec = build_test_tod_vector(self.cfg)
        hdir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
        hdir.mkdir(parents=True, exist_ok=True)
        compute_and_save_w3_gate_tod_heatmap({**ctx, "plot_type": "gate_tod_heatmap"},
                                             hooks_data=hooks, tod_vec=tod_vec, dataset=dataset, out_dir=hdir)

        ddir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
        ddir.mkdir(parents=True, exist_ok=True)
        compute_and_save_w3_gate_distribution({**ctx, "plot_type": "gate_distribution"},
                                              hooks_data=hooks, out_dir=ddir)

        # 3) results_W3.csv 1행 추가(직접근거 + 게이트 요약 일부 동시 저장)
        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed, "mode": mode,
            "model_tag": model_tag, "experiment_type": EXP_TAG,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": float(rmse_real), "mae": float(mae_real),
            # direct evidence 주요 지표 (있으면 저장)
            "cg_pearson_mean": direct.get("cg_pearson_mean", np.nan),
            "cg_spearman_mean": direct.get("cg_spearman_mean", np.nan),
            "cg_dcor_mean": direct.get("cg_dcor_mean", np.nan),
            "cg_event_gain": direct.get("cg_event_gain", np.nan),
            "cg_event_hit": direct.get("cg_event_hit", np.nan),
            "cg_maxcorr": direct.get("cg_maxcorr", np.nan),
            "cg_bestlag": direct.get("cg_bestlag", np.nan),
            "gc_kernel_tod_dcor": direct.get("gc_kernel_tod_dcor", np.nan),
            "gc_feat_tod_dcor": direct.get("gc_feat_tod_dcor", np.nan),
            "gc_feat_tod_r2": direct.get("gc_feat_tod_r2", np.nan),
            "gc_kernel_feat_dcor": direct.get("gc_kernel_feat_dcor", np.nan),
            "gc_kernel_feat_align": direct.get("gc_kernel_feat_align", np.nan),
        }
        _append_or_update_results(RESULTS_ROOT / f"results_{EXP_TAG}.csv",
                                  row, subset=["dataset","horizon","seed","mode","model_tag","experiment_type"])

        # 4) 집계: forest Δ / perturb delta‑bar (idempotent)
        rebuild_w3_forest_summary(
            detail_csv=fdir / "forest_plot_detail.csv",
            summary_csv=fdir / "forest_plot_summary.csv",
            gate_tod_summary_csv=hdir / "gate_tod_heatmap_summary.csv",
            group_keys=("experiment_type","dataset","horizon")
        )
        rebuild_w3_perturb_delta_bar(
            results_csv=RESULTS_ROOT / f"results_{EXP_TAG}.csv",
            detail_csv=RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar" / "perturb_delta_bar_detail.csv",
            summary_csv=RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar" / "perturb_delta_bar_summary.csv",
        )

    # ------------- 실행 진입점 -------------
    def run(self):
        """
        main.py와 run_suite.py에서 호출됨.
        """
        # dataset은 cfg["dataset"] 또는 self.dataset_tag 사용
        dataset = str(self.cfg.get("dataset") or self.dataset_tag)
        horizon = int(self.cfg.get("horizon", 96))
        seed    = int(self.cfg.get("seed", 42))
        mode    = str(self.cfg.get("mode", "none"))
        model_tag = getattr(self.model, "model_tag", "HyperConv")
        run_tag   = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}"

        try:
            # 표준 학습 → 검증 → 테스트 (학습 단계에서만 wrapper가 교란 적용)
            self.train()          # BaseExperiment 제공
            direct = self.evaluate_test()
            hooks = self._last_hooks_data

            self._after_eval_save(dataset, horizon, seed, mode, direct, hooks, model_tag, run_tag)
        finally:
            # 반드시 남은 .pt/메모리 정리
            self.cleanup()
