# ============================================================
# utils/experiment_metrics/w3_metrics.py
#   - W3: 데이터 구조 원인 확인 (baseline vs perturbed) 보조 계산기
#     * 기본적으로 그림 집계(rebuild_*)에서 Δ를 계산하며,
#       본 모듈은 필요시(둘 다 주어질 때) 동일 정의로 Δ를 산출.
# ============================================================

from __future__ import annotations
from typing import Dict, Optional
import numpy as np

def _as_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except Exception:
        return np.nan

def _delta(a, b):  # perturbed - baseline
    A = _as_float(a); B = _as_float(b)
    if np.isnan(A) or np.isnan(B):
        return np.nan
    return A - B

def compute_w3_metrics(
    baseline_metrics: Optional[Dict] = None,
    perturb_metrics: Optional[Dict] = None,
    win_errors_base: Optional[np.ndarray] = None,
    win_errors_pert: Optional[np.ndarray] = None,
    dz_ci: bool = False,
    **kwargs
) -> Dict:
    """
    W3 페어 전용 Δ 계산기(옵션). 둘 다 없으면 {} 반환.
      - ΔRMSE, ΔMAE
      - A4-TOD: Δ(gc_kernel_tod_dcor), Δ(gc_feat_tod_dcor), Δ(gc_feat_tod_r2)
      - A4-PEAK: Δ(cg_event_gain), Δ(|cg_bestlag|), Δ(cg_spearman_mean)
      - (선택) 윈도우 오차의 Cohen's d (win_errors_* 제공 시)
    """
    if not isinstance(baseline_metrics, dict) or not isinstance(perturb_metrics, dict):
        return {}

    out: Dict[str, float] = {}
    # 공통 오차 Δ
    out["w3_delta_rmse"] = _delta(perturb_metrics.get("rmse"), baseline_metrics.get("rmse"))
    out["w3_delta_mae"]  = _delta(perturb_metrics.get("mae"),  baseline_metrics.get("mae"))

    # TOD 계열 (A4-TOD)
    for k in ["gc_kernel_tod_dcor", "gc_feat_tod_dcor", "gc_feat_tod_r2"]:
        out[f"w3_delta_{k}"] = _delta(perturb_metrics.get(k), baseline_metrics.get(k))

    # PEAK 계열 (A4-PEAK)
    out["w3_delta_cg_event_gain"]   = _delta(perturb_metrics.get("cg_event_gain"),
                                             baseline_metrics.get("cg_event_gain"))
    # |bestlag|
    def _absd(a, b):
        A = _as_float(a); B = _as_float(b)
        if np.isnan(A) or np.isnan(B): return np.nan
        return abs(A) - abs(B)
    out["w3_delta_abs_cg_bestlag"]  = _absd(perturb_metrics.get("cg_bestlag"),
                                            baseline_metrics.get("cg_bestlag"))
    out["w3_delta_cg_spearman_mean"] = _delta(perturb_metrics.get("cg_spearman_mean"),
                                              baseline_metrics.get("cg_spearman_mean"))

    # (선택) Cohen’s d (윈도우 오차 시퀀스가 있을 때만)
    if isinstance(win_errors_base, np.ndarray) and isinstance(win_errors_pert, np.ndarray):
        xb = win_errors_base[np.isfinite(win_errors_base)]
        xp = win_errors_pert[np.isfinite(win_errors_pert)]
        if xb.size >= 3 and xp.size >= 3:
            mb, mp = float(np.mean(xb)), float(np.mean(xp))
            sb, sp = float(np.std(xb, ddof=1)), float(np.std(xp, ddof=1))
            # 풀드 표준편차
            n1, n2 = xb.size, xp.size
            s_pooled = np.sqrt(((n1-1)*sb*sb + (n2-1)*sp*sp) / max(1, (n1+n2-2)))
            out["w3_cohens_d"] = float((mp-mb) / s_pooled) if s_pooled > 0 else np.nan

    return out
