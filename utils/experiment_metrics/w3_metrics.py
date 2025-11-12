# ============================================================
# utils/experiment_metrics/w3_metrics.py
#   - W3: 데이터 구조 원인 확인 (baseline vs perturbed) Δ/효과크기 계산기 (완성본)
# ============================================================

from __future__ import annotations
from typing import Dict, Optional
import math
import numpy as np

# ---- 내부 유틸 -------------------------------------------------
def _f(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float("nan")
    except Exception:
        return float("nan")

def _delta(p, b) -> float:
    """Δ = perturbed - baseline"""
    pp, bb = _f(p), _f(b)
    if not np.isfinite(pp) or not np.isfinite(bb):
        return float("nan")
    return pp - bb

def _cohens_dz(mean_diff: float, sd_diff: float, n: int) -> float:
    """
    Paired Cohen's dz = mean(diff) / sd(diff)
    - diff = y_pert - y_base (윈도우별)
    """
    if n < 2 or sd_diff <= 0 or not np.isfinite(mean_diff) or not np.isfinite(sd_diff):
        return float("nan")
    return float(mean_diff / sd_diff)

def _ci95_for_dz(dz: float, n: int) -> (float, float):
    """
    dz의 95% CI를 간단 근사치로 제공: dz ± 1.96 / sqrt(n)
    (평균차의 CI를 sd로 나눈 근사; n이 충분히 크면 안정)
    """
    if n < 2 or not np.isfinite(dz):
        return (float("nan"), float("nan"))
    hw = 1.96 / math.sqrt(n)
    return (float(dz - hw), float(dz + hw))

# ---- 본 함수 ---------------------------------------------------
def compute_w3_metrics(
    baseline_metrics: Optional[Dict] = None,
    perturb_metrics: Optional[Dict] = None,
    win_errors_base: Optional[np.ndarray] = None,  # per-window error (baseline)
    win_errors_pert: Optional[np.ndarray] = None,  # per-window error (perturbed)
    dz_ci: bool = False
) -> Dict:
    """
    baseline(=none) vs perturbed(tod_shift|smooth) 한 쌍이 주어졌을 때,
    W3 보고 정의에 맞춰 Δ/효과크기(Cohen's dz)를 산출합니다.

    반환 키(대표):
      - delta_rmse, delta_rmse_pct, delta_mae
      - delta_tod_amp_mean (둘 다 있으면)
      - A4‑TOD 전용:  delta_gc_kernel_tod_dcor, delta_gc_feat_tod_dcor, delta_gc_feat_tod_r2
      - A4‑PEAK 전용: delta_cg_event_gain, delta_abs_cg_bestlag, delta_cg_spearman_mean
      - (옵션) cohens_dz, cohens_dz_ci95_low, cohens_dz_ci95_high, n_windows
    """
    out: Dict[str, float] = {}

    # 안전장치
    if not isinstance(baseline_metrics, dict) or not isinstance(perturb_metrics, dict):
        return out

    # 1) 기본 오차 Δ
    rmse_b = _f(baseline_metrics.get("rmse_real", baseline_metrics.get("rmse")))
    rmse_p = _f(perturb_metrics.get("rmse_real", perturb_metrics.get("rmse")))
    mae_b  = _f(baseline_metrics.get("mae_real", baseline_metrics.get("mae")))
    mae_p  = _f(perturb_metrics.get("mae_real", perturb_metrics.get("mae")))

    if np.isfinite(rmse_b) and np.isfinite(rmse_p):
        out["delta_rmse"] = float(rmse_p - rmse_b)
        out["delta_rmse_pct"] = float((rmse_p - rmse_b) / rmse_b * 100.0) if rmse_b > 0 else float("nan")
    if np.isfinite(mae_b) and np.isfinite(mae_p):
        out["delta_mae"] = float(mae_p - mae_b)

    # 2) TOD 민감도(heatmap 요약; 있으면)
    tod_b = _f(baseline_metrics.get("tod_amp_mean"))
    tod_p = _f(perturb_metrics.get("tod_amp_mean"))
    if np.isfinite(tod_b) and np.isfinite(tod_p):
        out["delta_tod_amp_mean"] = float(tod_p - tod_b)

    # 3) A4‑TOD 지표 Δ (음수 기대)
    for k in ["gc_kernel_tod_dcor", "gc_feat_tod_dcor", "gc_feat_tod_r2"]:
        pb = _f(perturb_metrics.get(k)); bb = _f(baseline_metrics.get(k))
        if np.isfinite(pb) and np.isfinite(bb):
            out[f"delta_{k}"] = float(pb - bb)

    # 4) A4‑PEAK 지표 Δ
    #   - cg_event_gain (음수 기대)
    #   - |cg_bestlag| (양수 기대) → delta_abs_cg_bestlag
    #   - cg_spearman_mean (음수 기대)
    pev = _f(perturb_metrics.get("cg_event_gain")); bev = _f(baseline_metrics.get("cg_event_gain"))
    if np.isfinite(pev) and np.isfinite(bev):
        out["delta_cg_event_gain"] = float(pev - bev)

    p_lag = _f(perturb_metrics.get("cg_bestlag")); b_lag = _f(baseline_metrics.get("cg_bestlag"))
    if np.isfinite(p_lag) and np.isfinite(b_lag):
        out["delta_abs_cg_bestlag"] = float(abs(p_lag) - abs(b_lag))

    p_sp  = _f(perturb_metrics.get("cg_spearman_mean")); b_sp  = _f(baseline_metrics.get("cg_spearman_mean"))
    if np.isfinite(p_sp) and np.isfinite(b_sp):
        out["delta_cg_spearman_mean"] = float(p_sp - b_sp)

    # 5) (옵션) paired Cohen's d(z) for per-window error vector
    #    diff = e_pert - e_base ; dz = mean(diff)/sd(diff)
    if isinstance(win_errors_base, np.ndarray) and isinstance(win_errors_pert, np.ndarray):
        xb = np.asarray(win_errors_base, dtype=float).ravel()
        xp = np.asarray(win_errors_pert, dtype=float).ravel()
        m = np.isfinite(xb) & np.isfinite(xp)
        d = (xp[m] - xb[m]).astype(float)
        n = int(d.size)
        out["n_windows"] = float(n)
        if n >= 2:
            mean_diff = float(np.mean(d))
            sd_diff   = float(np.std(d, ddof=1)) if n > 1 else float("nan")
            dz = _cohens_dz(mean_diff, sd_diff, n)
            out["cohens_dz"] = dz
            if dz_ci and np.isfinite(dz):
                lo, hi = _ci95_for_dz(dz, n)
                out["cohens_dz_ci95_low"]  = lo
                out["cohens_dz_ci95_high"] = hi

    return out
