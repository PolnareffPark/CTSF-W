# ============================================================
# utils/experiment_metrics/w3_metrics.py
#   - (선택) baseline vs perturbed 사후 Δ 유틸
# ============================================================
from typing import Dict, Optional
import numpy as np

def inject_h1h6_side_metrics(current_direct: dict, baseline_direct: dict) -> dict:
    """
    (선택) W3 보조 지표 주입기:
      - convGRU align/phase baseline/pert/delta 키를 current_direct에 보장적으로 세팅
      - ACF@day/SNR_time 키가 이미 evaluate_test에서 세팅됐다면 건드리지 않음
    """
    out = dict(current_direct)

    try:
        ab = baseline_direct.get("cg_pearson_mean", np.nan)
        ap = current_direct.get("cg_pearson_mean", np.nan)
        if "convgru_align_base" not in out:
            out["convgru_align_base"] = float(ab) if ab is not None else np.nan
        if "convgru_align_pert" not in out:
            out["convgru_align_pert"] = float(ap) if ap is not None else np.nan
        if "convgru_align_delta" not in out:
            out["convgru_align_delta"] = (
                float(ap) - float(ab) if np.isfinite(ab) and np.isfinite(ap) else np.nan
            )
    except Exception:
        pass

    try:
        lb = baseline_direct.get("cg_bestlag", np.nan)
        lp = current_direct.get("cg_bestlag", np.nan)
        if "convgru_phase_base" not in out:
            out["convgru_phase_base"] = float(lb) if lb is not None else np.nan
        if "convgru_phase_pert" not in out:
            out["convgru_phase_pert"] = float(lp) if lp is not None else np.nan
        if "convgru_phase_delta" not in out:
            out["convgru_phase_delta"] = (
                float(lp) - float(lb) if np.isfinite(lb) and np.isfinite(lp) else np.nan
            )
    except Exception:
        pass

    return out

def compute_w3_metrics(
    baseline_metrics: Optional[Dict] = None,
    perturb_metrics: Optional[Dict] = None,
    win_errors_base: Optional[np.ndarray] = None,
    win_errors_pert: Optional[np.ndarray] = None,
    dz_ci: bool = False
) -> Dict:
    out: Dict[str, float] = {}
    if not isinstance(baseline_metrics, dict) or not isinstance(perturb_metrics, dict):
        return out
    try:
        rb = float(baseline_metrics.get("rmse_real"))
        rp = float(perturb_metrics.get("rmse_real"))
        out["delta_rmse"] = float(rp - rb)
        out["delta_rmse_pct"] = float(out["delta_rmse"] / rb * 100.0) if rb > 0 else np.nan
    except Exception:
        pass
    try:
        mb = float(baseline_metrics.get("mae_real"))
        mp = float(perturb_metrics.get("mae_real"))
        out["delta_mae"] = float(mp - mb)
    except Exception:
        pass
    return out
