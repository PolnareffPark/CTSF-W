# ============================================================
# utils/experiment_metrics/w3_metrics.py
#   - (선택) baseline vs perturbed 사후 Δ 유틸
# ============================================================
from typing import Dict, Optional
import numpy as np

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
