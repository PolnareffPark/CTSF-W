# ============================================================
# utils/experiment_metrics/w3_metrics.py
#   - W3: 원인 교란(perturbation) 전/후 쌍 비교 지표 + 단일 조건 보조 통계
#   - W1/W2 의존성 없음. (utils.metrics._dcor_u 사용 가능)
# ============================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats

try:
    # 존재 시 거리상관 사용(W2와 동일 경로)
    from utils.metrics import _dcor_u  # optional
except Exception:
    _dcor_u = None  # 폴백: None이면 TOD 정렬 Δ는 NaN

# ---------- 내부 유틸 ----------
def _as_np(a):
    try:
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        return np.asarray(a)
    except Exception:
        return None

def _safe_entropy(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x) & (x > 1e-12)]
    if x.size == 0:
        return float("nan")
    p = x / (x.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

def _safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]):
        return 0.0
    try:
        return float(stats.kurtosis(x, fisher=True, bias=False, nan_policy="omit"))
    except Exception:
        return float("nan")

def _hoyer_sparsity(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    l1 = float(np.sum(x))
    l2 = float(np.linalg.norm(x))
    if l2 == 0.0:
        return 0.0
    return float((np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1.0 + 1e-12))

def _concat_gate_per_sample(hooks: Optional[Dict]) -> Optional[np.ndarray]:
    """
    hooks에서 gate_outputs를 (N,L,G) 또는 (N,...) 형태로 연결해 (N, -1) 강도 벡터를 반환.
    수집 실패 시 None.
    """
    if not isinstance(hooks, dict):
        return None
    src = None
    if "gate_outputs" in hooks and hooks["gate_outputs"]:
        bags = []
        for a in hooks["gate_outputs"]:
            a = _as_np(a)
            if a is None:
                continue
            a = np.asarray(a, float)
            if a.ndim >= 2:
                bags.append(a.reshape(a.shape[0], -1))
            else:
                bags.append(a.reshape(-1, 1))
        if bags:
            try:
                src = np.concatenate(bags, axis=0)  # (N, F)
            except Exception:
                src = bags[0]
    # stage dict만 있는 경우 평균으로 근사
    if src is None:
        for k in ("gates_by_stage", "gate_by_stage", "gate_logs_by_stage"):
            if k in hooks and isinstance(hooks[k], dict) and hooks[k]:
                arrs = []
                for s in ("S","M","D","s","m","d"):
                    if s in hooks[k] and hooks[k][s] is not None:
                        a = _as_np(hooks[k][s])
                        if a is not None:
                            a = np.asarray(a, float)
                            arrs.append(a.reshape(a.shape[0], -1) if a.ndim >= 2 else a.reshape(-1,1))
                if arrs:
                    try:
                        src = np.concatenate(arrs, axis=1)  # (N, F)
                    except Exception:
                        src = arrs[0]
                break
    return src  # (N,F) or None

def _tod_align_strength(gate_F: Optional[np.ndarray], tod_vec: Optional[np.ndarray]) -> float:
    """
    gate_F: (N,F) → N별 강도 스칼라로 요약(mean) 후 TOD(sin,cos)와 거리상관.
    """
    if _dcor_u is None or gate_F is None or tod_vec is None:
        return float("nan")
    try:
        if tod_vec.ndim == 1:
            ty = tod_vec.reshape(-1, 1)
        else:
            ty = tod_vec.reshape(len(tod_vec), -1)
        g_strength = np.nanmean(gate_F, axis=1).reshape(-1, 1)  # (N,1)
        L = min(g_strength.shape[0], ty.shape[0])
        if L < 3:
            return float("nan")
        return float(_dcor_u(g_strength[:L], ty[:L]))
    except Exception:
        return float("nan")

# ---------- 단일 조건 보조 통계 ----------
def compute_w3_single_metrics(
    hooks_data: Optional[Dict] = None
) -> Dict[str, float]:
    """
    게이트 분포 보조 통계 (있으면 기록, 없으면 NaN).
    - gate_mean, gate_std, gate_entropy, gate_kurtosis, gate_sparsity
    - gate_q05, q10, q25, q50, q75, q90, q95
    """
    m = {
        "gate_mean": float("nan"),
        "gate_std": float("nan"),
        "gate_entropy": float("nan"),
        "gate_kurtosis": float("nan"),
        "gate_sparsity": float("nan"),
    }
    q_keys = ["gate_q05","gate_q10","gate_q25","gate_q50","gate_q75","gate_q90","gate_q95"]
    for k in q_keys:
        m[k] = float("nan")

    G = _concat_gate_per_sample(hooks_data)
    if G is None:
        return m

    flat = G.reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return m

    m["gate_mean"] = float(np.nanmean(finite))
    m["gate_std"]  = float(np.nanstd(finite))
    m["gate_entropy"] = _safe_entropy(finite)
    m["gate_kurtosis"] = _safe_kurtosis(finite)
    m["gate_sparsity"] = _hoyer_sparsity(finite)

    qs = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
    try:
        qv = np.quantile(finite, qs).tolist()
        for k, v in zip(q_keys, qv):
            m[k] = float(v)
    except Exception:
        pass
    return m

# ---------- 쌍 비교(perturbed vs baseline) ----------
def compute_w3_pair_metrics(
    base_direct: Optional[Dict],
    pert_direct: Optional[Dict],
    base_hooks: Optional[Dict] = None,
    pert_hooks: Optional[Dict] = None,
    tod_vec: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    baseline ↔ perturbed 쌍에서 Δ지표를 계산.
    필수: mse_real / rmse / mae (둘 중 하나 있으면 나머지는 보강 계산)
    선택: TOD 정렬(거리상관), direct_evidence의 gc_*(TOD dcor) 컬럼들
    """
    def _get_err(d: Optional[Dict]) -> Tuple[float,float]:
        if not isinstance(d, dict):
            return (float("nan"), float("nan"))
        rmse = d.get("rmse", None)
        mse_real = d.get("mse_real", None)
        if rmse is None and mse_real is not None:
            rmse = float(np.sqrt(float(mse_real)))
        mae = d.get("mae", d.get("mae_real", float("nan")))
        try:
            return (float(rmse), float(mae))
        except Exception:
            return (float("nan"), float("nan"))

    base_rmse, base_mae = _get_err(base_direct)
    pert_rmse, pert_mae = _get_err(pert_direct)

    out = {
        "w3_delta_rmse": float("nan"),
        "w3_delta_rmse_pct": float("nan"),
        "w3_delta_mae": float("nan"),
        "w3_delta_tod_align": float("nan"),
        "w3_delta_gc_kernel_tod_dcor": float("nan"),
        "w3_delta_gc_feat_tod_dcor": float("nan"),
        "w3_delta_gc_feat_tod_r2": float("nan"),
    }

    if np.isfinite(base_rmse) and np.isfinite(pert_rmse):
        out["w3_delta_rmse"] = float(pert_rmse - base_rmse)
        if base_rmse > 0:
            out["w3_delta_rmse_pct"] = float((out["w3_delta_rmse"] / base_rmse) * 100.0)
        out["w3_delta_mae"] = float(pert_mae - base_mae) if np.isfinite(base_mae) and np.isfinite(pert_mae) else float("nan")

    # TOD 정렬(거리상관)의 Δ (가능할 때만)
    base_F = _concat_gate_per_sample(base_hooks)
    pert_F = _concat_gate_per_sample(pert_hooks)
    if (base_F is not None) and (pert_F is not None) and (tod_vec is not None) and (_dcor_u is not None):
        a0 = _tod_align_strength(base_F, tod_vec)
        a1 = _tod_align_strength(pert_F, tod_vec)
        if np.isfinite(a0) and np.isfinite(a1):
            out["w3_delta_tod_align"] = float(a1 - a0)

    # direct_evidence 내 TOD 관련 지표(있으면 Δ 기록)
    def _grab(d, k):
        if isinstance(d, dict) and k in d:
            try: return float(d[k])
            except Exception: return float("nan")
        return float("nan")

    for k in ("gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2"):
        b = _grab(base_direct, k)
        p = _grab(pert_direct, k)
        if np.isfinite(b) and np.isfinite(p):
            out["w3_delta_"+k] = float(p - b)

    return out
