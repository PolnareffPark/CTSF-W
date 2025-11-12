# utils/experiment_metrics/w3_metrics.py
# ============================================================
# W3 실험 특화 지표(개입 효과): ΔRMSE/ΔTOD/ΔPEAK, 효과크기(d, d_z),
# 순위 보존률(스피어만 ρ, Top-k 보존), 라그 분포 변화
# ============================================================

from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
from scipy import stats

# 안전 float 꺼내기
def _f(d: Dict, k: str, default=np.nan) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return default

def _safe_array(x) -> np.ndarray:
    try:
        a = np.asarray(x, dtype=float).ravel()
        return a[np.isfinite(a)]
    except Exception:
        return np.array([], dtype=float)

def _rmse_from(d: Dict) -> float:
    # rmse 우선, 없으면 mse_real → sqrt
    if "rmse" in d and np.isfinite(_f(d, "rmse")):
        return _f(d, "rmse")
    if "mse_real" in d and np.isfinite(_f(d, "mse_real")):
        return float(np.sqrt(_f(d, "mse_real")))
    return np.nan

def _cohens_d(base: np.ndarray, pert: np.ndarray) -> float:
    # 독립표본형 d (fallback): pooled SD
    bx = _safe_array(base); px = _safe_array(pert)
    if bx.size < 2 or px.size < 2: return np.nan
    m1, m2 = float(bx.mean()), float(px.mean())
    s1, s2 = float(bx.std(ddof=1)), float(px.std(ddof=1))
    if s1 == 0 and s2 == 0: return 0.0
    sp = np.sqrt(((bx.size-1)*s1*s1 + (px.size-1)*s2*s2) / max(1, (bx.size+px.size-2)))
    if not np.isfinite(sp) or sp == 0: return np.nan
    return (m2 - m1) / sp

def _cohens_dz(base: np.ndarray, pert: np.ndarray) -> Tuple[float, float, float]:
    # 대응표본형 d_z: diff = pert-base, d_z = mean(diff)/std(diff)
    bx = _safe_array(base); px = _safe_array(pert)
    L = min(bx.size, px.size)
    if L < 3: return (np.nan, np.nan, np.nan)
    diff = px[:L] - bx[:L]
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1)) if L > 1 else np.nan
    if not np.isfinite(sd) or sd == 0:
        return (np.nan, np.nan, np.nan)
    dz = md / sd
    # 95% CI (정규 근사)
    se = sd / np.sqrt(L)
    lo, hi = md - 1.96*se, md + 1.96*se
    # d_z의 CI는 보고 편의를 위해 mean diff의 CI를 d_z와 함께 보조 출력
    return (dz, lo, hi)

def _spearman_preserve(base_err: np.ndarray, pert_err: np.ndarray, topk: float=0.1) -> Tuple[float, float]:
    b = _safe_array(base_err); p = _safe_array(pert_err)
    L = min(b.size, p.size)
    if L < 5: return (np.nan, np.nan)
    b = b[:L]; p = p[:L]
    rho = stats.spearmanr(b, p, nan_policy="omit").correlation
    # Top-k 보존률: baseline 상위 k% 인덱스가 perturb에서도 상위 k%로 남는 비율
    k = max(1, int(np.ceil(L * topk)))
    top_b = np.argsort(-b)[:k]
    top_p = np.argsort(-p)[:k]
    inter = len(set(top_b.tolist()) & set(top_p.tolist()))
    preserve = inter / float(k)
    return (float(rho), float(preserve))

def compute_w3_metrics(
    *,
    baseline_metrics: Dict,
    perturb_metrics: Dict,
    win_errors_base: Optional[np.ndarray] = None,
    win_errors_pert: Optional[np.ndarray] = None,
    dz_ci: bool = True
) -> Dict:
    """
    개입 효과(perturb - base)를 계산해 반환.
    반환 키(대표):
      - w3_delta_rmse, w3_delta_rmse_pct, w3_delta_mae
      - w3_delta_gc_kernel_tod_dcor, w3_delta_gc_feat_tod_dcor, w3_delta_gc_feat_tod_r2   (A4‑TOD)
      - w3_delta_cg_event_gain, w3_delta_cg_bestlag_abs, w3_delta_cg_spearman_mean       (A4‑PEAK)
      - w3_effect_d, w3_effect_dz, w3_dz_lo, w3_dz_hi
      - w3_rank_spearman, w3_rank_topk_preserve
      - w3_hypothesis_tod_support, w3_hypothesis_peak_support (0/1)
    """
    out: Dict[str, float] = {}

    # --- 기본 오차 변화 ---
    rmse_b = _rmse_from(baseline_metrics)
    rmse_p = _rmse_from(perturb_metrics)
    mae_b  = _f(baseline_metrics, "mae")
    mae_p  = _f(perturb_metrics, "mae")

    out["w3_delta_rmse"]      = rmse_p - rmse_b if np.isfinite(rmse_p) and np.isfinite(rmse_b) else np.nan
    out["w3_delta_rmse_pct"]  = 100.0 * out["w3_delta_rmse"] / rmse_b if np.isfinite(rmse_b) and rmse_b > 0 else np.nan
    out["w3_delta_mae"]       = mae_p - mae_b if np.isfinite(mae_p) and np.isfinite(mae_b) else np.nan

    # --- TOD 민감도 변화(A4‑TOD) / PEAK 연관(A4‑PEAK) ---
    # TOD(문맥→필터) 계열: gc_* (커널/피처 vs TOD)
    for k in ["gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2"]:
        out[f"w3_delta_{k}"] = _f(perturb_metrics, k) - _f(baseline_metrics, k)

    # PEAK(모양→업데이트) 계열: cg_* (Conv→GRU 이벤트/순위/라그)
    out["w3_delta_cg_event_gain"]     = _f(perturb_metrics, "cg_event_gain")   - _f(baseline_metrics, "cg_event_gain")
    out["w3_delta_cg_spearman_mean"]  = _f(perturb_metrics, "cg_spearman_mean")- _f(baseline_metrics, "cg_spearman_mean")

    bl_b = abs(_f(baseline_metrics, "cg_bestlag"))
    bl_p = abs(_f(perturb_metrics, "cg_bestlag"))
    out["w3_delta_cg_bestlag_abs"]    = bl_p - bl_b

    # --- 랭크 보존/효과 크기 ---
    # window‑wise error가 주어지면 pairwise d_z 및 Spearman 보존률 계산
    base_w = _safe_array(win_errors_base)
    pert_w = _safe_array(win_errors_pert)

    # 독립표본 d (폴백): 윈도우 벡터가 존재할 때만
    out["w3_effect_d"] = _cohens_d(base_w, pert_w) if base_w.size >= 2 and pert_w.size >= 2 else np.nan

    if base_w.size >= 3 and pert_w.size >= 3:
        dz, lo, hi = _cohens_dz(base_w, pert_w) if dz_ci else (np.nan, np.nan, np.nan)
        out["w3_effect_dz"] = dz
        out["w3_dz_lo"]     = lo
        out["w3_dz_hi"]     = hi
        rho, keep = _spearman_preserve(base_w, pert_w, topk=0.10)
        out["w3_rank_spearman"]      = rho
        out["w3_rank_topk_preserve"] = keep
    else:
        out["w3_effect_dz"] = np.nan
        out["w3_dz_lo"] = out["w3_dz_hi"] = np.nan
        out["w3_rank_spearman"] = out["w3_rank_topk_preserve"] = np.nan

    # --- 가설 지표(이분법; 방향성 확인) ---
    # A4‑TOD: ΔRMSE% > 0 & ΔTOD 민감도 지표(거리상관 등) 감소
    tod_terms = [out.get("w3_delta_gc_kernel_tod_dcor"), out.get("w3_delta_gc_feat_tod_dcor"), out.get("w3_delta_gc_feat_tod_r2")]
    tod_drop  = [t for t in tod_terms if np.isfinite(t) and t < 0]
    out["w3_hypothesis_tod_support"]  = 1.0 if (np.isfinite(out["w3_delta_rmse_pct"]) and out["w3_delta_rmse_pct"] > 0 and len(tod_drop) >= 1) else 0.0

    # A4‑PEAK: ΔRMSE% > 0 & (Δevent_gain < 0) & (Δ|bestlag| > 0) & (Δspearman < 0)
    peak_ok = []
    if np.isfinite(out["w3_delta_cg_event_gain"]):    peak_ok.append(out["w3_delta_cg_event_gain"] < 0)
    if np.isfinite(out["w3_delta_cg_bestlag_abs"]):   peak_ok.append(out["w3_delta_cg_bestlag_abs"] > 0)
    if np.isfinite(out["w3_delta_cg_spearman_mean"]): peak_ok.append(out["w3_delta_cg_spearman_mean"] < 0)
    out["w3_hypothesis_peak_support"] = 1.0 if (np.isfinite(out["w3_delta_rmse_pct"]) and out["w3_delta_rmse_pct"] > 0 and all(peak_ok) and len(peak_ok)>=2) else 0.0

    return out
