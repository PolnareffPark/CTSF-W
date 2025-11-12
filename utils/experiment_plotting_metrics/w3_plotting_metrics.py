# ============================================================
# utils/experiment_plotting_metrics/w3_plotting_metrics.py
#   - W3 전용: Forest(교란 효과) / Perturb Delta Bar / Gate-TOD / Gate Distribution
#   - summary + detail 분리 저장 (append + dedupe, atomic write)
#   - W1/W2 스키마/키/경로 규칙 준수
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math
import time
import numpy as np
import pandas as pd

_W3_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed",
    "mode","perturbation","model_tag","run_tag"
]

# ---------------------------
# 공통 유틸 (W1/W2와 동일 정책)
# ---------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_dir(path); tmp = path.with_suffix(path.suffix + f".tmp{int(time.time()*1000)}")
    df.to_csv(tmp, index=False); tmp.replace(path)

def _append_or_update_row(csv_path: str | Path, row: Dict[str,Any], subset_keys: List[str]) -> None:
    path = _ensure_dir(csv_path); new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        if len(old)>0 and all(k in old.columns for k in subset_keys):
            mask = np.ones(len(old), dtype=bool)
            for k in subset_keys: mask &= (old[k] == row[k])
            old = pd.concat([old[~mask], new], ignore_index=True)
        else:
            old = pd.concat([old, new], ignore_index=True)
        _atomic_write_csv(old, path)
    else:
        _atomic_write_csv(new, path)

def _append_or_update_rows(csv_path: str | Path, rows: List[Dict[str,Any]], subset_keys: List[str]) -> None:
    if not rows: return
    path = _ensure_dir(csv_path); new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        for c in new.columns:
            if c not in old.columns: old[c] = np.nan
        for c in old.columns:
            if c not in new.columns: new[c] = np.nan
        df = pd.concat([old, new], ignore_index=True)
        df.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(df, path)
    else:
        new.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(new, path)

def _to_f32(v: Any, ndigits: int = 6) -> float:
    try:
        f = float(v); 
        if not np.isfinite(f): return np.nan
        return float(np.round(np.float32(f), ndigits))
    except Exception:
        return np.nan

def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    m = s.mean(); sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0: 
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd

# ------------------------------------------------------------
# Gate 유틸(W3에서 독립 제공: TOD bin 자동, 게이트 벡터 통계)
# ------------------------------------------------------------
def infer_tod_bins_by_dataset(dataset: str) -> int:
    ds = (dataset or "").lower()
    if "ettm" in ds:    return 96   # 15min
    if "etth" in ds:    return 24   # 1h
    if "weather" in ds: return 144  # 10min
    return 24  # 보수적 기본

def _reduce_time_axis_auto(arr: np.ndarray, dataset: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if a.ndim == 1:
        return a
    T_candidates = {infer_tod_bins_by_dataset(dataset)}
    cand = [ax for ax, sz in enumerate(a.shape) if sz in T_candidates]
    t_axis = cand[0] if cand else int(np.argmax(a.shape))
    axes = tuple(i for i in range(a.ndim) if i != t_axis)
    return np.nanmean(a, axis=axes)

def _gate_entropy_from_flat(flat: np.ndarray) -> float:
    v = np.asarray(flat, dtype=float).ravel()
    v = v[np.isfinite(v) & (v > 1e-12)]
    if v.size == 0: return np.nan
    p = v / (v.sum() + 1e-12)
    return float(-np.sum(p * np.log(p + 1e-12)))

# ============================================================
# 0) Gate‑TOD Heatmap / Gate Distribution (W3 컨텍스트)
# ============================================================
def save_w3_gate_tod_heatmap(
    ctx: Dict[str,Any],                    # plot_type='gate_tod_heatmap'
    gates_by_stage: Dict[str, np.ndarray], # {"S":..., "M":..., "D":...} any shape
    out_dir: str | Path,
    amp_method: str = "range",
) -> None:
    assert ctx.get("plot_type") == "gate_tod_heatmap", "plot_type must be 'gate_tod_heatmap'"
    s = gates_by_stage.get("S"); m = gates_by_stage.get("M"); d = gates_by_stage.get("D")
    if s is None and m is None and d is None:
        return
    ds = str(ctx.get("dataset"))
    def _as_1d(x):
        if x is None: return np.array([], dtype=float)
        return _reduce_time_axis_auto(x, ds)
    s1 = _as_1d(s); m1 = _as_1d(m); d1 = _as_1d(d)
    T = int(max(len(s1), len(m1), len(d1)))
    tod_labels = list(range(T))

    def _amp(x):
        if x.size == 0 or not np.isfinite(x).any(): return np.nan
        return float(np.nanstd(x)) if amp_method=="std" else float(np.nanmax(x) - np.nanmin(x))

    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "tod_bins": T,
        "tod_amp_s": _to_f32(_amp(s1)), "tod_amp_m": _to_f32(_amp(m1)), "tod_amp_d": _to_f32(_amp(d1)),
    }
    summary_row["tod_amp_mean"] = _to_f32(np.nanmean([summary_row["tod_amp_s"], summary_row["tod_amp_m"], summary_row["tod_amp_d"]]))
    # 품질
    all_vals = np.concatenate([v for v in [s1,m1,d1] if v.size>0], axis=0) if T>0 else np.array([],float)
    total = all_vals.size; valid = int(np.isfinite(all_vals).sum())
    summary_row["valid_ratio"] = _to_f32(valid/total if total>0 else np.nan)
    summary_row["nan_count"]   = int(total - valid)
    summary_row["amp_method"]  = amp_method

    out_dir = Path(out_dir)
    _append_or_update_row(out_dir/"gate_tod_heatmap_summary.csv", summary_row, subset_keys=_W3_KEYS)

    # detail
    rows: List[Dict[str,Any]] = []
    def _emit(stage, arr):
        for t,v in enumerate(arr):
            if np.isfinite(v):
                rows.append({
                    **{k: ctx[k] for k in _W3_KEYS if k in ctx},
                    "tod": int(tod_labels[t]),
                    "stage": stage,
                    "gate_mean": _to_f32(v)
                })
    if s1.size: _emit("S", s1)
    if m1.size: _emit("M", m1)
    if d1.size: _emit("D", d1)
    if rows:
        _append_or_update_rows(out_dir/"gate_tod_heatmap_detail.csv", rows, subset_keys=_W3_KEYS+["tod","stage"])


def save_w3_gate_distribution(
    ctx: Dict[str,Any],                    # plot_type='gate_distribution'
    gates_all: np.ndarray,                 # 1D or any shape → flatten
    out_dir: str | Path,
) -> None:
    assert ctx.get("plot_type") == "gate_distribution", "plot_type must be 'gate_distribution'"
    v = np.asarray(gates_all, dtype=float).ravel()
    v = v[np.isfinite(v)]
    g_mean = _to_f32(np.nanmean(v)) if v.size else np.nan
    g_std  = _to_f32(np.nanstd(v))  if v.size else np.nan
    g_ent  = _to_f32(_gate_entropy_from_flat(v))
    # Hoyer sparsity (0~1)
    if v.size:
        l1 = float(np.sum(np.abs(v))); l2 = float(np.linalg.norm(v))
        g_spa = float((np.sqrt(v.size) - (l1/(l2+1e-12))) / (np.sqrt(v.size)-1+1e-12)) if l2>0 else 0.0
    else:
        g_spa = np.nan
    # 첨도(정의역 내 안정화)
    g_kur = _to_f32(pd.Series(v).kurtosis(fisher=True)) if v.size>=4 else np.nan

    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "gate_mean": g_mean, "gate_std": g_std,
        "gate_entropy": g_ent, "gate_kurtosis": g_kur, "gate_sparsity": _to_f32(g_spa),
        "valid_ratio": _to_f32(1.0 if v.size>0 else np.nan),
        "nan_count": 0 if v.size>0 else np.nan
    }
    out_dir = Path(out_dir)
    _append_or_update_row(out_dir/"gate_distribution_summary.csv", summary_row, subset_keys=_W3_KEYS)

    # quantiles detail
    if v.size>=5:
        q = np.quantile(v, [0.05,0.10,0.25,0.50,0.75,0.90,0.95]).tolist()
        keys = ["gate_q05","gate_q10","gate_q25","gate_q50","gate_q75","gate_q90","gate_q95"]
        rows = [{
            **{k: ctx[k] for k in _W3_KEYS if k in ctx},
            "stat": kq, "value": _to_f32(val)
        } for kq, val in zip(keys, q)]
        _append_or_update_rows(out_dir/"gate_distribution_quantiles_detail.csv", rows, subset_keys=_W3_KEYS+["stat"])


# ============================================================
# 1) Forest Plot (baseline vs perturbed, Δ/CI/Top-pick)
#     group_keys = (experiment_type, dataset, horizon, perturbation)
#     pairing: mode 'none' ↔ mode in {'tod_shift','smooth'}
# ============================================================
def mode_to_perturbation(mode: str) -> str:
    m = (mode or "").lower()
    if m == "tod_shift": return "TOD"
    if m == "smooth":    return "PEAK"
    return "NONE"

def save_w3_forest_detail_row(
    ctx: Dict[str,Any], rmse_real: float, mae_real: float,
    detail_csv: str | Path
) -> None:
    row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "rmse_real": _to_f32(rmse_real),
        "mae_real":  _to_f32(mae_real),
    }
    _append_or_update_row(detail_csv, row, subset_keys=_W3_KEYS)

def _ci_mean(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    if a.size==0: return np.nan
    return _to_f32(float(np.mean(a)))

def rebuild_w3_forest_summary(
    detail_csv: str | Path,
    summary_csv: str | Path,
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon","perturbation"),
    baseline_mode: str = "none"
) -> pd.DataFrame:
    detail_csv = Path(detail_csv)
    if not detail_csv.exists():
        out = pd.DataFrame(columns=list(group_keys)+["n_pairs","delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean"])
        _atomic_write_csv(out, summary_csv); return out
    df = pd.read_csv(detail_csv)
    need = set(group_keys) | {"seed","mode","rmse_real","mae_real"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in W3 forest detail: {miss}")

    # baseline=none, perturbed={tod_shift|smooth}
    base = df[df["mode"]==baseline_mode].copy()
    pert = df[df["mode"]!=baseline_mode].copy()

    on = list(group_keys) + ["seed"]
    merged = pert.merge(base[on + ["rmse_real","mae_real"]], on=on, suffixes=("", "_base"))

    if len(merged)==0:
        out = pd.DataFrame(columns=list(group_keys)+["n_pairs","delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean"])
        _atomic_write_csv(out, summary_csv); return out

    merged["delta_rmse"]     = (merged["rmse_real"] - merged["rmse_real_base"]).astype(np.float32)
    merged["delta_rmse_pct"] = (merged["delta_rmse"] / merged["rmse_real_base"]).astype(np.float32) * 100.0
    merged["delta_mae"]      = (merged["mae_real"]  - merged["mae_real_base"]).astype(np.float32)

    rows = []
    for gvals, gdf in merged.groupby(list(group_keys)):
        row = {
            **{k:v for k,v in zip(group_keys,gvals)},
            "n_pairs": int(len(gdf)),
            "delta_rmse_mean":     _to_f32(np.mean(gdf["delta_rmse"].values)),
            "delta_rmse_pct_mean": _to_f32(np.mean(gdf["delta_rmse_pct"].values)),
            "delta_mae_mean":      _to_f32(np.mean(gdf["delta_mae"].values)),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    _atomic_write_csv(out, summary_csv)
    return out

# ============================================================
# 2) Perturb Delta Bar (원인 지표 Δ)
#   - A4‑TOD: Δ{gc_kernel_tod_dcor, gc_feat_tod_dcor, gc_feat_tod_r2}  (음수 기대)
#   - A4‑PEAK: Δ{cg_event_gain, |cg_bestlag|, cg_spearman_mean}       (event_gain/ spearman 음수, |lag| 양수 기대)
#   source: results/results_W3.csv (baseline/perturbed run raw)
# ============================================================

_TOD_METRICS = ["gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2"]
_PEAK_METRICS = ["cg_event_gain","abs_cg_bestlag","cg_spearman_mean"]

def save_w3_perturb_delta_detail(
    results_csv: str | Path,
    out_detail_csv: str | Path,
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon","perturbation"),
    baseline_mode: str = "none"
) -> pd.DataFrame:
    results_csv = Path(results_csv)
    if not results_csv.exists():
        _atomic_write_csv(pd.DataFrame(columns=list(group_keys)+["metric","seed","delta"]), out_detail_csv)
        return pd.DataFrame()

    df = pd.read_csv(results_csv)
    need = set(group_keys) | {"mode","seed"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in results_W3.csv: {miss}")

    # abs(cg_bestlag) 파생
    if "cg_bestlag" in df.columns and "abs_cg_bestlag" not in df.columns:
        try:
            df["abs_cg_bestlag"] = df["cg_bestlag"].abs().astype(float)
        except Exception:
            df["abs_cg_bestlag"] = np.nan

    base = df[df["mode"]==baseline_mode].copy()
    pert = df[df["mode"]!=baseline_mode].copy()
    on = list(group_keys) + ["seed"]
    merged = pert.merge(base[on + _TOD_METRICS + _PEAK_METRICS], on=on, suffixes=("", "_base"))

    rows = []
    for gvals, gdf in merged.groupby(list(group_keys)):
        pert_tag = gvals[-1]
        use_metrics = _TOD_METRICS if str(pert_tag)=="TOD" else _PEAK_METRICS
        for _, r in gdf.iterrows():
            for m in use_metrics:
                if m not in gdf.columns or f"{m}_base" not in gdf.columns: 
                    continue
                dv = _to_f32(r[m]) - _to_f32(r[f"{m}_base"])
                rows.append({
                    **{k:v for k,v in zip(group_keys,gvals)},
                    "metric": m, "seed": int(r["seed"]),
                    "delta": _to_f32(dv)
                })
    out = pd.DataFrame(rows)
    _atomic_write_csv(out, out_detail_csv)
    return out

def rebuild_w3_perturb_delta_summary(
    detail_csv: str | Path,
    summary_csv: str | Path,
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon","perturbation")
) -> pd.DataFrame:
    detail_csv = Path(detail_csv)
    if not detail_csv.exists():
        _atomic_write_csv(pd.DataFrame(columns=list(group_keys)+["metric","delta_mean"]), summary_csv)
        return pd.DataFrame()
    df = pd.read_csv(detail_csv)
    need = set(group_keys) | {"metric","delta"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in perturb_delta_bar_detail.csv: {miss}")

    rows = []
    for gvals, gdf in df.groupby(list(group_keys)+["metric"], group_keys=False):
        # 평균치(필요시 CI 추가 가능)
        delta_mean = _to_f32(np.mean(gdf["delta"].values.astype(np.float64)))
        row = {**{k:v for k,v in zip(list(group_keys)+["metric"], gvals)}, "delta_mean": delta_mean}
        rows.append(row)
    out = pd.DataFrame(rows)
    _atomic_write_csv(out, summary_csv)
    return out