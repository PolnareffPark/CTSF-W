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
import numpy as np
import pandas as pd

# 공통 키(모든 행 강제): experiment_type,dataset,plot_type,horizon,seed,mode,model_tag,run_tag
_W3_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag"
]

# ---------------------------
# 공통 유틸 (W1/W2와 동일 정책)
# ---------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def _append_or_update_row(csv_path: str | Path, row: Dict[str,Any], subset_keys: List[str]) -> None:
    path = _ensure_dir(csv_path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # 컬럼 합집합
        for c in new.columns:
            if c not in old.columns: old[c] = np.nan
        for c in old.columns:
            if c not in new.columns: new[c] = np.nan
        # 키 매칭 제거 후 추가
        mask = np.ones(len(old), dtype=bool)
        if len(old)>0 and all(k in old.columns for k in subset_keys):
            for k in subset_keys:
                mask &= (old[k] == row[k])
            old = pd.concat([old[~mask], new], ignore_index=True)
        else:
            old = pd.concat([old, new], ignore_index=True)
        _atomic_write_csv(old, path)
    else:
        _atomic_write_csv(new, path)

def _append_or_update_rows(csv_path: str | Path, rows: List[Dict[str,Any]], subset_keys: List[str]) -> None:
    if not rows: return
    path = _ensure_dir(csv_path)
    new = pd.DataFrame(rows)
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
        return float(np.round(np.float32(f), ndigits)) if np.isfinite(f) else np.nan
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
    ctx: Dict[str,Any],   # plot_type='forest_plot', mode ∈ {'none','tod_shift','smooth'}
    rmse_real: float,
    mae_real: float,
    detail_csv: str | Path
) -> None:
    row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "perturbation": mode_to_perturbation(ctx.get("mode","")),
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
    gate_tod_summary_csv: Optional[str | Path] = None,
    gate_dist_summary_csv: Optional[str | Path] = None,
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon","perturbation")
) -> pd.DataFrame:
    detail_csv = Path(detail_csv)
    if not detail_csv.exists():
        out = pd.DataFrame(columns=list(group_keys)+["n_pairs","delta_rmse_mean","delta_mae_mean","delta_tod_mean","delta_var_time_mean"])
        _atomic_write_csv(out, summary_csv)
        return out

    df = pd.read_csv(detail_csv)
    need = set(group_keys)|{"seed","mode","rmse_real","mae_real"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in forest detail: {miss}")

    # baseline vs perturbed pair 매칭
    base = df[df["mode"]=="none"].copy()
    per  = df[df["mode"].isin(["tod_shift","smooth"])].copy()
    if len(per)==0:
        out = pd.DataFrame(columns=list(group_keys)+["n_pairs","delta_rmse_mean","delta_mae_mean","delta_tod_mean","delta_var_time_mean"])
        _atomic_write_csv(out, summary_csv)
        return out

    # perturbation label 추가
    per["perturbation"] = per["mode"].map(mode_to_perturbation)
    on = ["experiment_type","dataset","horizon","seed"]
    base["perturbation"] = "TOD"  # dummy; merge 후 per의 label 사용
    merged = per.merge(base[on+["rmse_real","mae_real"]], on=on, suffixes=("_per","_base"))
    # Δ 계산
    merged["delta_rmse"] = (merged["rmse_real_per"] - merged["rmse_real_base"]).astype(np.float32)
    merged["delta_mae"]  = (merged["mae_real_per"]  - merged["mae_real_base"]).astype(np.float32)

    # (선택) 게이트 보조 조인
    if gate_tod_summary_csv and Path(gate_tod_summary_csv).exists():
        tdf = pd.read_csv(gate_tod_summary_csv)
        need_t = {"experiment_type","dataset","horizon","seed","mode","tod_amp_mean"}
        if need_t.issubset(set(tdf.columns)):
            t_per  = tdf[tdf["mode"].isin(["tod_shift","smooth"])][["experiment_type","dataset","horizon","seed","tod_amp_mean"]].rename(columns={"tod_amp_mean":"tod_amp_mean_per"})
            t_base = tdf[tdf["mode"]=="none"][["experiment_type","dataset","horizon","seed","tod_amp_mean"]].rename(columns={"tod_amp_mean":"tod_amp_mean_base"})
            merged = merged.merge(t_per,  on=on, how="left").merge(t_base, on=on, how="left")
            merged["delta_tod"] = (merged["tod_amp_mean_per"] - merged["tod_amp_mean_base"]).astype(np.float32)

    if gate_dist_summary_csv and Path(gate_dist_summary_csv).exists():
        vdf = pd.read_csv(gate_dist_summary_csv)
        if {"experiment_type","dataset","horizon","seed","mode","w2_gate_variability_time"}.issubset(set(vdf.columns)):
            v_per  = vdf[vdf["mode"].isin(["tod_shift","smooth"])][["experiment_type","dataset","horizon","seed","w2_gate_variability_time"]].rename(columns={"w2_gate_variability_time":"var_time_per"})
            v_base = vdf[vdf["mode"]=="none"][["experiment_type","dataset","horizon","seed","w2_gate_variability_time"]].rename(columns={"w2_gate_variability_time":"var_time_base"})
            merged = merged.merge(v_per, on=on, how="left").merge(v_base, on=on, how="left")
            merged["delta_var_time"] = (merged["var_time_per"] - merged["var_time_base"]).astype(np.float32)

    # 집계
    rows: List[Dict[str,Any]] = []
    for gvals, gdf in merged.groupby(list(group_keys), group_keys=False):
        rows.append({
            **{k:v for k,v in zip(group_keys,gvals)},
            "n_pairs": int(len(gdf)),
            "delta_rmse_mean":     _ci_mean(gdf["delta_rmse"].values.astype(np.float64)),
            "delta_mae_mean":      _ci_mean(gdf["delta_mae"].values.astype(np.float64)),
            "delta_tod_mean":      _ci_mean(gdf["delta_tod"].values.astype(np.float64)) if "delta_tod" in gdf.columns else np.nan,
            "delta_var_time_mean": _ci_mean(gdf["delta_var_time"].values.astype(np.float64)) if "delta_var_time" in gdf.columns else np.nan,
        })
    out = pd.DataFrame(rows)
    _atomic_write_csv(out, summary_csv)
    return out

# ============================================================
# 2) Perturb Delta Bar (원인 지표 Δ)
#   - A4‑TOD: Δ{gc_kernel_tod_dcor, gc_feat_tod_dcor, gc_feat_tod_r2}  (음수 기대)
#   - A4‑PEAK: Δ{cg_event_gain, |cg_bestlag|, cg_spearman_mean}       (event_gain/ spearman 음수, |lag| 양수 기대)
#   source: results/results_W3.csv (baseline/perturbed run raw)
# ============================================================

_TOD_KEYS  = ["gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2"]
_PEAK_KEYS = ["cg_event_gain","cg_spearman_mean","cg_bestlag"]  # |lag|는 집계에서 abs 처리

def rebuild_w3_perturb_delta_summary(
    results_w3_csv: str | Path,
    out_dir: str | Path,   # results/results_W3/<dataset>/perturb_delta_bar/
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon","perturbation")
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    results_w3_csv = Path(results_w3_csv)
    if not results_w3_csv.exists():
        # 빈 파일 생성
        empty = pd.DataFrame(columns=list(group_keys)+["metric","delta","seed"])
        _atomic_write_csv(empty, Path(out_dir)/"perturb_delta_bar_detail.csv")
        _atomic_write_csv(pd.DataFrame(columns=list(group_keys)+["metric","delta_mean"]), Path(out_dir)/"perturb_delta_bar_summary.csv")
        return empty, pd.DataFrame()

    df = pd.read_csv(results_w3_csv)
    need = {"experiment_type","dataset","horizon","seed","mode"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in results_W3.csv: {miss}")

    # baseline/perturbed 분리
    base = df[df["mode"]=="none"].copy()
    per  = df[df["mode"].isin(["tod_shift","smooth"])].copy()
    if len(per)==0:
        empty = pd.DataFrame(columns=list(group_keys)+["metric","delta","seed"])
        _atomic_write_csv(empty, Path(out_dir)/"perturb_delta_bar_detail.csv")
        _atomic_write_csv(pd.DataFrame(columns=list(group_keys)+["metric","delta_mean"]), Path(out_dir)/"perturb_delta_bar_summary.csv")
        return empty, pd.DataFrame()

    per["perturbation"] = per["mode"].map(mode_to_perturbation)
    on = ["experiment_type","dataset","horizon","seed"]
    merged = per.merge(base[on+_TOD_KEYS+_PEAK_KEYS], on=on, suffixes=("_per","_base"))

    # Δ 계산
    rows: List[Dict[str,Any]] = []
    for idx, r in merged.iterrows():
        ctx = {k:r[k] for k in on}
        pert = r["perturbation"]
        # A4‑TOD
        if pert=="TOD":
            for k in _TOD_KEYS:
                a = r[f"{k}_per"]; b = r[f"{k}_base"]
                delta = np.nan
                try:
                    a = float(a); b = float(b)
                    delta = a - b
                except Exception:
                    pass
                rows.append({**ctx, "perturbation": pert, "metric": k, "delta": _to_f32(delta), "seed": int(r["seed"])})
        # A4‑PEAK
        if pert=="PEAK":
            # event_gain, spearman
            for k in ["cg_event_gain","cg_spearman_mean"]:
                a = r[f"{k}_per"]; b = r[f"{k}_base"]
                delta = np.nan
                try:
                    a = float(a); b = float(b)
                    delta = a - b
                except Exception:
                    pass
                rows.append({**ctx, "perturbation": pert, "metric": k, "delta": _to_f32(delta), "seed": int(r["seed"])})
            # |bestlag|
            try:
                a = float(r["cg_bestlag_per"]); b = float(r["cg_bestlag_base"])
                delta = abs(a) - abs(b)
            except Exception:
                delta = np.nan
            rows.append({**ctx, "perturbation": pert, "metric": "abs_cg_bestlag", "delta": _to_f32(delta), "seed": int(r["seed"])})
    detail = pd.DataFrame(rows)
    _atomic_write_csv(detail, Path(out_dir)/"perturb_delta_bar_detail.csv")

    # summary
    rows2: List[Dict[str,Any]] = []
    for gvals, gdf in detail.groupby(list(group_keys)+["metric"], group_keys=False):
        rows2.append({
            **{k:v for k,v in zip(list(group_keys)+["metric"], gvals)},
            "delta_mean": _to_f32(float(np.nanmean(gdf["delta"].values.astype(np.float64))))
        })
    summ = pd.DataFrame(rows2)
    _atomic_write_csv(summ, Path(out_dir)/"perturb_delta_bar_summary.csv")
    return detail, summ
