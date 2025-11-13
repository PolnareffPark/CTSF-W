# ============================================================
# utils/experiment_plotting_metrics/w2_plotting_metrics.py
#   - W2 전용: Gate-TOD Heatmap / Gate Distribution / Forest Plot
#   - summary + detail 분리 저장 (append + dedupe, atomic write)
#   - W1 스키마/경로/키 규칙 100% 준수
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time, math
import numpy as np
import pandas as pd

# 모든 행에 강제 포함되는 표준 키 (W1과 동일 + run_tag/ model_tag)
_W2_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed","mode",
    "model_tag","run_tag"
]

# ---------------------------
# 공통 유틸 (W1과 동일 정책)
# ---------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time()*1000)}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def _append_or_update_row(csv_path: str | Path, row: Dict[str,Any], subset_keys: List[str]) -> None:
    path = _ensure_dir(csv_path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        if len(old) > 0 and all(k in old.columns for k in subset_keys):
            mask = np.ones(len(old), dtype=bool)
            for k in subset_keys:
                mask &= (old[k] == row[k])
            old = pd.concat([old[~mask], new], ignore_index=True)
        else:
            old = pd.concat([old, new], ignore_index=True)
        _atomic_write_csv(old, path)
    else:
        _atomic_write_csv(new, path)

def _append_or_update_rows(csv_path: str | Path, rows: List[Dict[str,Any]], subset_keys: List[str]) -> None:
    if not rows:
        return
    path = _ensure_dir(csv_path)
    new = pd.DataFrame(rows)
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
        df.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(df, path)
    else:
        new.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(new, path)

def _to_f32(v: Any, ndigits: int = 6) -> float:
    try:
        f = float(v)
        if not np.isfinite(f): return np.nan
        return float(np.round(np.float32(f), ndigits))
    except Exception:
        return np.nan

def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    m = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd


# ============================================================
# 1) Gate‑TOD Heatmap (summary + detail)
# ============================================================
def save_w2_gate_tod_heatmap(
    ctx: Dict[str,Any],                    # 키 6종 + model_tag + run_tag 포함, plot_type='gate_tod_heatmap'
    metrics: Dict[str,Any],                # 기대: gate_tod_mean_s/m/d(list), 선택적으로 tod_labels(list)
    out_dir: str | Path,                   # 예: results/results_W2/<dataset>/gate_tod_heatmap
    amp_method: str = "range",             # 'range'=(max-min), 'std' = 표준편차
) -> None:
    assert ctx.get("plot_type") == "gate_tod_heatmap", "plot_type must be 'gate_tod_heatmap'"

    def _get_list_any(keys: List[str]) -> Optional[List[float]]:
        for k in keys:
            if k in metrics and metrics[k] is not None:
                return list(metrics[k])
        return None

    s_list = _get_list_any(["gate_tod_mean_s","gate_tod_mean_S"])
    m_list = _get_list_any(["gate_tod_mean_m","gate_tod_mean_M"])
    d_list = _get_list_any(["gate_tod_mean_d","gate_tod_mean_D"])
    tod_labels = _get_list_any(["tod_labels","tod_bins"])
    if s_list is None and m_list is None and d_list is None:
        return

    s = np.array(s_list or [], dtype=float)
    m = np.array(m_list or [], dtype=float)
    d = np.array(d_list or [], dtype=float)

    T = max(len(s), len(m), len(d))
    if tod_labels is None or len(tod_labels) != T:
        tod_labels = list(range(T))

    def _amp(x: np.ndarray) -> float:
        if x.size == 0 or not np.isfinite(x).any(): return np.nan
        if amp_method == "std":
            return _to_f32(np.nanstd(x))
        return _to_f32(np.nanmax(x) - np.nanmin(x))

    amp_s = _amp(s); amp_m = _amp(m); amp_d = _amp(d)
    amp_mean = _to_f32(np.nanmean([v for v in [amp_s,amp_m,amp_d] if np.isfinite(v)]))

    all_vals = np.concatenate([v for v in [s,m,d] if v.size > 0], axis=0) if T > 0 else np.array([],dtype=float)
    valid = np.isfinite(all_vals).sum()
    total = all_vals.size
    valid_ratio = float(valid)/float(total) if total>0 else np.nan
    nan_count = int(total - valid)

    summary_row = {
        **{k: ctx[k] for k in _W2_KEYS if k in ctx},
        "tod_bins": int(T),
        "tod_amp_s": amp_s, "tod_amp_m": amp_m, "tod_amp_d": amp_d,
        "tod_amp_mean": amp_mean,
        "valid_ratio": _to_f32(valid_ratio), "nan_count": nan_count,
        "amp_method": amp_method
    }
    summary_csv = Path(out_dir) / "gate_tod_heatmap_summary.csv"
    _append_or_update_row(summary_csv, summary_row, subset_keys=_W2_KEYS)

    rows: List[Dict[str,Any]] = []
    def _emit(stage: str, arr: np.ndarray):
        for t in range(len(arr)):
            v = arr[t]
            if not np.isfinite(v):
                continue
            rows.append({
                **{k: ctx[k] for k in _W2_KEYS if k in ctx},
                "tod": int(tod_labels[t]),
                "stage": stage,
                "gate_mean": _to_f32(v)
            })
    if s.size: _emit("S", s)
    if m.size: _emit("M", m)
    if d.size: _emit("D", d)

    detail_csv = Path(out_dir) / "gate_tod_heatmap_detail.csv"
    _append_or_update_rows(detail_csv, rows, subset_keys=_W2_KEYS + ["tod","stage"])


# ============================================================
# 2) Gate Distribution (summary + detail)
# ============================================================
def save_w2_gate_distribution(
    ctx: Dict[str,Any],                    # plot_type='gate_distribution'
    metrics: Dict[str,Any],                # 기대: gate_mean/std/entropy/kurtosis/sparsity, gate_qXX..., (선택) w2_gate_variability_time
    out_dir: str | Path,                   # 예: results/results_W2/<dataset>/gate_distribution
    quantile_keys: Optional[List[str]] = None
) -> None:
    assert ctx.get("plot_type") == "gate_distribution", "plot_type must be 'gate_distribution'"

    g_mean = _to_f32(metrics.get("gate_mean"))
    g_std  = _to_f32(metrics.get("gate_std"))
    g_ent  = _to_f32(metrics.get("gate_entropy"))
    g_kur  = _to_f32(metrics.get("gate_kurtosis"))
    g_spa  = _to_f32(metrics.get("gate_sparsity"))
    g_varT = _to_f32(metrics.get("w2_gate_variability_time") or metrics.get("gate_var_time"))
    valid_ratio = _to_f32(metrics.get("gate_valid_ratio"))
    nan_count   = int(metrics.get("gate_nan_count", 0))

    summary_row = {
        **{k: ctx[k] for k in _W2_KEYS if k in ctx},
        "gate_mean": g_mean, "gate_std": g_std,
        "gate_entropy": g_ent, "gate_kurtosis": g_kur, "gate_sparsity": g_spa,
        "w2_gate_variability_time": g_varT,
        "valid_ratio": valid_ratio, "nan_count": nan_count
    }
    summary_csv = Path(out_dir) / "gate_distribution_summary.csv"
    _append_or_update_row(summary_csv, summary_row, subset_keys=_W2_KEYS)

    if quantile_keys is None:
        quantile_keys = ["gate_q05","gate_q10","gate_q25","gate_q50","gate_q75","gate_q90","gate_q95"]
    q_rows: List[Dict[str,Any]] = []
    for qk in quantile_keys:
        if qk in metrics:
            v = _to_f32(metrics[qk])
            if np.isfinite(v):
                q_rows.append({
                    **{k: ctx[k] for k in _W2_KEYS if k in ctx},
                    "stat": qk,
                    "value": v
                })
    if q_rows:
        q_csv = Path(out_dir) / "gate_distribution_quantiles_detail.csv"
        _append_or_update_rows(q_csv, q_rows, subset_keys=_W2_KEYS + ["stat"])

    ch_rows: List[Dict[str,Any]] = []
    ch_stats = metrics.get("gate_channel_stats")
    if isinstance(ch_stats, list) and len(ch_stats) > 0:
        for item in ch_stats:
            try:
                ch = int(item.get("ch"))
            except Exception:
                continue
            ch_rows.append({
                **{k: ctx[k] for k in _W2_KEYS if k in ctx},
                "channel": ch,
                "mean": _to_f32(item.get("mean")), "std":  _to_f32(item.get("std")),
                "entropy": _to_f32(item.get("entropy")), "kurtosis": _to_f32(item.get("kurtosis")),
                "sparsity": _to_f32(item.get("sparsity")),
            })
    if ch_rows:
        ch_csv = Path(out_dir) / "gate_distribution_per_channel_detail.csv"
        _append_or_update_rows(ch_csv, ch_rows, subset_keys=_W2_KEYS + ["channel"])


# ============================================================
# 3) Forest Plot (동적 vs 정적 비교, Δ/CI/Top-pick)
# ============================================================
def save_w2_forest_detail_row(
    ctx: Dict[str,Any],             # plot_type='forest_plot', mode ∈ {'dynamic','static'}
    rmse_real: float,
    mae_real: float,
    detail_csv: str | Path          # 예: results/results_W2/<dataset>/forest_plot/forest_plot_detail.csv
) -> None:
    row = {
        **{k: ctx[k] for k in _W2_KEYS if k in ctx},
        "rmse_real": _to_f32(rmse_real),
        "mae_real":  _to_f32(mae_real),
    }
    _append_or_update_row(detail_csv, row, subset_keys=_W2_KEYS)

# (하위호환) 과거 unit/agg 명칭을 호출하던 코드가 있으면 새 스키마로 우회
def save_w2_forest_unit_row(ctx: Dict[str,Any], rmse_real: float, mae_real: float, unit_csv: str | Path) -> None:
    if str(unit_csv).endswith("W2_forest_unit.csv"):
        unit_csv = Path(unit_csv).with_name("forest_plot_detail.csv")
    save_w2_forest_detail_row(ctx, rmse_real, mae_real, detail_csv=unit_csv)

def _ci_95(a: np.ndarray) -> Tuple[float,float,float]:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan, np.nan)
    m = float(np.mean(a))
    s = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
    se = s / math.sqrt(max(1, a.size))
    return (_to_f32(m), _to_f32(m-1.96*se), _to_f32(m+1.96*se))

def rebuild_w2_forest_summary(
    detail_csv: str | Path,          # 입력 raw(detail)
    summary_csv: str | Path,         # 출력 summary
    gate_tod_summary_csv: Optional[str | Path] = None,   # 있으면 TOD 민감도 조인
    gate_dist_summary_csv: Optional[str | Path] = None,  # 있으면 시간변동성(varT) 조인
    per_mode: str = "dynamic",
    last_mode: str = "static",
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon")
) -> pd.DataFrame:
    detail_csv = Path(detail_csv)
    if not detail_csv.exists():
        out = pd.DataFrame(columns=list(group_keys)+[
            "n_pairs","delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean",
            "delta_tod_mean","delta_var_time_mean"
        ])
        _atomic_write_csv(out, summary_csv)
        return out

    df = pd.read_csv(detail_csv)
    need = set(group_keys) | {"seed","mode","rmse_real","mae_real"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in forest detail csv: {miss}")

    per = df[df["mode"]==per_mode].copy()
    lst = df[df["mode"]==last_mode].copy()
    on = list(group_keys) + ["seed"]
    base = per.merge(lst, on=on, suffixes=("_per","_last"))

    # (선택) TOD/VarT 보조 지표 조인
    if gate_tod_summary_csv and Path(gate_tod_summary_csv).exists():
        tdf = pd.read_csv(gate_tod_summary_csv)
        need_t = set(group_keys) | {"seed","mode","tod_amp_mean"}
        if need_t.issubset(set(tdf.columns)):
            t_per  = tdf[tdf["mode"]==per_mode][list(group_keys)+["seed","tod_amp_mean"]].rename(columns={"tod_amp_mean":"tod_amp_mean_per"})
            t_last = tdf[tdf["mode"]==last_mode][list(group_keys)+["seed","tod_amp_mean"]].rename(columns={"tod_amp_mean":"tod_amp_mean_last"})
            base = base.merge(t_per,  on=on, how="left").merge(t_last, on=on, how="left")

    if gate_dist_summary_csv and Path(gate_dist_summary_csv).exists():
        vdf = pd.read_csv(gate_dist_summary_csv)
        var_col = "w2_gate_variability_time" if "w2_gate_variability_time" in vdf.columns else ("gate_var_time" if "gate_var_time" in vdf.columns else None)
        if var_col:
            v_per  = vdf[vdf["mode"]==per_mode][list(group_keys)+["seed",var_col]].rename(columns={var_col:"var_time_per"})
            v_last = vdf[vdf["mode"]==last_mode][list(group_keys)+["seed",var_col]].rename(columns={var_col:"var_time_last"})
            base = base.merge(v_per,  on=on, how="left").merge(v_last, on=on, how="left")

    if len(base)==0:
        out = pd.DataFrame(columns=list(group_keys)+[
            "n_pairs","delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean",
            "delta_tod_mean","delta_var_time_mean"
        ])
        _atomic_write_csv(out, summary_csv)
        return out

    base["delta_rmse"] = (base["rmse_real_last"] - base["rmse_real_per"]).astype(np.float32)
    base["delta_rmse_pct"] = (base["delta_rmse"] / base["rmse_real_last"]).astype(np.float32) * 100.0
    base["delta_mae"]  = (base["mae_real_last"]  - base["mae_real_per"]).astype(np.float32)

    if "tod_amp_mean_per" in base.columns and "tod_amp_mean_last" in base.columns:
        base["delta_tod"] = (base["tod_amp_mean_per"] - base["tod_amp_mean_last"]).astype(np.float32)
    else:
        base["delta_tod"] = np.nan

    if "var_time_per" in base.columns and "var_time_last" in base.columns:
        base["delta_var_time"] = (base["var_time_per"] - base["var_time_last"]).astype(np.float32)
    else:
        base["delta_var_time"] = np.nan

# --- Top-pick 점수 (무가중치 z 합) : deprecation-safe (transform 기반) ---
    def _safe_z(s: pd.Series) -> pd.Series:
        s = s.astype(float)
        m = s.mean()
        sd = s.std(ddof=0)
        if not np.isfinite(sd) or sd == 0:
            return pd.Series(np.zeros(len(s)), index=s.index)
        return (s - m) / sd

    gcols = list(group_keys)
    gp = base.groupby(gcols, group_keys=False)

    z_rmse = gp["delta_rmse_pct"].transform(_safe_z).fillna(0.0)
    z_tod  = gp["delta_tod"].transform(_safe_z).fillna(0.0)
    z_var  = gp["delta_var_time"].transform(_safe_z).fillna(0.0)

    base["top_pick_score_w2"] = z_rmse + z_tod + z_var
# -------------------------------------------------------------------------

    def _ci_mean(a: np.ndarray) -> float:
        a = a[np.isfinite(a)]
        if a.size == 0: return np.nan
        return _to_f32(float(np.mean(a)))

    rows: List[Dict[str,Any]] = []
    for gvals, gdf in base.groupby(list(group_keys)):
        row = {
            **{k:v for k,v in zip(group_keys,gvals)},
            "n_pairs": int(len(gdf)),
            "delta_rmse_mean":      _ci_mean(gdf["delta_rmse"].values.astype(np.float64)),
            "delta_rmse_pct_mean":  _ci_mean(gdf["delta_rmse_pct"].values.astype(np.float64)),
            "delta_mae_mean":       _ci_mean(gdf["delta_mae"].values.astype(np.float64)),
            "delta_tod_mean":       _ci_mean(gdf["delta_tod"].values.astype(np.float64)),
            "delta_var_time_mean":  _ci_mean(gdf["delta_var_time"].values.astype(np.float64)),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    _atomic_write_csv(out, summary_csv)
    return out
