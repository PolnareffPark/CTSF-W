# ============================================================
# utils/experiment_plotting_metrics/w3_plotting_metrics.py
#   - W3 전용: Gate-TOD Heatmap / Gate Distribution / Forest Δ / Perturb Δ Bar
#   - summary + detail (append+dedupe, atomic write)
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import pandas as pd

_W3_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag"]

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
        f = float(v)
        if not np.isfinite(f): return np.nan
        return float(np.round(np.float32(f), ndigits))
    except Exception:
        return np.nan

def _zscore(x: pd.Series) -> pd.Series:
    s = x.astype(float)
    m = s.mean(); sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd


# ============================================================
# 1) Gate‑TOD Heatmap (summary + detail)
# ============================================================
def compute_and_save_w3_gate_tod_heatmap(
    ctx: Dict[str,Any],                    # plot_type='gate_tod_heatmap'
    gates_by_stage: Dict[str,np.ndarray],  # {'S':arr, 'M':arr, 'D':arr} (시간축 포함)
    tod_labels: List[int],
    out_dir: str | Path,
    amp_method: str = "range",             # 'range' | 'std'
) -> None:
    assert ctx.get("plot_type") == "gate_tod_heatmap"
    def _collapse_time(a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, float)
        if a.ndim == 1: return a
        # 마지막 축을 시간축으로 가정: 다른 축 평균
        axes = tuple(i for i in range(a.ndim-1))
        return np.nanmean(a, axis=axes)

    S = np.asarray(gates_by_stage.get("S"), float) if "S" in gates_by_stage else None
    M = np.asarray(gates_by_stage.get("M"), float) if "M" in gates_by_stage else None
    D = np.asarray(gates_by_stage.get("D"), float) if "D" in gates_by_stage else None

    s = _collapse_time(S) if S is not None else np.array([], float)
    m = _collapse_time(M) if M is not None else np.array([], float)
    d = _collapse_time(D) if D is not None else np.array([], float)

    T = max(len(s), len(m), len(d))
    if len(tod_labels) != T:
        tod_labels = list(range(T))

    def _amp(x: np.ndarray) -> float:
        if x.size == 0 or not np.isfinite(x).any(): return np.nan
        return _to_f32(np.nanstd(x) if amp_method=="std" else (np.nanmax(x)-np.nanmin(x)))

    amp_s, amp_m, amp_d = _amp(s), _amp(m), _amp(d)
    amp_mean = _to_f32(np.nanmean([v for v in [amp_s,amp_m,amp_d] if np.isfinite(v)]))

    all_vals = np.concatenate([v for v in [s,m,d] if v.size>0], axis=0) if T>0 else np.array([],float)
    valid_ratio = _to_f32(np.isfinite(all_vals).mean()) if all_vals.size>0 else np.nan
    nan_count   = int(all_vals.size - int(np.isfinite(all_vals).sum()))

    # summary
    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "tod_bins": int(T),
        "tod_amp_s": amp_s, "tod_amp_m": amp_m, "tod_amp_d": amp_d,
        "tod_amp_mean": amp_mean,
        "valid_ratio": valid_ratio, "nan_count": nan_count,
        "amp_method": amp_method
    }
    _append_or_update_row(Path(out_dir)/"gate_tod_heatmap_summary.csv", summary_row, subset_keys=_W3_KEYS)

    # detail
    rows: List[Dict[str,Any]] = []
    def _emit(stage: str, arr: np.ndarray):
        for t, v in enumerate(arr):
            if not np.isfinite(v): continue
            rows.append({
                **{k: ctx[k] for k in _W3_KEYS if k in ctx},
                "tod": int(tod_labels[t]), "stage": stage, "gate_mean": _to_f32(v)
            })
    if s.size: _emit("S", s)
    if m.size: _emit("M", m)
    if d.size: _emit("D", d)
    _append_or_update_rows(Path(out_dir)/"gate_tod_heatmap_detail.csv", rows,
                           subset_keys=_W3_KEYS+["tod","stage"])


# ============================================================
# 2) Gate Distribution (summary + detail)
# ============================================================
def compute_and_save_w3_gate_distribution(
    ctx: Dict[str,Any],                    # plot_type='gate_distribution'
    gates_by_stage: Dict[str,np.ndarray],
    gates_all: np.ndarray,
    out_dir: str | Path,
    quantile_keys: Optional[List[str]] = None
) -> None:
    assert ctx.get("plot_type") == "gate_distribution"
    g = np.asarray(gates_all, float) if isinstance(gates_all, np.ndarray) else np.array([], float)

    g_mean = _to_f32(np.nanmean(g)) if g.size>0 and np.isfinite(g).any() else np.nan
    g_std  = _to_f32(np.nanstd(g))  if g.size>0 and np.isfinite(g).any() else np.nan
    g_ent  = np.nan
    if g.size>0:
        pos = g[np.isfinite(g) & (g > 1e-8)]
        if pos.size>0:
            p = pos / (pos.sum() + 1e-12)
            g_ent = _to_f32(-np.sum(p*np.log(p + 1e-12)))
    valid_ratio = _to_f32(np.isfinite(g).mean()) if g.size>0 else np.nan
    nan_count   = int(g.size - int(np.isfinite(g).sum()))

    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "gate_mean": g_mean, "gate_std": g_std, "gate_entropy": g_ent,
        "valid_ratio": valid_ratio, "nan_count": nan_count
    }
    _append_or_update_row(Path(out_dir)/"gate_distribution_summary.csv", summary_row, subset_keys=_W3_KEYS)

    if quantile_keys is None:
        quantile_keys = ["gate_q05","gate_q10","gate_q25","gate_q50","gate_q75","gate_q90","gate_q95"]
    qvals = {}
    if g.size>0 and np.isfinite(g).any():
        qs = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
        vals = np.quantile(g[np.isfinite(g)], qs).tolist()
        qvals = dict(zip(quantile_keys, [ _to_f32(v) for v in vals ]))
    q_rows: List[Dict[str,Any]] = []
    for k,v in qvals.items():
        q_rows.append({ **{k2: ctx[k2] for k2 in _W3_KEYS if k2 in ctx}, "stat": k, "value": v })
    if q_rows:
        _append_or_update_rows(Path(out_dir)/"gate_distribution_quantiles_detail.csv",
                               q_rows, subset_keys=_W3_KEYS+["stat"])


# ============================================================
# 3) Forest Δ (none vs perturbed)
# ============================================================
def save_w3_forest_detail_row(
    ctx: Dict[str,Any], rmse_real: float, mae_real: float, detail_csv: str | Path
) -> None:
    row = { **{k:ctx[k] for k in _W3_KEYS if k in ctx},
            "rmse_real": _to_f32(rmse_real), "mae_real": _to_f32(mae_real) }
    _append_or_update_row(detail_csv, row, subset_keys=_W3_KEYS)

def rebuild_w3_forest_summary(
    detail_csv: str | Path,
    summary_csv: str | Path,
    gate_tod_summary_csv: Optional[str | Path] = None,
    gate_dist_summary_csv: Optional[str | Path] = None,
    baseline_mode: str = "none",
    perturbed_modes: Tuple[str,...] = ("tod_shift","smooth"),
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon")
) -> pd.DataFrame:
    detail_csv = Path(detail_csv)
    if not detail_csv.exists():
        out = pd.DataFrame(columns=list(group_keys)+["perturbation","n_pairs",
                            "delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean",
                            "delta_tod_mean","delta_var_time_mean"])
        _atomic_write_csv(out, summary_csv)
        return out

    df = pd.read_csv(detail_csv)
    need = set(group_keys) | {"seed","mode","rmse_real","mae_real"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in forest detail csv: {miss}")

    rows: List[pd.DataFrame] = []
    on = list(group_keys)+["seed"]

    # 보조 요약 join 준비
    tdf = pd.read_csv(gate_tod_summary_csv) if (gate_tod_summary_csv and Path(gate_tod_summary_csv).exists()) else None
    vdf = pd.read_csv(gate_dist_summary_csv) if (gate_dist_summary_csv and Path(gate_dist_summary_csv).exists()) else None

    for pert in perturbed_modes:
        per = df[df["mode"]==pert].copy()
        base= df[df["mode"]==baseline_mode].copy()
        if len(per)==0 or len(base)==0:
            continue
        base = per.merge(base, on=on, suffixes=("_per","_base"))

        # TOD join
        if isinstance(tdf, pd.DataFrame):
            need_t = set(group_keys) | {"seed","mode","tod_amp_mean"}
            if need_t.issubset(set(tdf.columns)):
                t_per  = tdf[tdf["mode"]==pert][list(group_keys)+["seed","tod_amp_mean"]].rename(columns={"tod_amp_mean":"tod_amp_mean_per"})
                t_base = tdf[tdf["mode"]==baseline_mode][list(group_keys)+["seed","tod_amp_mean"]].rename(columns={"tod_amp_mean":"tod_amp_mean_base"})
                base = base.merge(t_per,on=on,how="left").merge(t_base,on=on,how="left")

        # VarT join
        if isinstance(vdf, pd.DataFrame):
            # W3 distribution summary에는 gate_var_time 명시가 없으므로 생략/혹은 gate_std를 대체지표로 사용 가능
            pass

        base["delta_rmse"] = (base["rmse_real_per"] - base["rmse_real_base"]).astype(np.float32)  # 교란으로 악화(+) 기대
        base["delta_rmse_pct"] = (base["delta_rmse"] / base["rmse_real_base"]).astype(np.float32) * 100.0
        base["delta_mae"]  = (base["mae_real_per"]  - base["mae_real_base"]).astype(np.float32)

        if "tod_amp_mean_per" in base.columns and "tod_amp_mean_base" in base.columns:
            base["delta_tod"] = (base["tod_amp_mean_per"] - base["tod_amp_mean_base"]).astype(np.float32)
        else:
            base["delta_tod"] = np.nan

        base["perturbation"] = pert
        rows.append(base)

    if not rows:
        out = pd.DataFrame(columns=list(group_keys)+["perturbation","n_pairs",
                            "delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean",
                            "delta_tod_mean"])
        _atomic_write_csv(out, summary_csv)
        return out

    base = pd.concat(rows, ignore_index=True)

    def _ci_mean(a: np.ndarray) -> Tuple[float,float,float]:
        a = a[np.isfinite(a)]
        if a.size==0: return (np.nan,np.nan,np.nan)
        m = float(np.mean(a))
        s = float(np.std(a, ddof=1)) if a.size>1 else 0.0
        se= s / math.sqrt(max(1,a.size))
        return (_to_f32(m), _to_f32(m-1.96*se), _to_f32(m+1.96*se))

    # groupby FutureWarning 회피: group_keys=False 명시
    agg = (base
           .groupby(list(group_keys)+["perturbation"], group_keys=False)
           .apply(lambda g: pd.Series({
               "n_pairs": int(len(g)),
               "delta_rmse_mean": _ci_mean(g["delta_rmse"].values)[0],
               "delta_rmse_pct_mean": _ci_mean(g["delta_rmse_pct"].values)[0],
               "delta_mae_mean":  _ci_mean(g["delta_mae"].values)[0],
               "delta_tod_mean":  _ci_mean(g["delta_tod"].values)[0],
           }))
           .reset_index())
    _atomic_write_csv(agg, summary_csv)
    return agg


# ============================================================
# 4) Perturb Δ Bar (원인 지표 Δ: TOD / PEAK)
# ============================================================
def rebuild_w3_perturb_delta_bar(
    results_csv: str | Path,
    out_dir: str | Path,
    baseline_mode: str = "none",
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon"),
) -> pd.DataFrame:
    """
    results_W3.csv에서 baseline('none')과 교란('tod_shift','smooth') 매칭하여
    원인 지표들의 Δ를 구해 bar용 detail/summary를 기록.
      - A4-TOD:  Δ(gc_kernel_tod_dcor), Δ(gc_feat_tod_dcor), Δ(gc_feat_tod_r2)  [음수 기대]
      - A4-PEAK: Δ(cg_event_gain), Δ(|cg_bestlag|), Δ(cg_spearman_mean)        [gain/spearman은 음수, |lag|는 양수 기대]
    """
    results_csv = Path(results_csv)
    if not results_csv.exists():
        return pd.DataFrame()
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)

    # 필요한 컬럼 검사(있으면 Δ 계산, 없으면 NaN)
    needed = set(group_keys) | {"seed","mode","cg_event_gain","cg_bestlag","cg_spearman_mean",
                                "gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2"}
    for c in (needed - set(df.columns)):
        df[c] = np.nan

    def _pair(pert: str, metrics: List[str], post: List[str]) -> pd.DataFrame:
        per = df[df["mode"]==pert].copy()
        base= df[df["mode"]==baseline_mode].copy()
        on = list(group_keys)+["seed"]
        if len(per)==0 or len(base)==0:
            return pd.DataFrame(columns=list(group_keys)+["seed","perturbation","metric","delta"])
        m = per.merge(base, on=on, suffixes=("_per","_base"))
        rows = []
        for met, sign in zip(metrics, post):
            x = m[f"{met}_per"].values
            y = m[f"{met}_base"].values
            if met=="cg_bestlag":
                # |lag| 기준으로 Δ = |per| - |base| (양수 기대)
                d = np.abs(x) - np.abs(y)
            else:
                d = x - y  # per - base
            for i in range(len(m)):
                rows.append({
                    **{k: m.iloc[i][k] for k in list(group_keys)},
                    "seed": int(m.iloc[i]["seed"]),
                    "perturbation": pert,
                    "metric": met,
                    "delta": _to_f32(d[i])
                })
        return pd.DataFrame(rows)

    # A4-TOD
    tod_df = _pair("tod_shift",
        metrics=["gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2"],
        post=["neg","neg","neg"]
    )
    # A4-PEAK
    peak_df = _pair("smooth",
        metrics=["cg_event_gain","cg_bestlag","cg_spearman_mean"],
        post=["neg","pos","neg"]
    )
    detail = pd.concat([tod_df, peak_df], ignore_index=True) if len(tod_df)+len(peak_df)>0 else pd.DataFrame(
        columns=list(group_keys)+["seed","perturbation","metric","delta"]
    )
    if len(detail)>0:
        _append_or_update_rows(Path(out_dir)/"perturb_delta_bar_detail.csv",
                               detail.to_dict("records"),
                               subset_keys=list(group_keys)+["seed","perturbation","metric"])

        # summary: 평균(±CI)
        def _ci_mean(a: np.ndarray) -> float:
            a = a[np.isfinite(a)]
            if a.size==0: return np.nan
            return _to_f32(float(np.mean(a)))
        summary = (detail
                   .groupby(list(group_keys)+["perturbation","metric"], group_keys=False)
                   .agg(delta_mean=("delta", _ci_mean), n=("delta","size"))
                   .reset_index())
        _atomic_write_csv(summary, Path(out_dir)/"perturb_delta_bar_summary.csv")
        return summary
    else:
        _atomic_write_csv(pd.DataFrame(columns=list(group_keys)+["perturbation","metric","delta_mean","n"]),
                          Path(out_dir)/"perturb_delta_bar_summary.csv")
        return detail
