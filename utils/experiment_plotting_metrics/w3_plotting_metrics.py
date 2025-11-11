# ============================================================
# utils/experiment_plotting_metrics/w3_plotting_metrics.py
#   - W3 전용: Forest Plot(Δ집계) + Perturbation ΔBar 저장
#   - summary + detail (append + dedupe + atomic write)
#   - pandas futurewarning 회피: include_groups=False 지원 시 사용
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import math
import numpy as np
import pandas as pd

_W3_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed","mode",
    "model_tag","run_tag","perturb_type","condition"  # baseline / perturbed
]

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

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

# =============== 1) Forest raw(detail) ===============
def save_w3_forest_detail_row(
    ctx: Dict[str,Any],             # plot_type='forest_plot'
    rmse_real: float,
    mae_real: float,
    detail_csv: str | Path
) -> None:
    """
    baseline/perturbed 각 1행씩 저장(조건은 ctx['condition']).
    """
    assert ctx.get("plot_type") == "forest_plot"
    row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "rmse_real": _to_f32(rmse_real),
        "mae_real":  _to_f32(mae_real),
    }
    _append_or_update_row(detail_csv, row, subset_keys=_W3_KEYS)

# =============== 2) Forest summary(Δ/CI/Top-pick) ===============
def rebuild_w3_forest_summary(
    detail_csv: str | Path,          # 입력 raw(detail)
    summary_csv: str | Path,         # 출력 summary
    per_cond: str = "perturbed",
    last_cond: str = "baseline",
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon","perturb_type")
) -> pd.DataFrame:
    """
    baseline↔perturbed 매칭하여 Δ지표 집계.
    Δ 정의: ΔRMSE = rmse_pert - rmse_base (양수=성능저하), ΔMAE 유사, ΔRMSE% = ΔRMSE / rmse_base * 100
    groupby.apply의 FutureWarning 회피: include_groups=False 사용(미지원 시 안전 폴백).
    """
    detail_csv = Path(detail_csv)
    if not detail_csv.exists():
        out = pd.DataFrame(columns=list(group_keys)+[
            "n_pairs","delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean","top_pick_score_w3_mean"
        ])
        _atomic_write_csv(out, summary_csv); return out

    df = pd.read_csv(detail_csv)
    need = set(group_keys) | {"seed","mode","condition","rmse_real","mae_real"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in forest detail csv: {miss}")

    per = df[df["condition"]==per_cond].copy()
    lst = df[df["condition"]==last_cond].copy()
    on = list(group_keys) + ["seed","mode"]
    base = per.merge(lst, on=on, suffixes=("_per","_base"))
    if len(base)==0:
        out = pd.DataFrame(columns=list(group_keys)+[
            "n_pairs","delta_rmse_mean","delta_rmse_pct_mean","delta_mae_mean","top_pick_score_w3_mean"
        ])
        _atomic_write_csv(out, summary_csv); return out

    base["delta_rmse"] = (base["rmse_real_per"] - base["rmse_real_base"]).astype(np.float32)
    base["delta_rmse_pct"] = (base["delta_rmse"] / base["rmse_real_base"]).astype(np.float32) * 100.0
    base["delta_mae"]  = (base["mae_real_per"]  - base["mae_real_base"]).astype(np.float32)

    # 그룹별 top-pick 점수(z‑sum). FutureWarning 방지 처리 포함.
    def _top_score(gr: pd.DataFrame) -> pd.DataFrame:
        z_rmsp = _zscore(gr["delta_rmse_pct"].fillna(0.0))
        gr["top_pick_score_w3"] = z_rmsp  # W3 기본은 ΔRMSE% 중심(필요 시 다른 Δ 추가 가능)
        return gr

    try:
        base = base.groupby(list(group_keys), group_keys=False).apply(_top_score, include_groups=False)
    except TypeError:
        base = base.groupby(list(group_keys), group_keys=False).apply(_top_score)

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
            "top_pick_score_w3_mean": _ci_mean(gdf["top_pick_score_w3"].values.astype(np.float64)),
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    _atomic_write_csv(out, summary_csv)
    return out

# =============== 3) Perturbation ΔBar (요약/상세) ===============
def save_w3_perturb_delta_bar(
    ctx: Dict[str,Any],               # plot_type='perturb_delta_bar'
    metrics: Dict[str,Any],           # 기대: w3_delta_rmse_pct (필수), 선택: 기타 Δ*
    out_dir: str | Path
) -> None:
    """
    바 플롯용 요약/상세 데이터 저장. (세부 플로팅은 노트북/스크립트에서 재현)
    summary: delta_rmse_pct / delta_rmse / delta_mae / top_pick 점수(있으면)
    detail: (metric, value) long form
    """
    assert ctx.get("plot_type") == "perturb_delta_bar"
    summary_csv = Path(out_dir) / "perturb_delta_bar_summary.csv"
    detail_csv  = Path(out_dir) / "perturb_delta_bar_detail.csv"

    row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "delta_rmse_pct": _to_f32(metrics.get("w3_delta_rmse_pct")),
        "delta_rmse":     _to_f32(metrics.get("w3_delta_rmse")),
        "delta_mae":      _to_f32(metrics.get("w3_delta_mae")),
        "delta_tod_align":_to_f32(metrics.get("w3_delta_tod_align")),
    }
    _append_or_update_row(summary_csv, row, subset_keys=_W3_KEYS)

    det_rows = []
    for k in ("w3_delta_rmse_pct","w3_delta_rmse","w3_delta_mae","w3_delta_tod_align",
              "w3_delta_gc_kernel_tod_dcor","w3_delta_gc_feat_tod_dcor","w3_delta_gc_feat_tod_r2"):
        if k in metrics and metrics[k] is not None:
            v = _to_f32(metrics.get(k))
            if np.isfinite(v):
                det_rows.append({**{kk: ctx[kk] for kk in _W3_KEYS if kk in ctx}, "metric": k, "value": v})
    if det_rows:
        _append_or_update_rows(detail_csv, det_rows, subset_keys=_W3_KEYS + ["metric"])
