# utils/experiment_plotting_metrics/w3_plotting_metrics.py
# ============================================================
# 저장 규칙: summary / detail 분리, append+dedupe, 원자 저장
# 키 규칙: W1/W2와 동일 + perturbation
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import math
import time

_W3_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed",
    "perturbation","model_tag","run_tag"
]

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

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
        # 컬럼 union
        for c in new.columns:
            if c not in old.columns: old[c] = np.nan
        for c in old.columns:
            if c not in new.columns: new[c] = np.nan
        # 중복 덮어쓰기
        mask = np.ones(len(old), dtype=bool)
        for k in subset_keys:
            if k in old.columns:
                mask &= (old[k] == row.get(k))
        out = pd.concat([old[~mask], new], ignore_index=True)
        _atomic_write_csv(out, path)
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

def _f32(x, ndigits=6):
    try:
        v = float(x)
        if not np.isfinite(v): return np.nan
        return float(np.round(np.float32(v), ndigits))
    except Exception:
        return np.nan

# ------------------------------------------------------------
# 1) 실행 raw 저장 (baseline/perturb 공통)
# ------------------------------------------------------------
def save_w3_run_detail_row(
    ctx: Dict[str,Any],   # plot_type='w3_run'
    rmse_real: float,
    mae_real: float,
    other_cols: Optional[Dict[str,Any]],
    detail_csv: str | Path
) -> None:
    """
    results_W3.csv와 동일 맥락의 실행 단위 raw를 detail로도 남겨둠(재현/검증용).
    """
    row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "rmse_real": _f32(rmse_real),
        "mae_real":  _f32(mae_real),
    }
    if other_cols:
        for k, v in other_cols.items():
            # 숫자 위주만 저장; 문자열은 필요 최소
            if isinstance(v, (int, float, np.floating)):
                row[k] = _f32(v)
    _append_or_update_row(detail_csv, row, subset_keys=_W3_KEYS)

# ------------------------------------------------------------
# 2) 개입 효과(Δ) 바 플롯 데이터: summary + detail
#    - ΔRMSE%, ΔTOD/ΔPEAK 핵심 지표, d, d_z
# ------------------------------------------------------------
def save_w3_perturb_delta_bar(
    ctx: Dict[str,Any],               # plot_type='perturb_delta_bar'
    effect: Dict[str,Any],            # compute_w3_metrics 결과
    out_dir: str | Path
) -> None:
    out_dir = Path(out_dir)
    # summary(1행)
    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "delta_rmse":       _f32(effect.get("w3_delta_rmse")),
        "delta_rmse_pct":   _f32(effect.get("w3_delta_rmse_pct")),
        "delta_mae":        _f32(effect.get("w3_delta_mae")),
        # TOD
        "delta_gc_kernel_tod_dcor": _f32(effect.get("w3_delta_gc_kernel_tod_dcor")),
        "delta_gc_feat_tod_dcor":   _f32(effect.get("w3_delta_gc_feat_tod_dcor")),
        "delta_gc_feat_tod_r2":     _f32(effect.get("w3_delta_gc_feat_tod_r2")),
        # PEAK
        "delta_cg_event_gain":      _f32(effect.get("w3_delta_cg_event_gain")),
        "delta_cg_bestlag_abs":     _f32(effect.get("w3_delta_cg_bestlag_abs")),
        "delta_cg_spearman_mean":   _f32(effect.get("w3_delta_cg_spearman_mean")),
        # Effect size
        "effect_d":  _f32(effect.get("w3_effect_d")),
        "effect_dz": _f32(effect.get("w3_effect_dz")),
        "dz_lo":     _f32(effect.get("w3_dz_lo")),
        "dz_hi":     _f32(effect.get("w3_dz_hi")),
        # Hypothesis flags
        "tod_support":   _f32(effect.get("w3_hypothesis_tod_support")),
        "peak_support":  _f32(effect.get("w3_hypothesis_peak_support")),
    }
    summary_csv = out_dir / "perturb_delta_bar_summary.csv"
    _append_or_update_row(summary_csv, summary_row, subset_keys=_W3_KEYS)

    # detail(long): metric, value
    detail_rows: List[Dict[str,Any]] = []
    for mk, col in [
        ("delta_rmse_pct", "delta_rmse_pct"),
        ("delta_rmse",     "delta_rmse"),
        ("delta_mae",      "delta_mae"),
        ("delta_gc_kernel_tod_dcor","delta_gc_kernel_tod_dcor"),
        ("delta_gc_feat_tod_dcor","delta_gc_feat_tod_dcor"),
        ("delta_gc_feat_tod_r2","delta_gc_feat_tod_r2"),
        ("delta_cg_event_gain","delta_cg_event_gain"),
        ("delta_cg_bestlag_abs","delta_cg_bestlag_abs"),
        ("delta_cg_spearman_mean","delta_cg_spearman_mean"),
        ("effect_d","effect_d"),("effect_dz","effect_dz")
    ]:
        v = summary_row.get(col, np.nan)
        if np.isfinite(v):
            detail_rows.append({
                **{k: ctx[k] for k in _W3_KEYS if k in ctx},
                "metric": mk, "value": _f32(v)
            })
    if detail_rows:
        detail_csv = out_dir / "perturb_delta_bar_detail.csv"
        _append_or_update_rows(detail_csv, detail_rows, subset_keys=_W3_KEYS + ["metric"])

# ------------------------------------------------------------
# 3) 순위 보존(rank) 저장: summary + detail
# ------------------------------------------------------------
def save_w3_rank_preservation(
    ctx: Dict[str,Any],               # plot_type='rank_preservation'
    effect: Dict[str,Any],            # compute_w3_metrics 결과
    out_dir: str | Path
) -> None:
    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "spearman_rho":  _f32(effect.get("w3_rank_spearman")),
        "topk_preserve": _f32(effect.get("w3_rank_topk_preserve")),
    }
    summary_csv = Path(out_dir) / "rank_preservation_summary.csv"
    _append_or_update_row(summary_csv, summary_row, subset_keys=_W3_KEYS)

# ------------------------------------------------------------
# 4) 라그 변화(PEAK 관련): summary + detail
# ------------------------------------------------------------
def save_w3_lag_shift(
    ctx: Dict[str,Any],               # plot_type='lag_shift'
    base_bestlag: float,
    pert_bestlag: float,
    out_dir: str | Path
) -> None:
    delta_abs = np.nan
    if np.isfinite(base_bestlag) and np.isfinite(pert_bestlag):
        delta_abs = abs(float(pert_bestlag)) - abs(float(base_bestlag))
    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "base_bestlag": _f32(base_bestlag),
        "pert_bestlag": _f32(pert_bestlag),
        "delta_abs":    _f32(delta_abs),
    }
    summary_csv = Path(out_dir) / "lag_shift_summary.csv"
    _append_or_update_row(summary_csv, summary_row, subset_keys=_W3_KEYS)

# ------------------------------------------------------------
# 5) 그룹 집계(시드 평균) - Δ 막대 요약 재작성
# ------------------------------------------------------------
def rebuild_w3_perturb_summary(
    delta_detail_csv: str | Path,     # perturb_delta_bar_detail.csv
    delta_summary_csv: str | Path,    # perturb_delta_bar_summary.csv (출력 재작성)
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon","perturbation"),
) -> pd.DataFrame:
    dpath = Path(delta_detail_csv)
    if not dpath.exists():
        out = pd.DataFrame(columns=list(group_keys)+["metric","value_mean"])
        _atomic_write_csv(out, delta_summary_csv)
        return out
    df = pd.read_csv(dpath)
    need = set(group_keys) | {"metric","value"}
    if not need.issubset(df.columns):
        _atomic_write_csv(pd.DataFrame(columns=list(need)), delta_summary_csv)
        return pd.DataFrame(columns=list(need))
    rows: List[Dict[str,Any]] = []
    for keys, g in df.groupby(list(group_keys), group_keys=False):
        g2 = g.copy()
        for mk, s in g2.groupby("metric", group_keys=False):
            rows.append({**{k:v for k,v in zip(group_keys, keys)}, "metric": mk, "value_mean": float(np.nanmean(s["value"]))})
    out = pd.DataFrame(rows)
    _atomic_write_csv(out, delta_summary_csv)
    return out
