# utils/experiment_plotting_metrics/all_plotting_metrics.py
# =====================================================================
# 목적
#   - W1~W5 공통: "보고용 그림 생성"에 필요한 추가 지표 계산과
#     CSV 저장 유틸리티(원자적 저장, 중복 덮어쓰기)를 단일 모듈로 통합합니다.
#   - 기존 plotting_metrics.py / plot_results.py 의 역할을 대체합니다.
#   - W1에서 즉시 필요한 CKA/Grad-Align 계산과 Forest(Δ/TopPick) 집계를 포함합니다.
# 사용 예시
#   from utils.experiment_plotting_metrics.all_plotting_metrics import (
#       compute_layerwise_cka, compute_gradient_alignment, build_forest_from_results,
#       append_or_update_row, append_or_update_long
#   )
# =====================================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------
# 공통 키/상수
# ---------------------------
REQUIRED_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode"]


# ---------------------------
# 공통 저장 유틸 (원자 저장 + 덮어쓰기)
# ---------------------------

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def append_or_update_row(path: str | Path, row: Dict[str, Any], subset_keys: List[str]) -> None:
    """한 행 덮어쓰기 저장(중복 키 제거 후 append)."""
    p = Path(path); _ensure_dir(p)
    new_df = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p)
        mask = np.ones(len(old), dtype=bool)
        for k in subset_keys:
            mask &= (old[k] == row[k])
        out = pd.concat([old[~mask], new_df], ignore_index=True)
    else:
        out = new_df
    _atomic_write_csv(out, p)

def append_or_update_long(path: str | Path, rows: List[Dict[str, Any]], subset_keys: List[str]) -> None:
    """여러 행 append + 중복제거 저장."""
    if not rows:
        return
    p = Path(path); _ensure_dir(p)
    new_df = pd.DataFrame(rows)
    if p.exists():
        old = pd.read_csv(p)
        out = pd.concat([old, new_df], ignore_index=True)
        out.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
    else:
        out = new_df
    _atomic_write_csv(out, p)


# ---------------------------
# 공통 수학/전처리 유틸
# ---------------------------

def _to_2d(x: np.ndarray) -> np.ndarray:
    """(samples, dim) 2D로 평탄화. 허용: (B,T,C)/(T,B,C)/(B,C)/(N,D)/(…)"""
    x = np.asarray(x)
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        return x.reshape(-1, x.shape[-1])
    if x.ndim == 1:
        return x[:, None]
    return x.reshape(-1, x.shape[-1])

def _split_bins_auto(n_layers: int) -> List[str]:
    """레이어 수를 3등분하여 ['S','M','D'] 라벨 부여."""
    if n_layers <= 0:
        return []
    b1 = int(np.floor(n_layers/3))
    b2 = int(np.floor(2*n_layers/3))
    return ["S" if i < b1 else "M" if i < b2 else "D" for i in range(n_layers)]

def _center(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X)
    X = X - np.nanmean(X, axis=0, keepdims=True)
    return X

def _hsic_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """Centered linear HSIC."""
    Xc = _center(X); Yc = _center(Y)
    mask = np.isfinite(Xc).all(axis=1) & np.isfinite(Yc).all(axis=1)
    if mask.sum() < 2:
        return np.nan
    Xc = Xc[mask]; Yc = Yc[mask]
    K = Xc @ Xc.T
    L = Yc @ Yc.T
    n = K.shape[0]
    if n < 2:
        return np.nan
    H = np.eye(n) - np.ones((n,n))/n
    KH = H @ K @ H
    LH = H @ L @ H
    return float(np.sum(KH * LH) / ((n-1)**2))

def _cka_linear(X: np.ndarray, Y: np.ndarray) -> float:
    """선형 CKA (HSIC 정규화)."""
    hs_xy = _hsic_linear(X, Y)
    hs_xx = _hsic_linear(X, X)
    hs_yy = _hsic_linear(Y, Y)
    if not np.isfinite(hs_xy) or not np.isfinite(hs_xx) or not np.isfinite(hs_yy):
        return np.nan
    denom = np.sqrt(hs_xx * hs_yy)
    if denom <= 0 or not np.isfinite(denom):
        return np.nan
    return float(np.clip(hs_xy / denom, -1.0, 1.0))

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    ma = np.linalg.norm(a); mb = np.linalg.norm(b)
    if ma == 0 or mb == 0 or not np.isfinite(ma) or not np.isfinite(mb):
        return np.nan
    return float(np.clip(np.dot(a,b) / (ma*mb), -1.0, 1.0))

def _nanmean(a) -> float:
    try:
        return float(np.nanmean(a))
    except Exception:
        return np.nan

def _zscore(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    m = x.mean()
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - m) / sd


# ---------------------------
# W1 공통 계산: CKA / Grad-Align
# ---------------------------

def compute_layerwise_cka(
    feature_dict: Dict[str, Sequence[np.ndarray]],
    cnn_key: str = "cnn",
    gru_key: str = "gru",
    max_samples: Optional[int] = 20000,
    depth_bins_cnn: Optional[List[str]] = None,
    depth_bins_gru: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    feature_dict['cnn'] = [feat_l0, feat_l1, ...]  # (N,D) or (B,T,C)
    feature_dict['gru'] = [feat_l0, feat_l1, ...]
    반환: {'cka_matrix': 2D list, 'cnn_depth_bins': [...], 'gru_depth_bins': [...]}
    """
    cnn_feats = feature_dict.get(cnn_key); gru_feats = feature_dict.get(gru_key)
    if cnn_feats is None or gru_feats is None:
        raise ValueError("feature_dict must contain 'cnn' and 'gru' lists.")
    n_c, n_g = len(cnn_feats), len(gru_feats)
    if n_c == 0 or n_g == 0:
        return {"cka_matrix": [], "cnn_depth_bins": [], "gru_depth_bins": []}

    def _prep(F):
        X = _to_2d(np.asarray(F))
        if max_samples and X.shape[0] > max_samples:
            idx = np.linspace(0, X.shape[0]-1, max_samples).astype(int)
            X = X[idx]
        valid = np.isfinite(X).any(axis=0)
        if valid.sum() == 0:
            return np.zeros((X.shape[0],1), dtype=float)
        return X[:, valid]

    M = np.full((n_c, n_g), np.nan, dtype=float)
    C = [ _prep(f) for f in cnn_feats ]
    G = [ _prep(f) for f in gru_feats ]
    for i in range(n_c):
        for j in range(n_g):
            try:
                M[i,j] = _cka_linear(C[i], G[j])
            except Exception:
                M[i,j] = np.nan

    bins_c = depth_bins_cnn or _split_bins_auto(n_c)
    bins_g = depth_bins_gru or _split_bins_auto(n_g)
    return {
        "cka_matrix": M.tolist(),
        "cnn_depth_bins": bins_c,
        "gru_depth_bins": bins_g
    }


def compute_gradient_alignment(
    grad_dict: Optional[Dict[str, Sequence[np.ndarray]]] = None,
    cnn_key: str = "cnn",
    gru_key: str = "gru",
    depth_bins: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    grad_dict['cnn'] = [g_l0, g_l1, ...]  # (D,) 또는 (D,1)
    grad_dict['gru'] = [g_l0, g_l1, ...]
    반환: {'grad_align_by_layer': [{layer_idx,depth_bin,grad_align_value}, ...],
           'grad_align_s','grad_align_m','grad_align_d','grad_align_mean'}
    """
    if grad_dict is None or cnn_key not in grad_dict or gru_key not in grad_dict:
        return {
            "grad_align_by_layer": [],
            "grad_align_s": np.nan,
            "grad_align_m": np.nan,
            "grad_align_d": np.nan,
            "grad_align_mean": np.nan,
        }
    gc, gg = grad_dict[cnn_key], grad_dict[gru_key]
    L = min(len(gc), len(gg))
    if L == 0:
        return {
            "grad_align_by_layer": [],
            "grad_align_s": np.nan,
            "grad_align_m": np.nan,
            "grad_align_d": np.nan,
            "grad_align_mean": np.nan,
        }
    if depth_bins is None:
        depth_bins = _split_bins_auto(L)

    rows, vals = [], []
    for i in range(L):
        a = np.ravel(np.asarray(gc[i], dtype=float))
        b = np.ravel(np.asarray(gg[i], dtype=float))
        v = _cosine(a, b)
        rows.append({"layer_idx": i, "depth_bin": depth_bins[i], "grad_align_value": v})
        if np.isfinite(v): vals.append(v)

    df = pd.DataFrame(rows)
    def _m(b):
        s = df.loc[df["depth_bin"]==b, "grad_align_value"]
        return float(np.nanmean(s)) if len(s) else np.nan

    return {
        "grad_align_by_layer": rows,
        "grad_align_s": _m("S"),
        "grad_align_m": _m("M"),
        "grad_align_d": _m("D"),
        "grad_align_mean": float(np.nanmean(vals)) if len(vals) else np.nan,
    }


# ---------------------------
# Forest Δ/Top-pick 집계 (공통화; W1에서 사용)
# ---------------------------

def build_forest_from_results(
    experiment_type: str,
    results_csv: str | Path,
    out_dir: str | Path,
    per_mode: str = "per_layer",
    last_mode: str = "last_layer",
    rmse_col_pref: Tuple[str, ...] = ("rmse_real","rmse"),
    cka_col: str = "cka_full_mean",
    grad_col: str = "grad_align_mean"
) -> None:
    """
    results_{exp}.csv를 읽어 per vs last를 조인하여
     - summary: Δ지표 + Top-pick 점수
     - detail: tidy(long-form) 생성.
    """
    results_csv = Path(results_csv)
    if not results_csv.exists():
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = out_dir / "forest_plot_summary.csv"
    detail_csv  = out_dir / "forest_plot_detail.csv"

    df = pd.read_csv(results_csv)
    if not set(["dataset","horizon","seed","mode"]).issubset(df.columns):
        return

    rmse_col = next((c for c in rmse_col_pref if c in df.columns), None)
    if rmse_col is None:
        return

    per  = df[df["mode"]==per_mode].copy()
    last = df[df["mode"]==last_mode].copy()
    if len(per)==0 or len(last)==0:
        return

    join_keys = ["dataset","horizon","seed"]
    base = per.merge(last, on=join_keys, suffixes=("_per","_last"))

    # Δ 계산 (성능: 감소가 향상 → Δ=last - per)
    base["delta_rmse"]     = base[f"{rmse_col}_last"] - base[f"{rmse_col}_per"]
    base["delta_rmse_pct"] = 100.0 * base["delta_rmse"] / base[f"{rmse_col}_last"]

    def _delta_or_nan(name_root: str) -> pd.Series:
        a = f"{name_root}_per"; b = f"{name_root}_last"
        if a in base.columns and b in base.columns:
            return base[a] - base[b]
        return pd.Series(np.nan, index=base.index)

    if cka_col:
        base["delta_cka_global"] = _delta_or_nan(cka_col)
    if grad_col:
        base["delta_grad_align_all"] = _delta_or_nan(grad_col)

    # Top-pick(무가중치 z-합; 그룹=dataset×horizon)
    grouped_frames = []
    for _, grp in base.groupby(["dataset", "horizon"], sort=False):
        grp = grp.copy()
        z_rmse = _zscore(grp["delta_rmse_pct"])
        z_cka  = _zscore(grp["delta_cka_global"]) if "delta_cka_global" in grp else pd.Series(0, index=grp.index)
        z_gal  = _zscore(grp["delta_grad_align_all"]) if "delta_grad_align_all" in grp else pd.Series(0, index=grp.index)
        grp["top_pick_score"] = z_rmse + z_cka + z_gal
        grouped_frames.append(grp)
    if grouped_frames:
        base = pd.concat(grouped_frames, ignore_index=True)
    else:
        base = base.assign(top_pick_score=np.nan)

    # summary
    cols = [
        "dataset","horizon","seed",
        f"{rmse_col}_per", f"{rmse_col}_last",
        "delta_rmse","delta_rmse_pct"
    ]
    if "delta_cka_global" in base.columns: cols.append("delta_cka_global")
    if "delta_grad_align_all" in base.columns: cols.append("delta_grad_align_all")
    cols.append("top_pick_score")

    summary = base[cols].copy()
    summary.insert(0, "experiment_type", experiment_type)
    summary.insert(1, "plot_type", "forest_plot")
    _atomic_write_csv(summary, summary_csv)

    # detail (tidy)
    rows = []
    for _, r in base.iterrows():
        ctx = {
            "experiment_type": experiment_type,
            "plot_type": "forest_plot",
            "dataset": r["dataset"],
            "horizon": int(r["horizon"]),
            "seed": int(r["seed"]),
        }
        rows += [
            {**ctx, "metric": f"{rmse_col}_per",  "value": float(r.get(f"{rmse_col}_per", np.nan))},
            {**ctx, "metric": f"{rmse_col}_last", "value": float(r.get(f"{rmse_col}_last", np.nan))},
            {**ctx, "metric": "delta_rmse",       "value": float(r.get("delta_rmse", np.nan))},
            {**ctx, "metric": "delta_rmse_pct",   "value": float(r.get("delta_rmse_pct", np.nan))}
        ]
        if "delta_cka_global" in base.columns:
            if f"{cka_col}_per" in base.columns:
                rows.append({**ctx, "metric": f"{cka_col}_per",  "value": float(r.get(f"{cka_col}_per", np.nan))})
            if f"{cka_col}_last" in base.columns:
                rows.append({**ctx, "metric": f"{cka_col}_last", "value": float(r.get(f"{cka_col}_last", np.nan))})
            rows.append({**ctx, "metric": "delta_cka_global", "value": float(r.get("delta_cka_global", np.nan))})
        if "delta_grad_align_all" in base.columns:
            if f"{grad_col}_per" in base.columns:
                rows.append({**ctx, "metric": f"{grad_col}_per",  "value": float(r.get(f"{grad_col}_per", np.nan))})
            if f"{grad_col}_last" in base.columns:
                rows.append({**ctx, "metric": f"{grad_col}_last", "value": float(r.get(f"{grad_col}_last", np.nan))})
            rows.append({**ctx, "metric": "delta_grad_align_all", "value": float(r.get("delta_grad_align_all", np.nan))})

    detail = pd.DataFrame(rows)
    _atomic_write_csv(detail, detail_csv)
