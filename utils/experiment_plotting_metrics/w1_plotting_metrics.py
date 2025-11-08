# utils/experiment_plotting_metrics/w1_plotting_metrics.py
# =====================================================================
# 목적
#   - W1(Per-layer Cross vs Last-layer Fusion) 전용 "그림 저장 로직" 제공
#   - 각 그림은 summary + detail(long-form) 두 종류 CSV를 생성
#   - 계산 함수는 기본적으로 all_plotting_metrics 에서 import 하여 실제 호출
#
# 의존
#   - from utils.experiment_plotting_metrics.all_plotting_metrics import
#       compute_layerwise_cka, compute_gradient_alignment, build_forest_from_results,
#       append_or_update_row, append_or_update_long, REQUIRED_KEYS
#
# 사용 예시 (평가 루틴 내부)
#   from utils.experiment_plotting_metrics.w1_plotting_metrics import (
#       compute_and_save_w1_cka_heatmap,
#       compute_and_save_w1_grad_align_bar,
#       update_w1_forest_summary_and_detail
#   )
#
#   base_ctx = {
#     "experiment_type": "W1",
#     "dataset": dataset_name,
#     "horizon": int(CFG["horizon"]),
#     "seed": int(CFG["seed"]),
#     "mode": CFG["mode"],  # 'per_layer' or 'last_layer'
#   }
#
#   # 1) CKA: (요약/상세 모두 저장)
#   compute_and_save_w1_cka_heatmap(
#       ctx={**base_ctx, "plot_type": "cka_heatmap"},
#       feature_dict=feature_dict,                 # {'cnn':[...], 'gru':[...]} 형태
#       out_dir=f"results_W1/{dataset_name}/cka_heatmap"
#   )
#
#   # 2) Grad-Align: (요약/상세 모두 저장)
#   compute_and_save_w1_grad_align_bar(
#       ctx={**base_ctx, "plot_type": "grad_align_bar"},
#       grad_dict=grad_dict,                      # {'cnn':[grad_l], 'gru':[grad_l]} 형태
#       out_dir=f"results_W1/{dataset_name}/grad_align_bar"
#   )
#
#   # 3) Forest: (요약/상세 동시 생성; W1 전체 실행 로그를 기반으로 Δ/TopPick 산출)
#   update_w1_forest_summary_and_detail(
#       results_w1_csv="results_W1.csv",
#       out_dir=f"results_W1/{dataset_name}/forest_plot"
#   )
#
# =====================================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np

from utils.experiment_plotting_metrics.all_plotting_metrics import (
    REQUIRED_KEYS,
    append_or_update_row, append_or_update_long,
    compute_layerwise_cka, compute_gradient_alignment,
    build_forest_from_results
)

# -------------------------------
# 내부 유틸
# -------------------------------

def _split_bins_auto(n_layers: int):
    if n_layers <= 0: return []
    b1 = int(np.floor(n_layers/3))
    b2 = int(np.floor(2*n_layers/3))
    return ["S" if i < b1 else "M" if i < b2 else "D" for i in range(n_layers)]

def _nanmean(a) -> float:
    import numpy as _np
    try:
        return float(_np.nanmean(a))
    except Exception:
        return _np.nan


# =======================================================
# (A) CKA Heatmap (summary + detail)
# =======================================================

def compute_and_save_w1_cka_heatmap(
    ctx: Dict[str, Any],
    feature_dict: Dict[str, Any],
    out_dir: str | Path,
    depth_bins_cnn=None, depth_bins_gru=None, max_samples: Optional[int] = 20000
) -> None:
    """all_plotting_metrics의 compute_layerwise_cka를 실제 호출하여 저장까지 수행."""
    metrics = compute_layerwise_cka(
        feature_dict=feature_dict,
        cnn_key="cnn", gru_key="gru",
        max_samples=max_samples,
        depth_bins_cnn=depth_bins_cnn, depth_bins_gru=depth_bins_gru
    )
    save_w1_cka_heatmap(ctx, metrics, out_dir)

def save_w1_cka_heatmap(ctx: Dict[str, Any], metrics: Dict[str, Any], out_dir: str | Path) -> None:
    """이미 계산된 metrics(cka_matrix 등)를 받아 summary/detail을 저장."""
    out_dir = Path(out_dir)
    summary_csv = out_dir / "cka_heatmap_summary.csv"
    detail_csv  = out_dir / "cka_heatmap_detail.csv"

    if "cka_matrix" not in metrics:
        return

    M = np.array(metrics["cka_matrix"], dtype=float)
    Lc, Lg = M.shape if M.ndim == 2 else (0, 0)
    if Lc == 0 or Lg == 0:
        return

    cnn_bins = metrics.get("cnn_depth_bins") or _split_bins_auto(Lc)
    gru_bins = metrics.get("gru_depth_bins") or _split_bins_auto(Lg)

    # 요약치
    full_mean = _nanmean(M)
    diag_mean = _nanmean(np.diag(M)) if Lc == Lg else full_mean

    def _band_mean(band: str) -> float:
        idx_c = [i for i,b in enumerate(cnn_bins) if b == band]
        idx_g = [j for j,b in enumerate(gru_bins) if b == band]
        if not idx_c or not idx_g: return np.nan
        return _nanmean(M[np.ix_(idx_c, idx_g)])

    cka_s = _band_mean("S"); cka_m = _band_mean("M"); cka_d = _band_mean("D")
    nan_frac = float(np.isnan(M).sum()) / float(M.size)

    # summary 저장
    row = {
        **{k: ctx.get(k) for k in REQUIRED_KEYS if k in ctx},
        "depth_cnn": Lc, "depth_gru": Lg,
        "cka_s": cka_s, "cka_m": cka_m, "cka_d": cka_d,
        "cka_diag_mean": diag_mean, "cka_full_mean": full_mean,
        "n_pairs": int(M.size), "nan_frac": nan_frac
    }
    append_or_update_row(summary_csv, row, subset_keys=REQUIRED_KEYS)

    # detail 저장 (long-form, 상삼각만)
    rows = []
    for i in range(Lc):
        for j in range(i, Lg):
            v = M[i, j]
            if np.isfinite(v):
                rows.append({
                    **{k: ctx.get(k) for k in REQUIRED_KEYS if k in ctx},
                    "cnn_layer": i, "gru_layer": j,
                    "cnn_depth_bin": cnn_bins[i], "gru_depth_bin": gru_bins[j],
                    "cka_value": round(float(v), 6)
                })
    append_or_update_long(detail_csv, rows, subset_keys=REQUIRED_KEYS+["cnn_layer","gru_layer"])


# =======================================================
# (B) Grad-Align Bar (summary + detail)
# =======================================================

def compute_and_save_w1_grad_align_bar(
    ctx: Dict[str, Any],
    grad_dict: Dict[str, Any],
    out_dir: str | Path,
    depth_bins=None
) -> None:
    """all_plotting_metrics의 compute_gradient_alignment를 실제 호출하여 저장까지 수행."""
    metrics = compute_gradient_alignment(
        grad_dict=grad_dict,
        cnn_key="cnn", gru_key="gru",
        depth_bins=depth_bins
    )
    save_w1_grad_align_bar(ctx, metrics, out_dir)

def save_w1_grad_align_bar(ctx: Dict[str, Any], metrics: Dict[str, Any], out_dir: str | Path) -> None:
    """이미 계산된 metrics(grad_align_by_layer 등)를 받아 summary/detail을 저장."""
    out_dir = Path(out_dir)
    summary_csv = out_dir / "grad_align_bar_summary.csv"
    detail_csv  = out_dir / "grad_align_bar_detail.csv"

    detail = metrics.get("grad_align_by_layer", None)
    s_mean = metrics.get("grad_align_s", None)
    m_mean = metrics.get("grad_align_m", None)
    d_mean = metrics.get("grad_align_d", None)
    all_mean = metrics.get("grad_align_mean", None)

    # detail 저장
    rows = []
    if isinstance(detail, (list, tuple)) and len(detail) > 0:
        for item in detail:
            if isinstance(item, dict):
                li = int(item.get("layer_idx", -1))
                db = str(item.get("depth_bin", ""))
                gv = float(item.get("grad_align_value", np.nan))
            else:
                try:
                    li = int(item[0]); db = str(item[1]); gv = float(item[2])
                except Exception:
                    continue
            if not np.isfinite(gv): continue
            rows.append({
                **{k: ctx.get(k) for k in REQUIRED_KEYS if k in ctx},
                "layer_idx": li, "depth_bin": db,
                "grad_align_value": round(gv, 6)
            })
        append_or_update_long(detail_csv, rows, subset_keys=REQUIRED_KEYS+["layer_idx"])

        # summary 재계산(필요시)
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            if s_mean is None:
                s_mean = float(df.loc[df["depth_bin"]=="S","grad_align_value"].mean())
            if m_mean is None:
                m_mean = float(df.loc[df["depth_bin"]=="M","grad_align_value"].mean())
            if d_mean is None:
                d_mean = float(df.loc[df["depth_bin"]=="D","grad_align_value"].mean())
            if all_mean is None:
                all_mean = float(df["grad_align_value"].mean())
        except Exception:
            pass

    # summary 저장
    row = {
        **{k: ctx.get(k) for k in REQUIRED_KEYS if k in ctx},
        "grad_align_s": s_mean if s_mean is not None else np.nan,
        "grad_align_m": m_mean if m_mean is not None else np.nan,
        "grad_align_d": d_mean if d_mean is not None else np.nan,
        "grad_align_mean": all_mean if all_mean is not None else np.nan
    }
    append_or_update_row(summary_csv, row, subset_keys=REQUIRED_KEYS)


# =======================================================
# (C) Forest Plot (Δ summary + detail)
# =======================================================

def update_w1_forest_summary_and_detail(results_w1_csv: str | Path, out_dir: str | Path) -> None:
    """
    W1 전체 실행 로그(results_W1.csv)를 읽어,
    같은 (dataset,horizon,seed)의 per_layer vs last_layer를 조인해
    Δ지표(성능/CKA/GradAlign)와 Top-pick 점수를 산출 → forest_plot_summary.csv,
    비교 쌍(tidy, long-form) → forest_plot_detail.csv 갱신.
    """
    build_forest_from_results(
        experiment_type="W1",
        results_csv=results_w1_csv,
        out_dir=out_dir,
        per_mode="per_layer",
        last_mode="last_layer",
        rmse_col_pref=("rmse_real","rmse"),
        cka_col="cka_full_mean",
        grad_col="grad_align_mean"
    )


# =======================================================
# (D) 번들 호출 (한 번에 W1 3종 저장)
# =======================================================

def save_w1_figures_bundle(
    dataset: str, horizon: int, seed: int, mode: str,
    feature_dict: Optional[Dict[str, Any]] = None,      # CKA 계산용
    grad_dict: Optional[Dict[str, Any]] = None,         # Grad-Align 계산용
    results_w1_csv: Optional[str | Path] = None,
    results_out_root: str | Path = "results_W1"
) -> None:
    """
    한 번의 평가(run)에서 W1 그림 3종을 한꺼번에 저장.
    - feature_dict 존재 시: CKA(summary/detail) 생성
    - grad_dict    존재 시: Grad-Align(summary/detail) 생성
    - results_w1_csv 존재 시: Forest(summary/detail) 갱신
    """
    base_ctx = {
        "experiment_type": "W1",
        "dataset": dataset,
        "horizon": int(horizon),
        "seed": int(seed),
        "mode": mode
    }

    # 1) CKA
    if feature_dict is not None:
        compute_and_save_w1_cka_heatmap(
            ctx={**base_ctx, "plot_type": "cka_heatmap"},
            feature_dict=feature_dict,
            out_dir=Path(results_out_root)/dataset/"cka_heatmap"
        )

    # 2) Grad-Align
    if grad_dict is not None:
        compute_and_save_w1_grad_align_bar(
            ctx={**base_ctx, "plot_type": "grad_align_bar"},
            grad_dict=grad_dict,
            out_dir=Path(results_out_root)/dataset/"grad_align_bar"
        )

    # 3) Forest
    if results_w1_csv is not None:
        update_w1_forest_summary_and_detail(
            results_w1_csv=results_w1_csv,
            out_dir=Path(results_out_root)/dataset/"forest_plot"
        )
