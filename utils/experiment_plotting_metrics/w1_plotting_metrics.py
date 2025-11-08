# ===========================================
# W1 전용 3종 패치 함수 (plot_results.py에서 import 해서 사용)
# ===========================================
from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import json
import os
from typing import Dict, Any, List

# ---------------------------
# 공통 유틸
# ---------------------------

"""
위치 권장: utils/plot_results.py (또는 별도 모듈로 추가 후 plot_results.py에서 import)

목적: (1) CKA Heatmap(요약+상세), (2) Grad‑Align Bar(요약+상세), (3) Forest Plot(Δ 비교+Top‑pick 점수) 저장을 스키마에 맞춰 기록

키(필수): experiment_type, dataset, plot_type, horizon, seed, mode

모든 함수는 append‑only + 중복 제거(키 기준) 방식으로 덮어쓰기 저장합니다.

주의: CKA/Grad‑Align 요약 계산은 plotting_metrics에서 제공하는 메트릭에 의존합니다.
‑ CKA: cka_matrix(np.ndarray or list of lists), (선택) cnn_depth_bins, gru_depth_bins
‑ Grad‑Align: grad_align_by_layer(list of {layer_idx, depth_bin, grad_align_value}) 또는 요약치
"""

"""
사용 예시

CKA Heatmap 저장 (W1 실행 1회의 평가 후)

# ctx 예시
ctx = {
  "experiment_type": "W1",
  "dataset": dataset_name,
  "plot_type": "cka_heatmap",
  "horizon": int(CFG["horizon"]),
  "seed": int(CFG["seed"]),
  "mode": CFG["mode"]  # 'per_layer' or 'last_layer'
}
# metrics는 plotting_metrics.compute_layerwise_cka() 결과 dict
save_w1_cka_heatmap(ctx, metrics, out_dir=f"results_W1/{dataset_name}/cka_heatmap")

---

Grad-Align Bar 저장

ctx["plot_type"] = "grad_align_bar"
# metrics는 plotting_metrics.compute_gradient_alignment() 결과 dict
save_w1_grad_align_bar(ctx, metrics, out_dir=f"results_W1/{dataset_name}/grad_align_bar")

---

Forest Plot 요약(Δ + Top‑pick) 업데이트
(W1 모든 실행이 results_W1.csv에 쌓이는 구조라면, 각 실행 저장 직후 또는 데이터셋/호라이즌 단위로 주기적 호출)

update_w1_forest_summary_from_results(
    results_w1_csv="results_W1.csv",
    out_csv=f"results_W1/{dataset_name}/forest_plot/forest_plot_summary.csv"
)

"""


_W1_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode"]

def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def _append_or_update_row(path: Path, row: Dict[str, Any], subset_keys: List[str]) -> None:
    _ensure_dir(path)
    row_df = pd.DataFrame([row])
    if path.exists():
        df = pd.read_csv(path)
        # drop old rows with same subset
        mask = np.ones(len(df), dtype=bool)
        for k in subset_keys:
            mask &= (df[k] == row[k])
        df = pd.concat([df[~mask], row_df], ignore_index=True)
    else:
        df = row_df
    _atomic_write_csv(df, path)

def _append_or_update_long(path: Path, rows: List[Dict[str, Any]], subset_keys: List[str]) -> None:
    if not rows:
        return
    _ensure_dir(path)
    new_df = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        df = pd.concat([old, new_df], ignore_index=True)
        df.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
    else:
        df = new_df
    _atomic_write_csv(df, path)

def _split_bins_auto(n_layers: int) -> List[str]:
    """레이어 수만 알고 S/M/D 구간을 균등 분할 라벨로 생성."""
    if n_layers <= 0:
        return []
    # 3등분 라벨
    borders = [int(np.floor(n_layers/3)), int(np.floor(2*n_layers/3))]
    bins = []
    for i in range(n_layers):
        if i < borders[0]:
            bins.append("S")
        elif i < borders[1]:
            bins.append("M")
        else:
            bins.append("D")
    return bins

def _safe_float(x, default=np.nan):
    try:
        v = float(x)
        if np.isfinite(v):
            return v
        return default
    except Exception:
        return default

def _nanmean(a):
    try:
        return float(np.nanmean(a))
    except Exception:
        return np.nan

def _zscore(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    m = s.mean()
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd

# =======================================================
# 1) W1 - CKA Heatmap (summary + detail)
# =======================================================

def save_w1_cka_heatmap(ctx: Dict[str, Any], metrics: Dict[str, Any], out_dir: str | Path) -> None:
    """
    ctx: {experiment_type, dataset, plot_type='cka_heatmap', horizon, seed, mode, ...}
    metrics: plotting_metrics.compute_layerwise_cka() 반환 내용을 포함해야 함.
      - 필수: 'cka_matrix' (2D array-like)
      - 선택: 'cnn_depth_bins', 'gru_depth_bins' (없으면 자동 균등분할)
    out_dir: results_W1/<dataset>/cka_heatmap 디렉토리
    """
    out_dir = Path(out_dir)
    summary_csv = out_dir / "cka_heatmap_summary.csv"
    detail_csv  = out_dir / "cka_heatmap_detail.csv"

    # 1) 행렬 확보
    if "cka_matrix" not in metrics:
        # 행렬이 없으면 summary 스킵(빈 행 저장 X) — detail도 없음
        return
    M = np.array(metrics["cka_matrix"], dtype=float)
    Lc, Lg = M.shape if M.ndim == 2 else (0, 0)

    # 2) 레이어 라벨
    cnn_layers = list(range(Lc))
    gru_layers = list(range(Lg))
    cnn_bins = metrics.get("cnn_depth_bins") or _split_bins_auto(Lc)
    gru_bins = metrics.get("gru_depth_bins") or _split_bins_auto(Lg)

    # 3) 요약치 계산
    full_mean = _nanmean(M)
    diag_mean = _nanmean(np.diag(M)) if Lc == Lg and Lc > 0 else full_mean
    # 밴드별(S/M/D) 평균: 동일 밴드 조합만 취득(S-S, M-M, D-D)
    def _band_mean(band: str) -> float:
        idx_c = [i for i,b in enumerate(cnn_bins) if b == band]
        idx_g = [j for j,b in enumerate(gru_bins) if b == band]
        if not idx_c or not idx_g:
            return np.nan
        sub = M[np.ix_(idx_c, idx_g)]
        return _nanmean(sub)
    cka_s = _band_mean("S")
    cka_m = _band_mean("M")
    cka_d = _band_mean("D")
    nan_frac = float(np.isnan(M).sum()) / float(M.size) if M.size > 0 else 0.0

    # 4) summary 저장(한 행)
    row = {
        **{k: ctx[k] for k in _W1_KEYS if k in ctx},
        "depth_cnn": Lc, "depth_gru": Lg,
        "cka_s": cka_s, "cka_m": cka_m, "cka_d": cka_d,
        "cka_diag_mean": diag_mean,
        "cka_full_mean": full_mean,
        "n_pairs": int(M.size),
        "nan_frac": nan_frac
    }
    _append_or_update_row(summary_csv, row, subset_keys=_W1_KEYS)

    # 5) detail 저장(long): 상삼각만 저장(중복 절감)
    rows = []
    for i in range(Lc):
        for j in range(i, Lg):
            v = M[i, j]
            if not np.isfinite(v):
                continue
            rows.append({
                **{k: ctx[k] for k in _W1_KEYS if k in ctx},
                "cnn_layer": i, "gru_layer": j,
                "cnn_depth_bin": cnn_bins[i] if i < len(cnn_bins) else "",
                "gru_depth_bin": gru_bins[j] if j < len(gru_bins) else "",
                "cka_value": round(float(v), 6)
            })
    _append_or_update_long(
        detail_csv, rows,
        subset_keys=_W1_KEYS + ["cnn_layer","gru_layer"]
    )

# =======================================================
# 2) W1 - Grad-Align Bar (summary + detail)
# =======================================================

def save_w1_grad_align_bar(ctx: Dict[str, Any], metrics: Dict[str, Any], out_dir: str | Path) -> None:
    """
    ctx: {experiment_type, dataset, plot_type='grad_align_bar', horizon, seed, mode, ...}
    metrics: plotting_metrics.compute_gradient_alignment() 결과(권장)
      - 권장: 'grad_align_by_layer' = [{layer_idx, depth_bin, grad_align_value}, ...]
      - 선택: 'grad_align_s','grad_align_m','grad_align_d','grad_align_mean'
    out_dir: results_W1/<dataset>/grad_align_bar
    """
    out_dir = Path(out_dir)
    summary_csv = out_dir / "grad_align_bar_summary.csv"
    detail_csv  = out_dir / "grad_align_bar_detail.csv"

    # 1) detail 우선 (있으면 summary도 계산 가능)
    detail = metrics.get("grad_align_by_layer", None)
    s_mean = metrics.get("grad_align_s", None)
    m_mean = metrics.get("grad_align_m", None)
    d_mean = metrics.get("grad_align_d", None)
    all_mean = metrics.get("grad_align_mean", None)

    # detail이 있으면 long 저장
    if isinstance(detail, (list, tuple)) and len(detail) > 0:
        rows = []
        for item in detail:
            if isinstance(item, dict):
                li = int(item.get("layer_idx", -1))
                db = str(item.get("depth_bin", ""))
                gv = _safe_float(item.get("grad_align_value", np.nan))
            else:
                # (layer_idx, depth_bin, grad) 형태도 허용
                try:
                    li = int(item[0]); db = str(item[1]); gv = _safe_float(item[2])
                except Exception:
                    continue
            if not np.isfinite(gv):
                continue
            rows.append({
                **{k: ctx[k] for k in _W1_KEYS if k in ctx},
                "layer_idx": li, "depth_bin": db,
                "grad_align_value": round(float(gv), 6)
            })
        _append_or_update_long(
            detail_csv, rows,
            subset_keys=_W1_KEYS + ["layer_idx"]
        )
        # detail로부터 요약 재계산(필요 시)
        try:
            df = pd.DataFrame(rows)
            s_val = df.loc[df["depth_bin"]=="S","grad_align_value"].mean()
            m_val = df.loc[df["depth_bin"]=="M","grad_align_value"].mean()
            d_val = df.loc[df["depth_bin"]=="D","grad_align_value"].mean()
            all_val = df["grad_align_value"].mean()
            if s_mean is None: s_mean = _safe_float(s_val)
            if m_mean is None: m_mean = _safe_float(m_val)
            if d_mean is None: d_mean = _safe_float(d_val)
            if all_mean is None: all_mean = _safe_float(all_val)
        except Exception:
            pass

    # 2) summary 저장(한 행)
    row = {
        **{k: ctx[k] for k in _W1_KEYS if k in ctx},
        "grad_align_s": _safe_float(s_mean),
        "grad_align_m": _safe_float(m_mean),
        "grad_align_d": _safe_float(d_mean),
        "grad_align_mean": _safe_float(all_mean)
    }
    _append_or_update_row(summary_csv, row, subset_keys=_W1_KEYS)

# =======================================================
# 3) W1 - Forest Plot (Δ 비교 + Top-pick 점수)
# =======================================================

def update_w1_forest_summary_from_results(
    results_w1_csv: str | Path,
    out_csv: str | Path
) -> None:
    """
    results_W1.csv를 읽어, 같은 (dataset,horizon,seed)의 per_layer vs last_layer를 조인해
    Δ 지표(성능/CKA/GradAlign)와 Top-pick 점수를 산출 → forest_plot_summary.csv 갱신.

    * 전제: results_W1.csv 안에 다음 컬럼이 존재하는 것이 이상적.
      - rmse_real (or rmse), mae_real (선택)
      - cka_full_mean (save_w1_cka_heatmap가 summary에 저장)
      - grad_align_mean (save_w1_grad_align_bar가 summary에 저장)

    동작:
      1) results_W1.csv 로드
      2) mode=='per_layer' vs mode=='last_layer' 조인
      3) Δ 계산: (last - per) 또는 (per - last)는 아래 일관 규칙 적용
         - ΔRMSE(개선) = (rmse_last - rmse_per)  => 값이 클수록 per가 좋음
         - ΔCKA        = (cka_full_mean_per - cka_full_mean_last)
         - ΔGradAlign  = (grad_align_mean_per - grad_align_mean_last)
         - ΔRMSE_pct   = (rmse_last - rmse_per) / rmse_last * 100
      4) (dataset,horizon) 그룹 내 z-score 정규화 후 TopPick_W1 = z(ΔRMSE_pct)+z(ΔCKA)+z(ΔGradAlign)
      5) out_csv에 저장(전체 재작성; 동일 키 덮어쓰기)
    """
    results_w1_csv = Path(results_w1_csv)
    out_csv = Path(out_csv)
    if not results_w1_csv.exists():
        return
    df = pd.read_csv(results_w1_csv)

    # 필수 하위셋
    req_cols = ["dataset","horizon","seed","mode","rmse_real"]
    for c in req_cols:
        if c not in df.columns:
            # 최소한 rmse_real은 필요
            return

    # per/last 분리 후 조인
    per = df[df["mode"]=="per_layer"].copy()
    last = df[df["mode"]=="last_layer"].copy()
    if len(per)==0 or len(last)==0:
        # 아직 한쪽이 없으면 skip
        return

    join_keys = ["dataset","horizon","seed"]
    base = per.merge(last, on=join_keys, suffixes=("_per","_last"))

    # Δ 계산
    def _col_or_nan(name):
        return name if name in base.columns else None

    base["delta_rmse"] = base["rmse_real_last"] - base["rmse_real_per"]
    base["delta_rmse_pct"] = 100.0 * base["delta_rmse"] / base["rmse_real_last"]

    cka_per  = _col_or_nan("cka_full_mean_per")
    cka_last = _col_or_nan("cka_full_mean_last")
    if cka_per and cka_last:
        base["delta_cka_global"] = base[cka_per] - base[cka_last]
    else:
        base["delta_cka_global"] = np.nan

    ga_per  = _col_or_nan("grad_align_mean_per")
    ga_last = _col_or_nan("grad_align_mean_last")
    if ga_per and ga_last:
        base["delta_grad_align_all"] = base[ga_per] - base[ga_last]
    else:
        base["delta_grad_align_all"] = np.nan

    # Top-pick 점수 (무가중치 z-합; 그룹=dataset×horizon)
    def _compute_top_score(grp: pd.DataFrame) -> pd.DataFrame:
        z_rmse = _zscore(grp["delta_rmse_pct"])
        z_cka  = _zscore(grp["delta_cka_global"]) if "delta_cka_global" in grp else pd.Series(0, index=grp.index)
        z_gal  = _zscore(grp["delta_grad_align_all"]) if "delta_grad_align_all" in grp else pd.Series(0, index=grp.index)
        score = z_rmse
        if z_cka is not None: score = score + z_cka
        if z_gal is not None: score = score + z_gal
        grp["top_pick_score_w1"] = score
        return grp

    base = base.groupby(["dataset","horizon"], group_keys=False).apply(_compute_top_score)

    # forest summary 스키마 정리
    forest_cols = [
        "dataset","horizon","seed",
        "rmse_real_per","rmse_real_last",
        "delta_rmse","delta_rmse_pct",
        "delta_cka_global","delta_grad_align_all",
        "top_pick_score_w1"
    ]
    # 존재하지 않을 수도 있는 컬럼은 제거
    forest_cols = [c for c in forest_cols if c in base.columns]
    forest = base[forest_cols].copy()
    forest.insert(0, "experiment_type", "W1")
    forest.insert(1, "plot_type", "forest_plot")

    # 저장(전체 재작성; 덮어쓰기)
    _atomic_write_csv(forest, out_csv)
