# ============================================================
# utils/experiment_plotting_metrics/w2_plotting_metrics.py
#   - W2 전용: Gate-TOD Heatmap / Gate Distribution / Forest Plot
#   - summary + detail 분리 저장 (append + dedupe, atomic write)
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import time, math
import numpy as np
import pandas as pd

"""
위치 권장: utils/experiment_plotting_metrics/w2_plotting_metrics.py (W1과 동일한 패턴)

대상 그림: ① Gate‑TOD Heatmap, ② Gate Distribution, ③ Forest Plot(동적 vs 정적 비교 집계)

저장 정책: summary(핵심 수치) / detail(그림 재현용 long‑form) 분리

모든 행에 키 6종(experiment_type, dataset, plot_type, horizon, seed, mode) + model_tag, run_tag를 강제 포함

덮어쓰기(append+dedupe), 원자적 저장(.tmp 후 교체), float32(소수 6자리) 적용

Forest는 실행 단위 raw(동적/정적) → 사후 집계(Δ/CI/Top‑pick) 2단계 구조

입력 메트릭 기대치

Gate‑TOD Heatmap:

gate_tod_mean_s/m/d (각각 길이 T의 리스트; 대소문자 혼용도 허용)

(선택) tod_labels(길이 T)

Gate Distribution:

중앙/산포/왜도류: gate_mean, gate_std, gate_entropy, gate_kurtosis, gate_sparsity

분위수: gate_q05, q10, q25, q50, q75, q90, q95 (있으면 summary/ detail에 반영)

Forest Plot:

results_W2.csv(실행 단위 raw가 쌓이는 파일; mode∈{dynamic, static}, rmse_real/mae_real 등)
"""

"""
사용 예시

Gate‑TOD Heatmap 저장(평가 직후):

ctx = {
  "experiment_type": "W2",
  "dataset": dataset_tag,
  "plot_type": "gate_tod_heatmap",
  "horizon": int(CFG["horizon"]),
  "seed": int(CFG["seed"]),
  "mode": CFG["mode"],            # "dynamic" or "static"
  "model_tag": "HyperConv",
  "run_tag": f"W2_{dataset_tag}_{CFG['horizon']}_{CFG['seed']}_{CFG['mode']}"
}
metrics = {  # plotting 단계에서 얻은 결과
  "gate_tod_mean_s": list_s,   # 길이 T
  "gate_tod_mean_m": list_m,
  "gate_tod_mean_d": list_d,
  "tod_labels": list(range(len(list_s)))  # 선택
}
save_w2_gate_tod_heatmap(ctx, metrics, out_dir=f"results_W2/{dataset_tag}/gate_tod_heatmap", amp_method="range")

---

Gate Distribution 저장:

ctx["plot_type"] = "gate_distribution"
metrics = {
  "gate_mean": g_mean, "gate_std": g_std, "gate_entropy": g_ent,
  "gate_kurtosis": g_kur, "gate_sparsity": g_spa,
  "gate_q05": q05, "gate_q10": q10, "gate_q25": q25, "gate_q50": q50,
  "gate_q75": q75, "gate_q90": q90, "gate_q95": q95
}
save_w2_gate_distribution(ctx, metrics, out_dir=f"results_W2/{dataset_tag}/gate_distribution")

---

Forest Plot (raw 저장 → 사후 집계):

# 실행 단위 raw 저장 (dynamic / static 각각 1행)
ctx_dyn = {**ctx, "plot_type": "forest_plot", "mode": "dynamic"}
ctx_sta = {**ctx, "plot_type": "forest_plot", "mode": "static"}

save_w2_forest_unit_row(ctx_dyn, rmse_real=rmse_dyn, mae_real=mae_dyn,
                        unit_csv=f"results_W2/{dataset_tag}/forest_plot/W2_forest_unit.csv")
save_w2_forest_unit_row(ctx_sta, rmse_real=rmse_sta, mae_real=mae_sta,
                        unit_csv=f"results_W2/{dataset_tag}/forest_plot/W2_forest_unit.csv")

# (시드/조합이 어느 정도 쌓인 뒤) Δ/CI/Top-pick 집계
rebuild_w2_forest_agg(
    unit_csv=f"results_W2/{dataset_tag}/forest_plot/W2_forest_unit.csv",
    agg_csv =f"results_W2/{dataset_tag}/forest_plot/W2_forest_agg.csv",
    per_mode="dynamic", last_mode="static",
    group_keys=("experiment_type","dataset","horizon")
)

"""
# ---------------------------
# 공통 유틸 (W1과 동일 정책)
# ---------------------------
_W2_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag"]

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
#    - 시간대별 평균 게이트(스테이지 S/M/D)
#    - 요약: TOD 민감도(진폭/표준편차), 정상성 지표
#    - 상세: long-form (tod_bin × stage)
# ============================================================

def save_w2_gate_tod_heatmap(
    ctx: Dict[str,Any],                    # 키 6종 + model_tag + run_tag 포함, plot_type='gate_tod_heatmap'
    metrics: Dict[str,Any],                # 기대: gate_tod_mean_s/m/d(list), 선택적으로 tod_labels(list)
    out_dir: str | Path,                   # 예: results_W2/<dataset>/gate_tod_heatmap
    amp_method: str = "range",             # 'range'=(max-min), 'std' = 표준편차
) -> None:
    """
    summary:
      - tod_amp_s/m/d: 시간대 민감도(스테이지별 진폭 또는 표준편차)
      - tod_amp_mean: 3스테이지 평균
      - valid_ratio: 유효값 비율 / nan_count
      - T: TOD bin 개수
    detail:
      - 한 행 = (tod_bin, stage, gate_mean)
    """
    assert ctx.get("plot_type") == "gate_tod_heatmap", "plot_type must be 'gate_tod_heatmap'"

    # 1) 입력 시그니처 유연 처리 (대/소문자 키 혼용 허용)
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
        # 저장할 것이 없음
        return

    # 길이 보정 (None → 길이 0)
    s = np.array(s_list or [], dtype=float)
    m = np.array(m_list or [], dtype=float)
    d = np.array(d_list or [], dtype=float)

    # TOD 길이/라벨
    T = max(len(s), len(m), len(d))
    if tod_labels is None or len(tod_labels) != T:
        tod_labels = list(range(T))

    # 2) 민감도 정의
    def _amp(x: np.ndarray) -> float:
        if x.size == 0 or not np.isfinite(x).any(): return np.nan
        if amp_method == "std":
            return _to_f32(np.nanstd(x))
        # default: range
        return _to_f32(np.nanmax(x) - np.nanmin(x))

    amp_s = _amp(s)
    amp_m = _amp(m)
    amp_d = _amp(d)
    amp_mean = _to_f32(np.nanmean([v for v in [amp_s,amp_m,amp_d] if np.isfinite(v)]))

    # 품질
    all_vals = np.concatenate([v for v in [s,m,d] if v.size > 0], axis=0) if T > 0 else np.array([],dtype=float)
    valid = np.isfinite(all_vals).sum()
    total = all_vals.size
    valid_ratio = float(valid)/float(total) if total>0 else np.nan
    nan_count = int(total - valid)

    # 3) summary 저장(1행)
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

    # 4) detail 저장 (long)
    detail_rows: List[Dict[str,Any]] = []
    def _emit(stage: str, arr: np.ndarray):
        for t in range(len(arr)):
            v = arr[t]
            if not np.isfinite(v): 
                continue
            detail_rows.append({
                **{k: ctx[k] for k in _W2_KEYS if k in ctx},
                "tod": int(tod_labels[t]),
                "stage": stage,
                "gate_mean": _to_f32(v)
            })

    if s.size: _emit("S", s)
    if m.size: _emit("M", m)
    if d.size: _emit("D", d)

    detail_csv = Path(out_dir) / "gate_tod_heatmap_detail.csv"
    _append_or_update_rows(
        detail_csv, detail_rows,
        subset_keys=_W2_KEYS + ["tod","stage"]
    )


# ============================================================
# 2) Gate Distribution (summary + detail)
#    - 게이트 전역 분포(평균, 표준편차, 엔트로피, 첨도, 희소도, 분위수 등)
#    - detail: quantile long / (선택) 채널별 통계가 있으면 per-channel long
# ============================================================

def save_w2_gate_distribution(
    ctx: Dict[str,Any],                    # plot_type='gate_distribution'
    metrics: Dict[str,Any],                # 기대: gate_mean/std/entropy/kurtosis/sparsity, gate_qXX...
    out_dir: str | Path,                   # 예: results_W2/<dataset>/gate_distribution
    quantile_keys: Optional[List[str]] = None  # 예: ["gate_q05","gate_q10","gate_q25","gate_q50","gate_q75","gate_q90","gate_q95"]
) -> None:
    """
    summary:
      - gate_mean/std/entropy/kurtosis/sparsity + 유효 비율
    detail:
      - quantile long (key, value)
      - (선택) 채널별 분포 통계가 제공된다면 per-channel long도 허용
    """
    assert ctx.get("plot_type") == "gate_distribution", "plot_type must be 'gate_distribution'"

    # 1) 요약 스칼라 수집
    g_mean = _to_f32(metrics.get("gate_mean"))
    g_std  = _to_f32(metrics.get("gate_std"))
    g_ent  = _to_f32(metrics.get("gate_entropy"))
    g_kur  = _to_f32(metrics.get("gate_kurtosis"))
    g_spa  = _to_f32(metrics.get("gate_sparsity"))

    # 유효성(가능하면 raw 벡터로부터 계산, 없으면 스칼라 정보만)
    valid_ratio = _to_f32(metrics.get("gate_valid_ratio"))
    nan_count   = int(metrics.get("gate_nan_count", 0))

    summary_row = {
        **{k: ctx[k] for k in _W2_KEYS if k in ctx},
        "gate_mean": g_mean, "gate_std": g_std,
        "gate_entropy": g_ent, "gate_kurtosis": g_kur, "gate_sparsity": g_spa,
        "valid_ratio": valid_ratio, "nan_count": nan_count
    }
    summary_csv = Path(out_dir) / "gate_distribution_summary.csv"
    _append_or_update_row(summary_csv, summary_row, subset_keys=_W2_KEYS)

    # 2) detail: quantiles
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
    # (선택) 채널별 분포 통계가 제공되는 경우
    # ex) metrics["gate_channel_stats"] = [{"ch":0,"mean":..,"std":..,"entropy":..}, ...]
    ch_rows: List[Dict[str,Any]] = []
    ch_stats = metrics.get("gate_channel_stats")
    if isinstance(ch_stats, list) and len(ch_stats) > 0:
        for item in ch_stats:
            try:
                ch = int(item.get("ch"))
            except Exception:
                continue
            row = {
                **{k: ctx[k] for k in _W2_KEYS if k in ctx},
                "channel": ch,
                "mean": _to_f32(item.get("mean")),
                "std":  _to_f32(item.get("std")),
                "entropy": _to_f32(item.get("entropy")),
                "kurtosis": _to_f32(item.get("kurtosis")),
                "sparsity": _to_f32(item.get("sparsity")),
            }
            ch_rows.append(row)

    # 저장
    if q_rows:
        q_csv = Path(out_dir) / "gate_distribution_quantiles_detail.csv"
        _append_or_update_rows(q_csv, q_rows, subset_keys=_W2_KEYS + ["stat"])
    if ch_rows:
        ch_csv = Path(out_dir) / "gate_distribution_per_channel_detail.csv"
        _append_or_update_rows(ch_csv, ch_rows, subset_keys=_W2_KEYS + ["channel"])


# ============================================================
# 3) Forest Plot (동적 vs 정적 비교, Δ/CI/Top-pick)
#    - unit(raw) 누적 → agg(집계) 재작성
# ============================================================

def save_w2_forest_unit_row(
    ctx: Dict[str,Any],             # plot_type='forest_plot', mode ∈ {'dynamic','static'}
    rmse_real: float,
    mae_real: float,
    unit_csv: str | Path            # 예: results_W2/forest_plot/W2_forest_unit.csv
) -> None:
    row = {
        **{k: ctx[k] for k in _W2_KEYS if k in ctx},
        "rmse_real": _to_f32(rmse_real),
        "mae_real":  _to_f32(mae_real),
    }
    _append_or_update_row(unit_csv, row, subset_keys=_W2_KEYS)


def rebuild_w2_forest_agg(
    unit_csv: str | Path,           # 입력 raw
    agg_csv: str | Path,            # 출력 집계
    per_mode: str = "dynamic",      # 비교쌍: dynamic vs static
    last_mode: str = "static",
    group_keys: Tuple[str,...] = ("experiment_type","dataset","horizon")
) -> pd.DataFrame:
    """
    unit.csv에서 (dataset,horizon,seed) 단위로 dynamic/static를 매칭하여 Δ/CI/Top-pick을 집계.
    Δ 정의:
      - ΔRMSE = rmse_static - rmse_dynamic (양수일수록 dynamic이 우수)
      - ΔMAE  = mae_static  - mae_dynamic
      - ΔTOD  = tod_amp_mean_dynamic - tod_amp_mean_static  (있을 때만)
      - ΔVarT = gate_variability_time_dynamic - gate_variability_time_static (있을 때만)
    Top-pick(무가중치): z(ΔRMSE_pct) + z(ΔTOD) + z(ΔVarT)
    """
    unit_csv = Path(unit_csv)
    if not unit_csv.exists():
        raise FileNotFoundError(f"unit csv not found: {unit_csv}")
    df = pd.read_csv(unit_csv)

    need = set(group_keys) | {"seed","mode","rmse_real","mae_real"}
    miss = [c for c in need if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in unit csv: {miss}")

    per = df[df["mode"]==per_mode].copy()
    lst = df[df["mode"]==last_mode].copy()
    on = list(group_keys) + ["seed"]
    base = per.merge(lst, on=on, suffixes=("_per","_last"))
    if len(base)==0:
        out = pd.DataFrame(columns=list(group_keys)+[
            "n_pairs","delta_rmse","delta_rmse_pct","delta_mae","delta_tod","delta_var_time","top_pick_score_w2"
        ])
        _atomic_write_csv(out, agg_csv)
        return out

    # Δ 계산
    base["delta_rmse"] = (base["rmse_real_last"] - base["rmse_real_per"]).astype(np.float32)
    base["delta_rmse_pct"] = (base["delta_rmse"] / base["rmse_real_last"]).astype(np.float32) * 100.0
    base["delta_mae"]  = (base["mae_real_last"]  - base["mae_real_per"]).astype(np.float32)

    # 옵션: TOD 민감도/게이트 변동성 (있으면 사용)
    def _get(name):
        return name if name in base.columns else None

    # W2 Gate‑TOD summary에서 저장한 컬럼명 가정: 'tod_amp_mean'
    t_per  = _get("tod_amp_mean_per")
    t_last = _get("tod_amp_mean_last")
    if t_per and t_last:
        base["delta_tod"] = (base[t_per] - base[t_last]).astype(np.float32)  # dynamic - static (양수면 동적이 더 민감)
    else:
        base["delta_tod"] = np.nan

    # W2 Gate Distribution에서 저장한 변동성 시간축 지표(있을 경우)
    v_per  = _get("w2_gate_variability_time_per") or _get("gate_var_time_per")
    v_last = _get("w2_gate_variability_time_last") or _get("gate_var_time_last")
    if v_per and v_last:
        base["delta_var_time"] = (base[v_per] - base[v_last]).astype(np.float32)
    else:
        base["delta_var_time"] = np.nan

    # 그룹별 Top-pick 산출
    def _top_score(gr: pd.DataFrame) -> pd.DataFrame:
        z_rmse = _zscore(gr["delta_rmse_pct"].fillna(0.0))
        z_tod  = _zscore(gr["delta_tod"].fillna(0.0))
        z_var  = _zscore(gr["delta_var_time"].fillna(0.0))
        gr["top_pick_score_w2"] = z_rmse + z_tod + z_var
        return gr

    base = base.groupby(list(group_keys), group_keys=False).apply(_top_score)

    # 집계(평균/CI)
    def _ci_95(a: np.ndarray) -> Tuple[float,float,float]:
        a = a[np.isfinite(a)]
        if a.size == 0:
            return (np.nan, np.nan, np.nan)
        m = float(np.mean(a))
        s = float(np.std(a, ddof=1)) if a.size > 1 else 0.0
        se = s / math.sqrt(max(1, a.size))
        return (_to_f32(m), _to_f32(m-1.96*se), _to_f32(m+1.96*se))

    rows: List[Dict[str,Any]] = []
    for gvals, gdf in base.groupby(list(group_keys)):
        d_rmse, _, _ = _ci_95(gdf["delta_rmse"].values.astype(np.float64))
        d_rmsp,_,_  = _ci_95(gdf["delta_rmse_pct"].values.astype(np.float64))
        d_mae, _, _ = _ci_95(gdf["delta_mae"].values.astype(np.float64))
        d_tod, _, _ = _ci_95(gdf["delta_tod"].values.astype(np.float64))
        d_var, _, _ = _ci_95(gdf["delta_var_time"].values.astype(np.float64))
        row = {
            **{k:v for k,v in zip(group_keys,gvals)},
            "n_pairs": int(len(gdf)),
            "delta_rmse_mean": d_rmse,
            "delta_rmse_pct_mean": d_rmsp,
            "delta_mae_mean": d_mae,
            "delta_tod_mean": d_tod,
            "delta_var_time_mean": d_var,
            # 참고: 개별 run의 top_pick_score_w2는 base에 있음(필요 시 별도 저장)
        }
        rows.append(row)

    out = pd.DataFrame(rows)
    _atomic_write_csv(out, agg_csv)
    return out
