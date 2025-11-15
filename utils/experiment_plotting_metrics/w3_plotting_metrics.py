# ============================================================
# utils/experiment_plotting_metrics/w3_plotting_metrics.py
#   - W3: Gate‑TOD Heatmap / Gate Distribution / Forest / Perturb Δ
#   - summary + detail, append+dedupe, atomic write, pandas warning fix
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

_W3_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag"]

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p

def _atomic_write_csv(path: str | Path, df: pd.DataFrame) -> None:
    """Atomic CSV write (single temp file, then replace)."""
    path = _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)

def _append_or_update_row(csv_path: str | Path, row: Dict[str,Any], subset_keys: List[str]) -> None:
    """Append (or overwrite) a single row based on subset_keys."""
    path = _ensure_dir(csv_path)
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        # column union (keeps backward/forward compat)
        for c in new.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in new.columns:
                new[c] = np.nan
        # overwrite-by-key (drop duplicates on key set, keep last)
        mask = np.ones(len(old), dtype=bool)
        for k in subset_keys:
            mask &= (old.get(k) == row.get(k))
        out = pd.concat([old[~mask], new], ignore_index=True)
    else:
        out = new
    _atomic_write_csv(path, out)

def _append_or_update_rows(csv_path: str | Path, rows: List[Dict[str,Any]], subset_keys: List[str]) -> None:
    """Append (or overwrite) many rows based on subset_keys."""
    if not rows:
        return
    path = _ensure_dir(csv_path)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        # column union
        for c in new.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in new.columns:
                new[c] = np.nan
        out = pd.concat([old, new], ignore_index=True)
        out.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
    else:
        out = new
    _atomic_write_csv(path, out)

def _infer_tod_bins(dataset: str) -> int:
    ds = (dataset or "").lower()
    if "ettm" in ds: return 96
    if "etth" in ds: return 24
    if "weather" in ds: return 144
    return 24

def _tod_index_from_vec(tod_vec: np.ndarray, T: int) -> np.ndarray:
    """
    tod_vec: shape (N,2) with [sin, cos]
    return: (N,) integer bin in [0, T-1]
    """
    if not isinstance(tod_vec, np.ndarray) or tod_vec.ndim != 2 or tod_vec.shape[1] < 2:
        return np.array([], dtype=int)
    s = np.asarray(tod_vec[:, 0], float)
    c = np.asarray(tod_vec[:, 1], float)
    ang = np.arctan2(s, c)                # [-pi, pi]
    ang = (ang + 2*np.pi) % (2*np.pi)     # [0, 2pi)
    idx = np.floor(ang / (2*np.pi) * T).astype(int)
    return np.clip(idx, 0, T-1)

def _stage_split_mean_per_sample(gates_2d: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Fallback: split features equally into 3 stages along feature axis and average per sample.
    gates_2d: (N, F_total)
    return {'S': (N,), 'M': (N,), 'D': (N,)}
    """
    x = np.asarray(gates_2d, float)
    if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
        return {}
    N, F = x.shape
    s = F // 3
    if s == 0:
        # not enough features; use whole as 'D'
        return {"D": x.mean(axis=1)}
    aS = x[:, :s].mean(axis=1)
    aM = x[:, s:2*s].mean(axis=1) if 2*s <= F else np.array([], float)
    aD = x[:, 2*s:].mean(axis=1)  if 2*s  < F else np.array([], float)
    out: Dict[str,np.ndarray] = {"S": aS}
    if aM.size: out["M"] = aM
    if aD.size: out["D"] = aD
    return out

def _collect_gate_arrays_from_hooks(hooks_data: Dict[str,Any]) -> Optional[np.ndarray]:
    """
    Try common keys to gather gate outputs from hooks. Return (N, F) as float if possible.
    """
    if not isinstance(hooks_data, dict):
        return None
    cand_keys = ("gate_outputs","gates","gates_all","gate_logs")
    buf: List[np.ndarray] = []
    for k in cand_keys:
        v = hooks_data.get(k, None)
        if v is None:
            continue
        if isinstance(v, np.ndarray):
            buf.append(np.asarray(v, float))
        elif isinstance(v, list):
            for a in v:
                if isinstance(a, np.ndarray):
                    buf.append(np.asarray(a, float))
    if not buf:
        return None
    # reshape each to (N, -1), then concat along features; fall back to first on error
    try:
        parts = [a.reshape(a.shape[0], -1) for a in buf]
        # align N to min length
        n = min(p.shape[0] for p in parts)
        parts = [p[:n] for p in parts]
        out = np.concatenate(parts, axis=1)  # (N, F_total)
    except Exception:
        a0 = buf[0]
        n = a0.shape[0]
        out = a0.reshape(n, -1)
    return out

# ============================================================
# 1) Gate‑TOD heatmap
#    - 샘플별 gate 강도 ↔ 해당 샘플 TOD bin 으로 집계 (T=dataset별)
# ============================================================
def compute_and_save_w3_gate_tod_heatmap(
    ctx: Dict[str, Any],                 # must include _W3_KEYS
    hooks_data: Dict[str, Any],          # gate hooks or stage dict
    tod_vec: np.ndarray,                 # (N,2) sin/cos TOD vector
    dataset: str,                        # e.g., 'ETTm2', 'ETTh2'
    out_dir: str | Path,                 # results/.../gate_tod_heatmap
    amp_method: str = "range",           # 'range' | 'std'
) -> None:
    """
    Fixes signature mismatch (critical) and performs robust TOD-binned gate aggregation.
    - Summary: gate_tod_heatmap_summary.csv (tod_bins, tod_amp_s/m/d, tod_amp_mean, valid_ratio, nan_count)
    - Detail : gate_tod_heatmap_detail.csv  (one row per (stage, tod), with gate_mean and sample_count)
    """
    assert ctx.get("plot_type") == "gate_tod_heatmap", "plot_type must be 'gate_tod_heatmap'"

    # --- infer TOD bins & bin index from sin/cos vector
    T = _infer_tod_bins(dataset)
    idx = _tod_index_from_vec(np.asarray(tod_vec, float) if tod_vec is not None else np.array([], float), T)

    # --- build stage-wise per-sample mean gates
    # 1) if a stage dict exists in hooks, use it
    st: Dict[str, np.ndarray] = {}
    if isinstance(hooks_data, dict):
        for k in ("gates_by_stage","gate_by_stage","gate_logs_by_stage"):
            sd = hooks_data.get(k, None)
            if isinstance(sd, dict) and sd:
                for stage in ("S","M","D"):
                    a = sd.get(stage) or sd.get(stage.lower())
                    if a is None:
                        continue
                    aa = np.asarray(a, float)
                    aa = aa.reshape(aa.shape[0], -1).mean(axis=1)  # (N,*) → (N,)
                    st[stage] = aa
                break

    # 2) otherwise: collect generic gates → split into S/M/D
    if not st:
        gates2d = _collect_gate_arrays_from_hooks(hooks_data)
        if gates2d is None:
            # nothing to save → write a minimal summary with tod_bins=0
            _append_or_update_row(
                Path(out_dir) / "gate_tod_heatmap_summary.csv",
                {
                    **{k: ctx[k] for k in _W3_KEYS if k in ctx},
                    "tod_bins": 0, "tod_amp_s": np.nan, "tod_amp_m": np.nan, "tod_amp_d": np.nan,
                    "tod_amp_mean": np.nan, "valid_ratio": 0.0, "nan_count": 0
                },
                subset_keys=_W3_KEYS
            )
            return
        st = _stage_split_mean_per_sample(gates2d)  # {'S': (N,), 'M': (N,), 'D': (N,)}

    # idx guard
    if not isinstance(idx, np.ndarray) or idx.size == 0:
        _append_or_update_row(
            Path(out_dir) / "gate_tod_heatmap_summary.csv",
            {
                **{k: ctx[k] for k in _W3_KEYS if k in ctx},
                "tod_bins": 0, "tod_amp_s": np.nan, "tod_amp_m": np.nan, "tod_amp_d": np.nan,
                "tod_amp_mean": np.nan, "valid_ratio": 0.0, "nan_count": 0
            },
            subset_keys=_W3_KEYS
        )
        return

    # --- stage-wise TOD aggregation
    rows: List[Dict[str,Any]] = []
    amp: Dict[str, float] = {"S": np.nan, "M": np.nan, "D": np.nan}
    covered = np.zeros(T, dtype=bool)

    def _amp_fn(x: np.ndarray) -> float:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        if amp_method == "std":
            return float(np.nanstd(x))
        return float(np.nanmax(x) - np.nanmin(x))

    for stage in ("S","M","D"):
        v = np.asarray(st.get(stage), float)
        if v.size == 0:
            continue
        N = min(v.shape[0], idx.shape[0])
        if N <= 0:
            continue
        v = v[:N]; b = idx[:N]
        msk = np.isfinite(v)
        bins: List[List[float]] = [[] for _ in range(T)]
        for i in range(N):
            if not msk[i]:
                continue
            j = int(b[i])
            bins[j].append(float(v[i]))

        means = [float(np.mean(bucket)) if len(bucket) > 0 else np.nan for bucket in bins]
        counts = [int(len(bucket)) for bucket in bins]

        for t in range(T):
            if counts[t] > 0:
                covered[t] = True
            mv = means[t]
            rows.append({
                **{k: ctx[k] for k in _W3_KEYS if k in ctx},
                "tod": int(t),
                "stage": stage,
                "gate_mean": float(mv) if np.isfinite(mv) else np.nan,
                "sample_count": counts[t],
            })

        finite_means = [m for m in means if np.isfinite(m)]
        amp[stage] = _amp_fn(np.array(finite_means, float)) if finite_means else np.nan

    # --- write detail (append+dedupe)
    det_csv = Path(out_dir) / "gate_tod_heatmap_detail.csv"
    _append_or_update_rows(
        det_csv,
        rows,
        subset_keys=_W3_KEYS + ["tod","stage"]
    )

    # --- write summary
    valid_bins = int(covered.sum())
    total_bins = int(T)
    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "tod_bins": total_bins,
        "tod_amp_s": float(amp["S"]) if np.isfinite(amp["S"]) else np.nan,
        "tod_amp_m": float(amp["M"]) if np.isfinite(amp["M"]) else np.nan,
        "tod_amp_d": float(amp["D"]) if np.isfinite(amp["D"]) else np.nan,
        "tod_amp_mean": float(np.nanmean([amp["S"], amp["M"], amp["D"]])) if any(np.isfinite(list(amp.values()))) else np.nan,
        "valid_ratio": float(valid_bins / total_bins) if total_bins > 0 else 0.0,
        "nan_count": int((~covered).sum())
    }
    _append_or_update_row(
        Path(out_dir) / "gate_tod_heatmap_summary.csv",
        summary_row,
        subset_keys=_W3_KEYS
    )

# ============================================================
# 2) Gate distribution (게이트 전역 분포 요약 + 분위수 long)
# ============================================================
def compute_and_save_w3_gate_distribution(
    ctx: Dict[str,Any],                    # plot_type='gate_distribution'
    hooks_data: Dict[str,Any],
    out_dir: str | Path,
) -> None:
    assert ctx.get("plot_type") == "gate_distribution"
    gates = _collect_gate_arrays_from_hooks(hooks_data)
    if gates is None: return
    a = np.asarray(gates, float).reshape(-1)
    fin = a[np.isfinite(a)]
    if fin.size == 0: return

    def _entropy_pos(x: np.ndarray) -> float:
        x = x[x > 1e-8]
        if x.size == 0: return np.nan
        p = x / (x.sum() + 1e-12)
        return float(-np.sum(p*np.log(p + 1e-12)))

    row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "gate_mean": float(np.mean(fin)),
        "gate_std":  float(np.std(fin, ddof=0)),
        "gate_entropy": _entropy_pos(fin),
        "valid_ratio": 1.0, "nan_count": 0
    }
    _append_or_update_row(Path(out_dir)/"gate_distribution_summary.csv", row, subset_keys=_W3_KEYS)

    qs = [0.05,0.10,0.25,0.50,0.75,0.90,0.95]
    qv = np.quantile(fin, qs)
    q_rows = []
    for p, v in zip(qs, qv):
        q_rows.append({
            **{k: ctx[k] for k in _W3_KEYS if k in ctx},
            "stat": f"gate_q{int(p*100):02d}",
            "value": float(v)
        })
    _append_or_update_rows(Path(out_dir)/"gate_distribution_quantiles_detail.csv",
                           q_rows, subset_keys=_W3_KEYS+["stat"])


# ============================================================
# 3) Forest: raw(detail) → summary(Δ)
# ============================================================
def save_w3_forest_detail_row(
    ctx: Dict[str,Any],                    # plot_type='forest_plot'
    rmse_real: float, mae_real: float,
    detail_csv: str | Path,
) -> None:
    assert ctx.get("plot_type") == "forest_plot"
    row = {**{k: ctx[k] for k in _W3_KEYS if k in ctx}, "rmse_real": float(rmse_real), "mae_real": float(mae_real)}
    _append_or_update_row(detail_csv, row, subset_keys=_W3_KEYS)

def rebuild_w3_forest_summary(
    detail_csv: str | Path,
    summary_csv: str | Path,
    gate_tod_summary_csv: Optional[str | Path] = None,
    group_keys: Tuple[str, ...] = ("experiment_type","dataset","horizon"),
) -> None:
    dpath = Path(detail_csv)
    if not dpath.exists(): return
    df = pd.read_csv(dpath)
    if df.empty: return

    # none(베이스) ↔ {tod_shift, smooth} 매칭
    base = df[df["mode"]=="none"]
    pert = df[df["mode"]!="none"]

    # RMSE/MAE Δ
    merged = pert.merge(base,
                        on=list(group_keys)+["seed","model_tag"],
                        suffixes=("_p","_b"),
                        how="left")
    if merged.empty: return

    out_rows = []
    for keys, g in merged.groupby(list(group_keys), as_index=False):
        for perturbation in ["tod_shift","smooth"]:
            gi = g[g["mode_p"]==perturbation]
            if gi.empty: 
                continue
            delta_rmse = (gi["rmse_real_p"] - gi["rmse_real_b"]).astype(float)
            delta_mae  = (gi["mae_real_p"]  - gi["mae_real_b"]).astype(float)
            row = {k: gi[k].iloc[0] for k in group_keys}
            row.update({
                "perturbation": perturbation,
                "n_pairs": int(len(gi)),
                "delta_rmse_mean": float(delta_rmse.mean()),
                "delta_rmse_pct_mean": float((delta_rmse / gi["rmse_real_b"].astype(float)).mean() * 100.0) if len(gi)>0 else np.nan,
                "delta_mae_mean": float(delta_mae.mean()),
                "delta_tod_mean": np.nan   # 아래 delta‑bar엔 TOD 민감도 상세가 있음
            })
            out_rows.append(row)

    if out_rows:
        out = pd.DataFrame(out_rows)
        _atomic_write_csv(summary_csv, out)

# ============================================================
# 4) Perturb delta‑bar (원인 지표 Δ)
#    - ETTm2: {gc_kernel_tod_dcor, gc_feat_tod_dcor, gc_feat_tod_r2}
#    - ETTh2: {cg_event_gain, |cg_bestlag|, cg_spearman_mean}
# ============================================================
def rebuild_w3_perturb_delta_bar(
    results_csv: str | Path,
    detail_csv: str | Path,
    summary_csv: str | Path,
    filter_dataset: Optional[str] = None,
) -> None:
    rpath = Path(results_csv)

    # 헤더만이라도 보장 생성하는 헬퍼
    def _emit_empties():
        _atomic_write_csv(
            detail_csv,
            pd.DataFrame(columns=["experiment_type","dataset","horizon","seed","perturbation","metric","delta"])
        )
        _atomic_write_csv(
            summary_csv,
            pd.DataFrame(columns=["experiment_type","dataset","horizon","perturbation","metric","delta_mean","n"])
        )

    if not rpath.exists():
        _emit_empties(); return

    R = pd.read_csv(rpath)
    if R.empty:
        _emit_empties(); return

    base = R[R["mode"]=="none"]
    pert = R[R["mode"]!="none"]
    key_cols = ["experiment_type","dataset","horizon","seed","model_tag"]
    M = pert.merge(base, on=key_cols, suffixes=("_p","_b"), how="left")

    # 현재 데이터셋 폴더에 쓰는 동안, 해당 데이터셋만 남겨 저장
    if filter_dataset is not None:
        M = M[M["dataset"].astype(str) == str(filter_dataset)]

    if M.empty:
        _emit_empties(); return

    detail_rows: List[Dict[str,Any]] = []

    for _, row in M.iterrows():
        ds = str(row["dataset"])                 # 조인 키: 접미사 없음
        pert_name = str(row.get("mode_p", ""))   # 비키: 접미사 사용
        H = int(row["horizon"])                  # 조인 키: 접미사 없음
        S = int(row["seed"])                     # 조인 키: 접미사 없음

        if pert_name == "tod_shift":
            for m in ["gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2"]:
                vp = float(row.get(m+"_p", np.nan))
                vb = float(row.get(m+"_b", np.nan))
                if np.isfinite(vp) and np.isfinite(vb):
                    detail_rows.append({
                        "experiment_type":"W3","dataset":ds,"horizon":H,"seed":S,
                        "perturbation":"tod_shift","metric":m,"delta":float(vp - vb)
                    })

        elif pert_name == "smooth":
            # event_gain
            vp = float(row.get("cg_event_gain_p", np.nan))
            vb = float(row.get("cg_event_gain_b", np.nan))
            if np.isfinite(vp) and np.isfinite(vb):
                detail_rows.append({
                    "experiment_type":"W3","dataset":ds,"horizon":H,"seed":S,
                    "perturbation":"smooth","metric":"cg_event_gain","delta":float(vp - vb)
                })
            # |bestlag|
            vp = float(row.get("cg_bestlag_p", np.nan))
            vb = float(row.get("cg_bestlag_b", np.nan))
            if np.isfinite(vp) and np.isfinite(vb):
                detail_rows.append({
                    "experiment_type":"W3","dataset":ds,"horizon":H,"seed":S,
                    "perturbation":"smooth","metric":"cg_bestlag","delta":float(abs(vp) - abs(vb))
                })
            # spearman
            vp = float(row.get("cg_spearman_mean_p", np.nan))
            vb = float(row.get("cg_spearman_mean_b", np.nan))
            if np.isfinite(vp) and np.isfinite(vb):
                detail_rows.append({
                    "experiment_type":"W3","dataset":ds,"horizon":H,"seed":S,
                    "perturbation":"smooth","metric":"cg_spearman_mean","delta":float(vp - vb)
                })

    D = pd.DataFrame(detail_rows) if detail_rows else pd.DataFrame(
        columns=["experiment_type","dataset","horizon","seed","perturbation","metric","delta"]
    )
    _atomic_write_csv(detail_csv, D)

    if not D.empty:
        S = (D.groupby(["experiment_type","dataset","horizon","perturbation","metric"], as_index=False)
               .agg(delta_mean=("delta","mean"), n=("delta","size")))
    else:
        S = pd.DataFrame(columns=["experiment_type","dataset","horizon","perturbation","metric","delta_mean","n"])
    _atomic_write_csv(summary_csv, S)

# ============================================================
# 5) Window errors (창 오차 집계)
# ============================================================
def save_w3_window_errors_detail(ctx, base_errors, pert_errors, perturbation, out_root="results"):
    """
    W3 window-wise errors 저장(요구사항 (4)/(4-1)):
      - per-(dataset/horizon/seed) 분할 CSV: window_errors_detail_H{h}_S{seed}.csv
      - 요약 CSV(window_errors_summary.csv) 갱신(중복-업데이트)
      - 전체 합본 Parquet(W3_window_errors_all.parquet) 통합(메모리 안전)
    - 모든 라벨은 RMSE가 아니라 'MSE'로 저장 (실제 값은 mean squared error)
    """
    dataset = str(ctx.get("dataset", ""))
    horizon = int(ctx.get("horizon", 96))
    seed = int(ctx.get("seed", 0))
    model_tag = str(ctx.get("model_tag", "CTSF"))
    exp_type = str(ctx.get("experiment_type", "W3"))
    run_tag = str(ctx.get("run_tag", f"{dataset}-h{horizon}-s{seed}-{exp_type}-{perturbation}"))

    base_errors = np.asarray(base_errors, dtype=np.float64)
    pert_errors = np.asarray(pert_errors, dtype=np.float64)

    # 출력 디렉토리
    base_dir = Path(out_root) / f"results_{exp_type}" / dataset / "window_errors"
    base_dir.mkdir(parents=True, exist_ok=True)

    # 1) per-run detail CSV (wide; mse_base, mse_pert)
    n = int(min(len(base_errors), len(pert_errors)))
    df = pd.DataFrame({
        "experiment_type": [exp_type] * n,
        "dataset": [dataset] * n,
        "horizon": np.full(n, horizon, dtype=np.int32),
        "seed": np.full(n, seed, dtype=np.int32),
        "model_tag": [model_tag] * n,
        "perturbation": [str(perturbation)] * n,
        "run_tag": [run_tag] * n,
        "window_index": np.arange(n, dtype=np.int32),
        "mse_base": base_errors[:n].astype("float32"),
        "mse_perturb": pert_errors[:n].astype("float32"),
    })
    detail_path = base_dir / f"window_errors_detail_H{horizon}_S{seed}.csv"
    _atomic_write_csv(detail_path, df)

    # 2) summary CSV (중복-업데이트)
    summary_row = {
        "experiment_type": exp_type,
        "dataset": dataset,
        "horizon": horizon,
        "seed": seed,
        "perturbation": str(perturbation),
        "run_tag": run_tag,
        "n_windows": int(n),
        "window_mse_median_base": float(np.nanmedian(base_errors[:n])) if n > 0 else np.nan,
        "window_mse_p90_base": float(np.nanpercentile(base_errors[:n], 90)) if n > 0 else np.nan,
        "window_mse_median_perturb": float(np.nanmedian(pert_errors[:n])) if n > 0 else np.nan,
        "window_mse_p90_perturb": float(np.nanpercentile(pert_errors[:n], 90)) if n > 0 else np.nan,
        "rank_preservation_rate": float(np.mean((pert_errors[:n] > base_errors[:n]).astype(np.float32))) if n > 0 else np.nan,
    }
    summ_path = base_dir / "window_errors_summary.csv"
    _append_or_update_row(summ_path, summary_row,
                          subset_keys=["experiment_type", "dataset", "horizon", "seed", "perturbation", "run_tag"])

    # 3) 전량 Parquet 통합 (메모리 안전; CSV → Arrow Table 순차 변환)
    try:
        consolidate_window_errors_parquet(results_root=str(out_root), exp_type=exp_type)
    except Exception as _e:
        print(f"[W3] Parquet 통합 경고: {_e}")

def consolidate_window_errors_parquet(results_root="results", exp_type="W3"):
    """
    모든 per-run detail CSV를 하나의 Parquet로 통합 (스키마/압축 지정, 파일 단위 스트리밍).
    - 메모리에 모든 데이터를 한 번에 올리지 않는다.
    - 스키마: (dataset:str, horizon:int16, seed:int16, mode/perturbation:str, window_index:int32, mse_base:float32, mse_perturb:float32)
    결과: {results_root}/results_{exp_type}/W3_window_errors_all.parquet
    """
    results_dir = Path(results_root) / f"results_{exp_type}"
    out_path = results_dir / "W3_window_errors_all.parquet"
    schema = pa.schema([
        ("experiment_type", pa.string()),
        ("dataset", pa.string()),
        ("horizon", pa.int16()),
        ("seed", pa.int16()),
        ("model_tag", pa.string()),
        ("perturbation", pa.string()),
        ("run_tag", pa.string()),
        ("window_index", pa.int32()),
        ("mse_base", pa.float32()),
        ("mse_perturb", pa.float32()),
    ])

    writer = None
    try:
        for dataset_dir in results_dir.iterdir():
            if not dataset_dir.is_dir():
                continue
            wnd_dir = dataset_dir / "window_errors"
            if not wnd_dir.exists():
                continue
            for csv_file in sorted(wnd_dir.glob("window_errors_detail_H*_S*.csv")):
                # Arrow CSV reader (스트리밍)
                try:
                    table = pa.csv.read_csv(
                        csv_file,
                        read_options=pa.csv.ReadOptions(use_threads=True, block_size=1 << 20),
                    )
                    # 필요 열만 캐스팅/보정
                    keep_cols = [f.name for f in schema]
                    # 누락 칼럼 보정
                    for name in keep_cols:
                        if name not in table.column_names:
                            # 빈 칼럼 추가
                            table = table.append_column(name, pa.array([None] * table.num_rows, type=schema.field(name).type))
                    # 순서/캐스팅
                    table = table.select(keep_cols).cast(schema)
                except Exception:
                    # pandas 경유 fallback
                    df = pd.read_csv(csv_file, dtype={
                        "experiment_type": "string",
                        "dataset": "string",
                        "horizon": "Int16",
                        "seed": "Int16",
                        "model_tag": "string",
                        "perturbation": "string",
                        "run_tag": "string",
                        "window_index": "Int32",
                        "mse_base": "float32",
                        "mse_perturb": "float32",
                    })
                    table = pa.Table.from_pandas(df, schema=schema, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(out_path, schema, compression="snappy")
                writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
