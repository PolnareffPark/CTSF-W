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
    ctx: Dict[str, Any],
    hooks_data: Optional[Dict[str, Any]] = None,
    tod_vec: Optional[np.ndarray] = None,
    dataset: Optional[str] = None,
    out_dir: Optional[Path] = None,
    amp_method: str = "range",
) -> None:
    """
    W3 Gate×TOD heatmap writer (detail + summary), robust to the W3 call form.

    Accepts:
      - ctx:     standard context dict (experiment_type, dataset, horizon, seed, mode, model_tag, run_tag)
      - hooks_data: gate outputs collected during evaluation (optional; if missing, function no-ops)
      - tod_vec: [sin, cos] time-of-day vectors for test windows (optional; if missing, T is inferred)
      - dataset: dataset tag (defaults to ctx["dataset"])
      - out_dir: directory to write detail/summary CSVs (defaults to results_W3/<dataset>/gate_tod_heatmap)
      - amp_method: "range" (default) or "std" for amplitude summary

    Output:
      - gate_tod_heatmap_detail.csv (rows: S/M/D × TOD bins, columns: _W3_KEYS + ["tod","stage","gate_mean"])
      - gate_tod_heatmap_summary.csv (single row per run with tod_amp_s/m/d/mean and data coverage)
    """
    if not isinstance(ctx, dict):
        return

    dataset = dataset or str(ctx.get("dataset", ""))
    out_dir = Path(out_dir) if out_dir is not None else (Path("results") / f"results_{ctx.get('experiment_type','W3')}" / dataset / "gate_tod_heatmap")
    out_dir.mkdir(parents=True, exist_ok=True)

    # If no hooks, nothing to do (keep robustness)
    if not isinstance(hooks_data, dict) or not hooks_data:
        return

    # Helper local functions
    def _to_f32(x):
        try: return np.float32(x).item()
        except Exception: return np.nan

    def _zscore(arr):
        mu = np.nanmean(arr); sd = np.nanstd(arr)
        if not np.isfinite(sd) or sd == 0: return np.zeros_like(arr, dtype=np.float64)
        return (arr - mu) / sd

    def _logit(arr):
        eps = 1e-6
        p = np.clip(arr, eps, 1 - eps)
        return np.log(p / (1 - p))

    def _amp(arr: np.ndarray) -> float:
        if arr.size == 0 or not np.isfinite(arr).any():
            return np.nan
        return _to_f32(np.nanstd(arr) if amp_method == "std" else (np.nanmax(arr) - np.nanmin(arr)))

    # Infer TOD bins
    T = _infer_tod_bins(dataset)
    tod_idx = _tod_index_from_vec(np.asarray(tod_vec) if tod_vec is not None else np.zeros((0, 2), dtype=float), T)

    # hooks_data may already contain per-stage means (W2/W3 compatible). Try both ways.
    # Preferred keys for per‑stage daily profile vectors:
    s = np.asarray(hooks_data.get("gate_tod_mean_s", []), dtype=np.float64)
    m = np.asarray(hooks_data.get("gate_tod_mean_m", []), dtype=np.float64)
    d = np.asarray(hooks_data.get("gate_tod_mean_d", []), dtype=np.float64)

    # If empty, try to aggregate from raw gates ({'S': [...], 'M': [...], 'D': [...]})
    if max(len(s), len(m), len(d)) == 0:
        # Fallback: raw stage arrays (each of length N_samples). We bin by tod_idx and average.
        def _bin_mean(vec: np.ndarray, idx: np.ndarray, T: int) -> np.ndarray:
            if vec.size == 0 or idx.size == 0: return np.full((T,), np.nan, dtype=np.float64)
            out = np.full((T,), np.nan, dtype=np.float64)
            for t in range(T):
                sel = vec[idx == t]
                if sel.size > 0:
                    out[t] = float(np.nanmean(sel))
            return out

        raw_S = np.asarray(hooks_data.get("S", []), dtype=np.float64).reshape(-1) if "S" in hooks_data else np.array([], dtype=np.float64)
        raw_M = np.asarray(hooks_data.get("M", []), dtype=np.float64).reshape(-1) if "M" in hooks_data else np.array([], dtype=np.float64)
        raw_D = np.asarray(hooks_data.get("D", []), dtype=np.float64).reshape(-1) if "D" in hooks_data else np.array([], dtype=np.float64)
        s = _bin_mean(raw_S, tod_idx, T)
        m = _bin_mean(raw_M, tod_idx, T)
        d = _bin_mean(raw_D, tod_idx, T)

    # Optional internal transforms (kept off by default; enable via module-level flags if present)
    try:
        if "_W3_TOD_USE_LOGIT" in globals() and globals()["_W3_TOD_USE_LOGIT"]:
            if s.size: s = _logit(s)
            if m.size: m = _logit(m)
            if d.size: d = _logit(d)
    except Exception:
        pass

    try:
        if "_W3_TOD_USE_ZSCORE" in globals() and globals()["_W3_TOD_USE_ZSCORE"]:
            if s.size: s = _zscore(s)
            if m.size: m = _zscore(m)
            if d.size: d = _zscore(d)
    except Exception:
        pass

    # Build detail DataFrame
    tod_labels = np.arange(max(len(s), len(m), len(d), T), dtype=np.int32)
    rows = []
    base = {k: ctx.get(k) for k in _W3_KEYS}
    for stage, vec in (("s", s), ("m", m), ("d", d)):
        L = len(vec)
        for t in range(len(tod_labels)):
            val = float(vec[t]) if t < L and np.isfinite(vec[t]) else np.nan
            r = dict(base)
            r.update({"tod": int(t), "stage": stage, "gate_mean": val})
            rows.append(r)
    detail = pd.DataFrame(rows)

    # Append/update detail
    _append_or_update_rows(
        csv_path=out_dir / "gate_tod_heatmap_detail.csv",
        rows=rows,
        subset_keys=_W3_KEYS + ["tod", "stage"],
    )

    # Summary amplitudes
    tod_amp_s = _amp(s)
    tod_amp_m = _amp(m)
    tod_amp_d = _amp(d)
    tod_amp_mean = _to_f32(np.nanmean([tod_amp_s, tod_amp_m, tod_amp_d]))
    summary_row = dict(base)
    summary_row.update({
        "tod_amp_s": tod_amp_s,
        "tod_amp_m": tod_amp_m,
        "tod_amp_d": tod_amp_d,
        "tod_amp_mean": tod_amp_mean,
        "valid_ratio": float(np.isfinite(detail["gate_mean"]).mean()) if "gate_mean" in detail else np.nan,
        "nan_count": int(detail["gate_mean"].isna().sum()) if "gate_mean" in detail else 0,
    })
    _append_or_update_row(
        csv_path=out_dir / "gate_tod_heatmap_summary.csv",
        row=summary_row,
        subset_keys=_W3_KEYS,
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
def save_w3_window_errors_detail(
    ctx: Dict[str, Any],
    errors: np.ndarray,
    mode: str,
    out_root: str = "results"
) -> None:
    """
    Save window-wise **MSE** for a single (dataset,horizon,seed,mode) run in LONG format.

    Writes (per dataset):
      1) results_W3/<dataset>/window_errors/window_errors_detail_H{H}_S{seed}.csv
         - Appends/updates rows (long format): one column "mse", de-dup by ("mode","window_index")
         - File is per (H,S), so subset keys can be just ("mode","window_index").
      2) results_W3/<dataset>/window_errors/window_errors_summary.csv
         - Upserts a single summary row per (dataset,horizon,seed,model_tag,mode)
           with stats computed from the single "mse" column.
      3) results_W3/<dataset>/window_errors/window_errors_all.parquet
         - Consolidated Parquet (append-like via read+concat+write), dtype-optimized.

    Notes:
      * All labels intentionally use **MSE** (not RMSE).
      * Memory-safe: processes only the current run's vector; no global in-memory accumulation.
    """

    # ---- Context ----
    exp_type  = str(ctx.get("experiment_type", "W3"))
    dataset   = str(ctx.get("dataset", ""))
    horizon   = int(ctx.get("horizon"))
    seed      = int(ctx.get("seed"))
    model_tag = str(ctx.get("model_tag", "HyperConv"))
    run_tag   = str(ctx.get("run_tag", f"{exp_type}_{dataset}_{horizon}_{seed}_{mode}"))

    arr = np.asarray(errors, dtype=np.float64)
    n   = int(arr.size)

    # ---- Output layout ----
    base_dir    = Path(out_root) / f"results_{exp_type}" / dataset / "window_errors"
    base_dir.mkdir(parents=True, exist_ok=True)
    detail_csv  = base_dir / f"window_errors_detail_H{horizon}_S{seed}.csv"
    summary_csv = base_dir / "window_errors_summary.csv"
    parquet_all = base_dir / "window_errors_all.parquet"

    # ---- 1) DETAIL CSV (long format; one column "mse") ----
    df_new = pd.DataFrame({
        "experiment_type": [exp_type] * n,
        "dataset":         [dataset]  * n,
        "horizon":         np.full(n, horizon, dtype=np.int16),
        "seed":            np.full(n, seed,    dtype=np.int16),
        "model_tag":       [model_tag] * n,
        "mode":            [str(mode)] * n,
        "run_tag":         [run_tag] * n,
        "window_index":    np.arange(n, dtype=np.int32),
        "mse":             arr.astype("float32"),
    })

    # Upsert by per-file keys (file already split by dataset/H/S)
    subset_keys = ["mode", "window_index"]
    if detail_csv.exists():
        old = pd.read_csv(detail_csv)
        # column union for forward/backward compatibility
        for c in df_new.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in df_new.columns:
                df_new[c] = np.nan
        merged = pd.concat([old, df_new], ignore_index=True)
        merged.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        # enforce compact dtypes
        merged = merged.astype({
            "horizon": "int16",
            "seed": "int16",
            "window_index": "int32",
            "mse": "float32",
        }, errors="ignore")
        _atomic_write_csv(detail_csv, merged)
    else:
        df_new.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(detail_csv, df_new)

    # ---- 2) SUMMARY CSV (one row per mode) ----
    s = arr.astype(np.float64)  # for robust nan-safe stats
    summ_row = {
        "experiment_type": exp_type,
        "dataset":         dataset,
        "horizon":         int(horizon),
        "seed":            int(seed),
        "model_tag":       model_tag,
        "mode":            str(mode),
        "run_tag":         run_tag,
        "n_windows":       int(n),
        # single-column "mse" stats
        "win_mse_mean":    float(np.nanmean(s))   if n > 0 else np.nan,
        "win_mse_std":     float(np.nanstd(s))    if n > 0 else np.nan,
        "win_mse_median":  float(np.nanmedian(s)) if n > 0 else np.nan,
        "win_mse_p90":     float(np.nanpercentile(s, 90)) if n > 0 else np.nan,
        "win_mse_p95":     float(np.nanpercentile(s, 95)) if n > 0 else np.nan,
        "win_mse_max":     float(np.nanmax(s))    if n > 0 else np.nan,
    }
    df_s = pd.DataFrame([summ_row])

    # Upsert one summary row per (dataset,horizon,seed,model_tag,mode)
    sum_keys = ["experiment_type","dataset","horizon","seed","model_tag","mode"]
    if summary_csv.exists():
        old_s = pd.read_csv(summary_csv)
        for c in df_s.columns:
            if c not in old_s.columns:
                old_s[c] = np.nan
        for c in old_s.columns:
            if c not in df_s.columns:
                df_s[c] = np.nan
        merged_s = pd.concat([old_s, df_s], ignore_index=True)
        merged_s.drop_duplicates(subset=sum_keys, keep="last", inplace=True)
        _atomic_write_csv(summary_csv, merged_s)
    else:
        _atomic_write_csv(summary_csv, df_s)

    # ---- 3) Parquet consolidation (append-like via read+concat) ----
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(
            df_new.astype({
                "horizon": "int16",
                "seed": "int16",
                "window_index": "int32",
                "mse": "float32",
            }, errors="ignore"),
            schema=pa.schema([
                ("experiment_type", pa.string()),
                ("dataset",        pa.string()),
                ("horizon",        pa.int16()),
                ("seed",           pa.int16()),
                ("model_tag",      pa.string()),
                ("mode",           pa.string()),
                ("run_tag",        pa.string()),
                ("window_index",   pa.int32()),
                ("mse",            pa.float32()),
            ]),
            preserve_index=False
        )
        if parquet_all.exists():
            old_tbl = pq.read_table(parquet_all)
            pq.write_table(pa.concat_tables([old_tbl, table]), parquet_all, compression="snappy")
        else:
            pq.write_table(table, parquet_all, compression="snappy")
    except Exception as e:
        # Parquet is optional; CSVs are authoritative
        print(f"[W3] Parquet consolidation skipped: {e}")

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
