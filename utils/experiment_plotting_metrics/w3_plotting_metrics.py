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

_W3_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag"]

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

def _atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False); tmp.replace(path)

def _append_or_update_rows(csv_path: str | Path, rows: List[Dict[str,Any]], subset_keys: List[str]) -> None:
    if not rows: return
    path = _ensure_dir(csv_path)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        # union
        for c in new.columns:
            if c not in old.columns: old[c] = np.nan
        for c in old.columns:
            if c not in new.columns: new[c] = np.nan
        df = pd.concat([old, new], ignore_index=True)
        df.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write(df, path)
    else:
        new.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write(new, path)

def _append_or_update_row(csv_path: str | Path, row: Dict[str,Any], subset_keys: List[str]) -> None:
    _append_or_update_rows(csv_path, [row], subset_keys)

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
    ang = np.arctan2(s, c)  # [-pi, pi]
    ang = (ang + 2*np.pi) % (2*np.pi)
    idx = np.floor(ang / (2*np.pi) * T).astype(int)
    idx = np.clip(idx, 0, T-1)
    return idx

def _stage_split_mean_per_sample(gates: np.ndarray) -> Dict[str, np.ndarray]:
    """
    gates: (N, L, G) or (Batches ... concat) → 샘플축 N, 레이어축 L, gate축 G
    반환: {'S': (N,), 'M': (N,), 'D': (N,)}
    """
    if gates is None or not isinstance(gates, np.ndarray) or gates.ndim < 2:
        return {}
    arr = np.asarray(gates, float)
    if arr.ndim == 2:  # (N, G)
        arr = arr.reshape(arr.shape[0], 1, arr.shape[1])
    N, L, G = arr.shape[0], arr.shape[1], arr.shape[2]
    if L <= 0:
        return {}
    # 레이어를 3분할
    b1 = max(1, L // 3)
    b2 = max(b1+1, (2*L)//3)
    ss = arr[:, :b1, :].mean(axis=(1, 2))
    mm = arr[:, b1:b2, :].mean(axis=(1, 2))
    dd = arr[:, b2:,  :].mean(axis=(1, 2)) if b2 < L else arr[:, -1:, :].mean(axis=(1, 2))
    return {"S": ss, "M": mm, "D": dd}

def _collect_gate_arrays_from_hooks(hooks_data: Dict[str,Any]) -> Optional[np.ndarray]:
    if not isinstance(hooks_data, dict): 
        return None

    # 0) 우선순위: 이미 S/M/D로 집계된 값을 쓰자
    for k in ("gates_by_stage","gate_by_stage","gate_logs_by_stage"):
        stage_dict = hooks_data.get(k, None)
        if isinstance(stage_dict, dict) and stage_dict:
            out = {}
            for st in ("S","M","D"):
                a = stage_dict.get(st) or stage_dict.get(st.lower())
                if a is None:
                    continue
                aa = np.asarray(a, float)
                # (N, *) → (N,)으로 평균
                out[st] = aa.reshape(aa.shape[0], -1).mean(axis=1)
            if out:   # {'S':(N,), 'M':(N,), 'D':(N,)} 반환 신호
                # 특별 반환 신호: dict를 담아 보내면 caller가 그대로 사용
                out["_is_stage_dict"] = True
                return out  # type: ignore

    # 1) 원본 텐서(게이트 로그) 병합 → (N,L,G)
    src = None
    if "gate_outputs" in hooks_data and hooks_data["gate_outputs"]:
        bags = []
        for a in hooks_data["gate_outputs"]:
            try:
                a = np.asarray(a, float)
                if a.ndim == 2:   # (B,G) → (B,1,G)
                    a = a.reshape(a.shape[0], 1, a.shape[1])
                elif a.ndim > 3:  # (B,*,G) → (B,L,G)로 접기
                    L = int(np.prod(a.shape[1:-1]))
                    a = a.reshape(a.shape[0], L, a.shape[-1])
                if a.ndim == 3:
                    bags.append(a)
            except Exception:
                continue
        if bags:
            try:    src = np.concatenate(bags, axis=0)
            except Exception:
            # 실패 시 첫 번째만 사용
                src = bags[0]
    return src


# ============================================================
# 1) Gate‑TOD heatmap
#    - 샘플별 gate 강도 ↔ 해당 샘플 TOD bin 으로 집계 (T=dataset별)
# ============================================================
def compute_and_save_w3_gate_tod_heatmap(
    ctx: Dict[str,Any],                    # plot_type='gate_tod_heatmap'
    hooks_data: Dict[str,Any],
    tod_vec: np.ndarray,
    dataset: str,
    out_dir: str | Path,
    amp_method: str = "range",
) -> None:
    assert ctx.get("plot_type") == "gate_tod_heatmap"
    T = _infer_tod_bins(dataset)
    idx = _tod_index_from_vec(tod_vec, T)
    if idx.size == 0:
        return

    # 1) stage dict가 있으면 우선 사용
    st: Dict[str, np.ndarray] = {}
    for k in ("gates_by_stage","gate_by_stage","gate_logs_by_stage"):
        sd = hooks_data.get(k, None)
        if isinstance(sd, dict) and sd:
            for stage in ("S","M","D"):
                a = sd.get(stage) or sd.get(stage.lower())
                if a is None:
                    continue
                aa = np.asarray(a, float)
                aa = aa.reshape(aa.shape[0], -1).mean(axis=1)  # (N,*)→(N,)
                st[stage] = aa
            break

    # 2) 없으면 원래 경로: gate_outputs → stage split
    if not st:
        gates = _collect_gate_arrays_from_hooks(hooks_data)
        if gates is None:
            return
        st = _stage_split_mean_per_sample(np.asarray(gates, float))  # {'S':(N,), 'M':(N,), 'D':(N,)}

    rows: List[Dict[str,Any]] = []
    amp: Dict[str, float] = {"S": np.nan, "M": np.nan, "D": np.nan}
    covered = np.zeros(T, dtype=bool)  # 어떤 stage든 sample이 존재한 TOD bin

    def _amp_fn(x: np.ndarray) -> float:
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if x.size == 0:
            return np.nan
        return float(np.nanstd(x) if amp_method == "std"
                     else (np.nanmax(x) - np.nanmin(x)))

    # 3) stage별 TOD 집계
    for stage in ("S","M","D"):
        v = np.asarray(st.get(stage), float)
        if v.size == 0:
            continue
        bins: List[List[float]] = [[] for _ in range(T)]
        msk = np.isfinite(v)
        N = min(v.shape[0], idx.shape[0])
        for i in range(N):
            if not msk[i]:
                continue
            j = int(idx[i])
            bins[j].append(float(v[i]))

        means = [float(np.mean(b)) if len(b) > 0 else np.nan for b in bins]
        counts = [int(len(b)) for b in bins]

        # detail: 빈 bin도 NaN + sample_count로 기록
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

        # amp
        finite_means = [m for m in means if np.isfinite(m)]
        amp[stage] = _amp_fn(np.array(finite_means, float)) if finite_means else np.nan

    # 4) summary
    valid_bins = int(covered.sum())
    summary_row = {
        **{k: ctx[k] for k in _W3_KEYS if k in ctx},
        "tod_bins": int(T),
        "tod_amp_s": amp.get("S", np.nan),
        "tod_amp_m": amp.get("M", np.nan),
        "tod_amp_d": amp.get("D", np.nan),
        "tod_amp_mean": float(np.nanmean([v for v in amp.values() if np.isfinite(v)])) \
                        if any(np.isfinite(list(amp.values()))) else np.nan,
        "valid_ratio": float(valid_bins) / float(T),
        "nan_count": int(T - valid_bins),
        "amp_method": amp_method,
    }

    _append_or_update_row(Path(out_dir) / "gate_tod_heatmap_summary.csv",
                          summary_row, subset_keys=_W3_KEYS)
    _append_or_update_rows(Path(out_dir) / "gate_tod_heatmap_detail.csv",
                           rows, subset_keys=_W3_KEYS + ["tod", "stage"])


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
    import pandas as pd, numpy as np
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
        _atomic_write(out, _ensure_dir(summary_csv))


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
    import pandas as pd, numpy as np
    rpath = Path(results_csv)

    # 헤더만이라도 보장 생성하는 헬퍼
    def _emit_empties():
        _atomic_write(
            pd.DataFrame(columns=["experiment_type","dataset","horizon","seed","perturbation","metric","delta"]),
            _ensure_dir(detail_csv)
        )
        _atomic_write(
            pd.DataFrame(columns=["experiment_type","dataset","horizon","perturbation","metric","delta_mean","n"]),
            _ensure_dir(summary_csv)
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
    _atomic_write(D, _ensure_dir(detail_csv))

    if not D.empty:
        S = (D.groupby(["experiment_type","dataset","horizon","perturbation","metric"], as_index=False)
               .agg(delta_mean=("delta","mean"), n=("delta","size")))
    else:
        S = pd.DataFrame(columns=["experiment_type","dataset","horizon","perturbation","metric","delta_mean","n"])
    _atomic_write(S, _ensure_dir(summary_csv))
