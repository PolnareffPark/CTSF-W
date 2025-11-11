# ============================================================
# utils/experiment_metrics/w2_metrics.py
#   - W2 실험 특화 지표(정렬·변동성·선택도·이벤트 반응) 완성본
# ============================================================

from __future__ import annotations
from typing import Dict, Optional
import numpy as np
from scipy import stats

from utils.metrics import _dcor_u  # 거리상관(U-statistic)

def _safe_pearson_r2(x, y) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 3 or y.size < 3:
        return np.nan
    sx, sy = np.std(x, ddof=0), np.std(y, ddof=0)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return float(r*r) if np.isfinite(r) else np.nan

def _asf(x):
    try:
        a = np.asarray(x, dtype=float)
        return a if np.isfinite(a).any() else None
    except Exception:
        return None

def _collapse_per_sample(a: np.ndarray, n_expected: Optional[int] = None) -> np.ndarray:
    """
    배치/레이어/게이트 축을 평균해 샘플 축만 남김.
    n_expected가 주어지면 그 크기와 일치하는 축을 우선적으로 샘플 축으로 선택.
    """
    a = np.asarray(a, dtype=float)
    if a.ndim == 1:
        return a
    if n_expected is not None and n_expected > 1:
        cand = [ax for ax, sz in enumerate(a.shape) if sz == n_expected]
        s_ax = cand[0] if cand else int(np.argmax(a.shape))
    else:
        s_ax = int(np.argmax(a.shape))
    axes = tuple(i for i in range(a.ndim) if i != s_ax)
    v = np.nanmean(a, axis=axes) if axes else a
    return v.reshape(-1)

def _align_same_len(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    L = min(len(x), len(y))
    if L < 3:
        return np.array([]), np.array([])
    return x[:L], y[:L]

def compute_w2_metrics(
    model,
    hooks_data: Optional[Dict] = None,
    tod_vec: Optional[np.ndarray] = None,
    direct_evidence: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    동적/정적 모두 대응. 훅 키를 폭넓게 수용하고, 길이 불일치 시 공통 길이로 자동 정렬.
    dynamic일 때 w2_gate_tod_alignment / w2_gate_gru_state_alignment가 비지 않도록 강건화.
    """
    metrics: Dict[str, float] = {}

    gate_per_sample = None
    gru_per_sample  = None
    N_tod = tod_vec.shape[0] if isinstance(tod_vec, np.ndarray) else None

    # 1) 게이트/GRU 훅 수집
    if hooks_data:
        # Gate candidates
        gate_src = None
        for k in ["gate_outputs","gates","gate_values","gate_output_list","gates_time_major"]:
            if k in hooks_data and hooks_data[k] is not None:
                gate_src = hooks_data[k]; break
        if gate_src is None:
            # stage dict에서도 수집 허용
            for k in ["gates_by_stage","gate_by_stage","gate_logs_by_stage"]:
                if k in hooks_data and isinstance(hooks_data[k], dict) and hooks_data[k]:
                    arrs = []
                    for s in ["S","M","D","s","m","d"]:
                        if s in hooks_data[k]:
                            a = _asf(hooks_data[k][s])
                            if a is not None: arrs.append(a)
                    if arrs:
                        gate_src = np.stack(arrs, axis=0)
                    break
        if gate_src is not None:
            if isinstance(gate_src, list):
                ps = []
                for a in gate_src:
                    a = _asf(a)
                    if a is not None:
                        ps.append(_collapse_per_sample(a, n_expected=N_tod))
                if len(ps)>0:
                    gate_per_sample = np.concatenate(ps, axis=0)
            else:
                a = _asf(gate_src)
                if a is not None:
                    gate_per_sample = _collapse_per_sample(a, n_expected=N_tod)

        # GRU candidates
        for k in ["gru_states","gru_hidden","gru_h_list","h_list","hidden_states"]:
            if k in hooks_data and hooks_data[k] is not None:
                v = hooks_data[k]
                if isinstance(v, list):
                    ps = []
                    for a in v:
                        a = _asf(a)
                        if a is not None:
                            ps.append(_collapse_per_sample(a, n_expected=N_tod))
                    if len(ps)>0:
                        gru_per_sample = np.concatenate(ps, axis=0)
                else:
                    a = _asf(v)
                    if a is not None:
                        gru_per_sample = _collapse_per_sample(a, n_expected=N_tod)
                break

    # 2) 변동성/엔트로피
    if gate_per_sample is not None:
        metrics["w2_gate_variability_time"]   = float(np.nanstd(gate_per_sample))
        metrics["w2_gate_variability_sample"] = float(np.nanstd(gate_per_sample))
        pos = gate_per_sample[gate_per_sample > 1e-8]
        if pos.size > 0:
            p = pos / (pos.sum() + 1e-12)
            metrics["w2_gate_entropy"] = float(-np.sum(p*np.log(p + 1e-12)))
        else:
            metrics["w2_gate_entropy"] = float("nan")
    else:
        metrics["w2_gate_variability_time"]   = float("nan")
        metrics["w2_gate_variability_sample"] = float("nan")
        metrics["w2_gate_entropy"]            = float("nan")

    metrics.setdefault("w2_channel_selectivity_kurtosis", float("nan"))
    metrics.setdefault("w2_channel_selectivity_sparsity", float("nan"))

    # 3) TOD 정렬
    if gate_per_sample is not None and isinstance(tod_vec, np.ndarray):
        gx = gate_per_sample.reshape(-1)
        ty = np.asarray(tod_vec, dtype=float).reshape(-1, 2)
        gx, ty = _align_same_len(gx, ty)
        if gx.size >= 3 and ty.size >= 3:
            try:
                metrics["w2_gate_tod_alignment"] = _dcor_u(gx.reshape(-1,1), ty)
            except Exception:
                metrics["w2_gate_tod_alignment"] = float("nan")
        else:
            metrics["w2_gate_tod_alignment"] = float("nan")
    else:
        metrics["w2_gate_tod_alignment"] = float("nan")

    # 4) GRU 정렬(r^2)
    if gate_per_sample is not None and gru_per_sample is not None:
        gx = gate_per_sample.reshape(-1)
        hy = gru_per_sample.reshape(-1)
        gx, hy = _align_same_len(gx, hy)
        metrics["w2_gate_gru_state_alignment"] = _safe_pearson_r2(gx, hy)
    else:
        metrics["w2_gate_gru_state_alignment"] = float("nan")

    # 5) 채널 선택도 (첨도/희소성) - 훅이 채널 축을 제공하지 않는 환경에서는 NaN 유지가 안전
    # (추후 per-channel 훅이 추가되면 여기를 보강)

    # 6) 이벤트 조건 반응(동적에 한해 정의)
    if direct_evidence is not None and gate_per_sample is not None:
        cg_event_gain = direct_evidence.get("cg_event_gain", np.nan)
        if np.isfinite(cg_event_gain):
            gv = metrics.get("w2_gate_variability_time", np.nan)
            metrics["w2_event_conditional_response"] = float(gv * cg_event_gain) if np.isfinite(gv) else float("nan")
        else:
            metrics["w2_event_conditional_response"] = float("nan")
    else:
        metrics["w2_event_conditional_response"] = float("nan")

    return metrics
