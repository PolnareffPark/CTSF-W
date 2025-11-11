# ============================================================
# utils/experiment_metrics/w2_metrics.py
#   - W2 실험 특화 지표(정렬·변동성·선택도·이벤트 반응) 완성본
# ============================================================

from typing import Dict, Optional
import numpy as np
from scipy import stats
from utils.metrics import _dcor_u

def _as_np(a):
    try:
        if hasattr(a, "detach"):
            a = a.detach().cpu().numpy()
        return np.asarray(a)
    except Exception:
        return None

def _to_NLG(arr: np.ndarray) -> Optional[np.ndarray]:
    """게이트 텐서를 (N,L,G)로 표준화."""
    a = _as_np(arr)
    if a is None or a.ndim < 2:
        return None
    if a.ndim == 3:  # (N,L,G)
        return a
    if a.ndim >= 4:
        N = a.shape[0]
        G = a.shape[-1]
        L = int(np.prod(a.shape[1:-1]))
        return a.reshape(N, L, G)
    if a.ndim == 2:  # (N,G) 가정
        return a.reshape(a.shape[0], 1, a.shape[1])
    return None

def _concat_gate_outputs(gate_list):
    """list of arrays -> (N,L,G) 연결."""
    bags = []
    for item in gate_list:
        ng = _to_NLG(item)
        if ng is not None:
            bags.append(ng)
    if not bags:
        return None
    try:
        return np.concatenate(bags, axis=0)
    except Exception:
        return bags[0]

def _safe_kurtosis(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]):  # 상수벡터
        return 0.0
    try:
        return float(stats.kurtosis(x, fisher=True, bias=False, nan_policy="omit"))
    except Exception:
        return float("nan")

def _hoyer_sparsity(x: np.ndarray) -> float:
    x = np.abs(np.asarray(x, float))
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return float("nan")
    l1 = float(np.sum(x))
    l2 = float(np.linalg.norm(x))
    if l2 == 0.0:
        return 0.0
    # (sqrt(n) - L1/L2) / (sqrt(n) - 1)  in [0,1]
    return float((np.sqrt(n) - l1 / l2) / (np.sqrt(n) - 1.0 + 1e-12))

def _align_len(x: np.ndarray, y: np.ndarray):
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
    W2 특화 지표(동적/정적). 동적이면 실측 gate_outputs, 정적이면 stage별/alpha 기반.
    반환: w2_gate_variability_time, w2_gate_variability_sample, w2_gate_entropy,
         w2_gate_tod_alignment, w2_gate_gru_state_alignment,
         w2_event_conditional_response, w2_channel_selectivity_kurtosis, w2_channel_selectivity_sparsity
    """
    m: Dict[str, float] = {
        "w2_gate_variability_time": float("nan"),
        "w2_gate_variability_sample": float("nan"),
        "w2_gate_entropy": float("nan"),
        "w2_gate_tod_alignment": float("nan"),
        "w2_gate_gru_state_alignment": float("nan"),
        "w2_event_conditional_response": float("nan"),
        "w2_channel_selectivity_kurtosis": float("nan"),
        "w2_channel_selectivity_sparsity": float("nan"),
    }

    # -----------------------
    # 1) 게이트 텐서 수집
    # -----------------------
    gates_NLG = None          # (N,L,G)
    gate_channel_vector = None  # (G,)
    gru_list = []

    if isinstance(hooks_data, dict):
        # a) 동적 게이트: gate_outputs(list) 우선
        if "gate_outputs" in hooks_data and hooks_data["gate_outputs"]:
            gates_NLG = _concat_gate_outputs(hooks_data["gate_outputs"])
        # b) stage dict 허용
        if gates_NLG is None:
            for k in ("gates_by_stage", "gate_by_stage", "gate_logs_by_stage"):
                d = hooks_data.get(k)
                if isinstance(d, dict) and d:
                    stage_vecs = []
                    for s in ("S", "M", "D", "s", "m", "d"):
                        if s in d and d[s] is not None:
                            a = _as_np(d[s])
                            if a is None:
                                continue
                            a = np.asarray(a, float)
                            # 마지막 축이 게이트 차원이라고 가정하고 나머지 평균
                            if a.ndim == 1:
                                stage_vecs.append(a.reshape(1, -1))
                            else:
                                gdim = -1
                                axes = tuple(i for i in range(a.ndim) if i != gdim)
                                stage_vecs.append(np.nanmean(a, axis=axes, keepdims=True))
                    if stage_vecs:
                        gate_channel_vector = np.nanmean(np.concatenate(stage_vecs, axis=0), axis=0).reshape(-1)
                    break
        # GRU 상태(가능하면)
        for k in ("gru_states", "gru_hidden", "gru_h_list", "h_list", "hidden_states"):
            if k in hooks_data and hooks_data[k] is not None:
                gru_list = hooks_data[k] if isinstance(hooks_data[k], list) else [hooks_data[k]]
                break

    # c) 완전 폴백(정적 모델 파라미터)
    if gates_NLG is None and gate_channel_vector is None and hasattr(model, "xhconv_blks"):
        chans = []
        for blk in getattr(model, "xhconv_blks", []):
            for name in ("alpha", "cross_gate", "gate", "w", "g"):
                if hasattr(blk, name):
                    v = _as_np(getattr(blk, name))
                    if v is None:
                        continue
                    v = np.asarray(v, float)
                    if v.ndim == 2:  # (2,2) 등
                        chans.append(v.reshape(-1))
                        break
                    if v.ndim == 1:
                        chans.append(v)
                        break
        if chans:
            gate_channel_vector = np.nanmean(np.stack(chans, axis=0), axis=0).reshape(-1)

    # -----------------------
    # 2) 지표 계산
    # -----------------------
    is_dynamic = isinstance(gates_NLG, np.ndarray)

    if is_dynamic:
        # (N,L,G)
        N, L, G = gates_NLG.shape
        # 시간/샘플 변동성
        m["w2_gate_variability_time"]   = float(np.nanmean(np.nanstd(gates_NLG.reshape(N, -1), axis=1)))
        m["w2_gate_variability_sample"] = float(np.nanmean(np.nanstd(gates_NLG, axis=0)))  # (L,G) std의 평균
        # 엔트로피
        flat = gates_NLG.reshape(-1)
        pos = flat[np.isfinite(flat) & (flat > 1e-8)]
        if pos.size > 0:
            p = pos / (pos.sum() + 1e-12)
            m["w2_gate_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))
        # TOD 정렬
        if isinstance(tod_vec, np.ndarray) and tod_vec.ndim == 2 and tod_vec.shape[0] > 2:
            g_strength = np.nanmean(gates_NLG, axis=(1, 2))  # (N,)
            gx, ty = _align_len(g_strength.reshape(-1), tod_vec.reshape(len(tod_vec), -1))
            if gx.size >= 3:
                try:
                    m["w2_gate_tod_alignment"] = float(_dcor_u(gx.reshape(-1,1), ty))
                except Exception:
                    m["w2_gate_tod_alignment"] = float("nan")
        # GRU 정렬
        if gru_list:
            try:
                gru = np.concatenate([_as_np(a) for a in gru_list if _as_np(a) is not None], axis=0)
                if gru.ndim >= 2:
                    g_strength = np.nanmean(gates_NLG, axis=(1, 2))
                    gn = np.linalg.norm(gru, axis=1)
                    gx, hy = _align_len(g_strength.reshape(-1), gn.reshape(-1))
                    if gx.size >= 3:
                        r = np.corrcoef(gx, hy)[0, 1]
                        m["w2_gate_gru_state_alignment"] = float(r * r) if np.isfinite(r) else float("nan")
            except Exception:
                pass
        # 채널 선택도(동적): (G,)
        gate_channel_vector = np.nanmean(gates_NLG, axis=(0, 1))  # (G,)
    else:
        # static: TOD/GRU 정렬은 정의상 0.0
        m["w2_gate_tod_alignment"] = 0.0
        m["w2_gate_gru_state_alignment"] = 0.0

    # 채널 선택도(첨도/희소성): 동적이면 위에서 설정, 아니면 폴백 벡터 사용
    if gate_channel_vector is not None and gate_channel_vector.size >= 2:
        m["w2_channel_selectivity_kurtosis"] = _safe_kurtosis(gate_channel_vector)
        m["w2_channel_selectivity_sparsity"] = _hoyer_sparsity(gate_channel_vector)

    # 이벤트 조건 반응(있는 경우만)
    cg_event_gain = None
    if isinstance(direct_evidence, dict):
        cg_event_gain = direct_evidence.get("cg_event_gain", None)
    if (cg_event_gain is not None and isinstance(cg_event_gain, (int,float)) 
        and np.isfinite(cg_event_gain) and np.isfinite(m.get("w2_gate_variability_time", np.nan))):
        m["w2_event_conditional_response"] = float(m["w2_gate_variability_time"] * cg_event_gain)

    return m
