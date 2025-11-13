# ============================================================
# utils/experiment_metrics/w2_metrics.py
#   - W2 실험 특화 지표(정렬·변동성·선택도·이벤트 반응) 완성본
# ============================================================

from typing import Dict, Optional
import numpy as np
from scipy import stats
from utils.metrics import _dcor_u

def _pearson_r2_safe(x, y, eps: float = 1e-12) -> float:
    """
    피어슨 상관의 제곱(r^2)을 안전하게 계산.
    - 유한값만 사용
    - 길이 < 3 이면 NaN
    - 표준편차(분산) == 0 이면 0.0 (정렬 없음으로 간주)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    L = min(x.size, y.size)
    if L < 3:
        return float("nan")
    x = x[:L] - np.nanmean(x[:L])
    y = y[:L] - np.nanmean(y[:L])
    sx = np.nanstd(x, ddof=0); sy = np.nanstd(y, ddof=0)
    if sx <= eps or sy <= eps:
        return 0.0
    r = float(np.dot(x, y) / (L * sx * sy))
    r = float(np.clip(r, -1.0, 1.0))
    return r * r

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
    W2 특화 지표(동적/정적). 동적이면 실측 gate_outputs, 정적이면 파라미터 폴백.
    반환:
      - w2_gate_variability_time / w2_gate_variability_sample / w2_gate_entropy
      - w2_gate_tod_alignment (dCor), w2_gate_gru_state_alignment (r^2 안전계산)
      - w2_event_conditional_response (= var_time * cg_event_gain; 동적일 때만)
      - w2_channel_selectivity_kurtosis / w2_channel_selectivity_sparsity (채널 선택도)
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
    # 1) 게이트 / GRU 수집
    # -----------------------
    gates_NLG = None                 # (N,L,G)
    gate_channel_vector = None       # (G,)
    gru_list = []

    if isinstance(hooks_data, dict):
        # (a) 동적 게이트: gate_outputs 우선
        go = hooks_data.get("gate_outputs", None)
        if go:
            gates_NLG = _concat_gate_outputs(go)  # 각 원소를 (N,L,G)로 표준화 후 concat

        # (b) stage dict에서 채널 벡터 폴백 구성
        if gate_channel_vector is None:
            for k in ("gates_by_stage", "gate_by_stage", "gate_logs_by_stage"):
                d = hooks_data.get(k)
                if isinstance(d, dict) and d:
                    stage_vecs = []
                    for s in ("S","M","D","s","m","d"):
                        if s in d and d[s] is not None:
                            a = _as_np(d[s])
                            if a is None: 
                                continue
                            a = np.asarray(a, float)
                            # 마지막 축을 게이트 채널로 가정하고 나머지 평균 → (G,)
                            if a.ndim == 1:
                                v = a
                            else:
                                gdim = -1
                                axes = tuple(i for i in range(a.ndim) if i != gdim)
                                v = np.nanmean(a, axis=axes)
                            if v is not None and v.size >= 1:
                                stage_vecs.append(v.reshape(-1))
                    if stage_vecs:
                        gate_channel_vector = np.nanmean(np.stack(stage_vecs, axis=0), axis=0).reshape(-1)
                    break

        # (c) GRU 상태 수집
        for k in ("gru_states","gru_hidden","gru_h_list","h_list","hidden_states"):
            if k in hooks_data and hooks_data[k] is not None:
                gru_list = hooks_data[k] if isinstance(hooks_data[k], list) else [hooks_data[k]]
                break

    # (d) 완전 폴백: 모델 파라미터에서 채널 벡터 구성(정적에도 값이 나오도록)
    if gates_NLG is None and gate_channel_vector is None and hasattr(model, "xhconv_blks"):
        chans = []
        for blk in getattr(model, "xhconv_blks", []):
            for name in ("alpha","cross_gate","gate","w","g"):
                if hasattr(blk, name):
                    v = _as_np(getattr(blk, name))
                    if v is None:
                        continue
                    v = np.asarray(v, float)
                    if v.ndim >= 2:
                        chans.append(v.reshape(-1))  # (2,2) 등은 4채널로 간주
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
        # (N,L,G) 보장
        if gates_NLG.ndim != 3:
            # 중간축을 L로 펴서 3차원으로 표준화
            gates_NLG = _to_NLG(gates_NLG)
        N, L, G = gates_NLG.shape

        # 시간/샘플 변동성
        #  - 시간 변동성: L축 표준편차의 평균
        #  - 샘플 변동성: N축 표준편차의 평균
        try:
            m["w2_gate_variability_time"]   = float(np.nanmean(np.nanstd(gates_NLG, axis=1)))  # (N,G) -> mean
        except Exception:
            # 보수적 폴백(샘플별 flatten 표준편차 평균)
            m["w2_gate_variability_time"] = float(np.nanmean(np.nanstd(gates_NLG.reshape(N, -1), axis=1)))
        m["w2_gate_variability_sample"] = float(np.nanmean(np.nanstd(gates_NLG, axis=0)))      # (L,G) -> mean

        # 엔트로피(양수영역 정규화)
        flat = gates_NLG.reshape(-1)
        pos  = flat[np.isfinite(flat) & (flat > 1e-8)]
        if pos.size > 0:
            p = pos / (pos.sum() + 1e-12)
            m["w2_gate_entropy"] = float(-np.sum(p * np.log(p + 1e-12)))

        # TOD 정렬 (거리상관)
        if isinstance(tod_vec, np.ndarray) and tod_vec.ndim == 2 and tod_vec.shape[0] > 2:
            g_strength = np.nanmean(gates_NLG, axis=(1, 2))  # (N,)
            gx, ty = _align_len(g_strength.reshape(-1), tod_vec.reshape(len(tod_vec), -1))
            if gx.size >= 3:
                try:
                    m["w2_gate_tod_alignment"] = float(_dcor_u(gx.reshape(-1,1), ty))
                except Exception:
                    m["w2_gate_tod_alignment"] = float("nan")

        # GRU 정렬 (안전한 r^2)
        if gru_list:
            try:
                gru = np.concatenate([_as_np(a) for a in gru_list if _as_np(a) is not None], axis=0)
                if gru.ndim == 1:
                    gn = np.abs(gru).reshape(-1)
                else:
                    # (N, d) 또는 (N,*,d) → (N,-1)로 평탄화 후 L2 norm
                    if gru.ndim > 2:
                        gru = gru.reshape(gru.shape[0], -1)
                    gn = np.linalg.norm(gru, axis=1)  # (N,)
                g_strength = np.nanmean(gates_NLG, axis=(1, 2))  # (N,)
                gx, hy = _align_len(g_strength.reshape(-1), gn.reshape(-1))
                m["w2_gate_gru_state_alignment"] = _pearson_r2_safe(gx, hy)
            except Exception:
                # 훅이 없거나 길이 불일치 등 → NaN 유지
                pass

        # 채널 선택도(동적): (G,) = mean(N,L)
        gate_channel_vector = np.nanmean(gates_NLG, axis=(0, 1))  # (G,)

    else:
        # 정적: TOD/GRU 정렬은 정의상 0(시간/샘플 변화가 없으므로)
        m["w2_gate_tod_alignment"] = 0.0
        m["w2_gate_gru_state_alignment"] = 0.0
        # 변동성 지표는 패키징 루트(분포 요약)에서 병합되므로 여기서는 건드리지 않음

    # 채널 선택도(첨도/희소성): 동적 또는 폴백 벡터가 있으면 항상 채움
    if gate_channel_vector is not None and gate_channel_vector.size >= 2:
        m["w2_channel_selectivity_kurtosis"] = _safe_kurtosis(gate_channel_vector)
        m["w2_channel_selectivity_sparsity"] = _hoyer_sparsity(gate_channel_vector)

    # 이벤트 조건 반응(동적일 때만 의미)
    cg_event_gain = None
    if isinstance(direct_evidence, dict):
        cg_event_gain = direct_evidence.get("cg_event_gain", None)
    if is_dynamic and isinstance(cg_event_gain, (int, float)) and np.isfinite(cg_event_gain):
        gv = m.get("w2_gate_variability_time", np.nan)
        if np.isfinite(gv):
            m["w2_event_conditional_response"] = float(gv * float(cg_event_gain))

    return m
