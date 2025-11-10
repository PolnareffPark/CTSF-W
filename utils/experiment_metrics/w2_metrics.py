"""
W2 실험 특화 지표: 동적 vs 정적 교차
"""

import torch
import numpy as np
from typing import Dict, Optional
from utils.metrics import _dcor_u, _multioutput_r2
from scipy import stats

def _safe_pearson_r2(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if x.size < 2 or y.size < 2:
        return np.nan
    sx, sy = np.std(x, ddof=0), np.std(y, ddof=0)
    if not np.isfinite(sx) or not np.isfinite(sy) or sx == 0 or sy == 0:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return float(r*r) if np.isfinite(r) else np.nan

def compute_w2_metrics(
    model,
    hooks_data: Optional[Dict] = None,
    tod_vec: Optional[np.ndarray] = None,
    direct_evidence: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    W2 실험 특화 지표 계산
    
    Args:
        model: 모델
        hooks_data: 훅 데이터 (gate_outputs, gru_states 등)
        tod_vec: (N_samples, 2) TOD 벡터 [sin, cos]
        direct_evidence: 직접 근거 지표 (cg_event_gain, E_list 등)
    
    Returns:
        dict with keys:
        - w2_gate_variability_time: 시간별 게이트 변동성
        - w2_gate_variability_sample: 샘플별 게이트 변동성
        - w2_gate_entropy: 게이트 엔트로피
        - w2_gate_tod_alignment: 게이트-TOD 정렬 (거리상관)
        - w2_gate_gru_state_alignment: 게이트-GRU 상태 정렬 (상관계수)
        - w2_event_conditional_response: 이벤트 조건 반응
        - w2_channel_selectivity_kurtosis: 채널 선택도 (첨도)
        - w2_channel_selectivity_sparsity: 채널 선택도 (희소성)
    """
    metrics = {}
    
    # 동적 게이트 출력 수집
    # hooks_data에서 실제 게이트 출력을 받아와야 함
    # 각 배치의 게이트 값: List[np.ndarray] 형태, 각 원소는 (B, n_layers, 4) 등
    gate_outputs_list = []
    gru_states_list = []
    gate_per_sample = []  # 샘플별 게이트 평균
    
    if hooks_data is not None:
        # 동적 게이트 출력 (각 배치의 게이트 값)
        gate_outputs = hooks_data.get("gate_outputs")  # List of (B, n_layers, 4) arrays
        if gate_outputs is not None and len(gate_outputs) > 0:
            gate_outputs_list = gate_outputs
            # 모든 샘플을 하나로 합침
            gate_all = np.concatenate(gate_outputs, axis=0)  # (N, n_layers, 4)
            gate_per_sample = gate_all
        
        # GRU 상태 (각 배치의 GRU hidden state)
        gru_states = hooks_data.get("gru_states")  # List of (B, d) arrays
        if gru_states is not None and len(gru_states) > 0:
            gru_states_list = gru_states
    
    # 동적 게이트 출력이 없으면 학습된 alpha 파라미터로 대체
    if len(gate_outputs_list) == 0:
        # 정적 모드이거나 동적 게이트 출력을 수집하지 않은 경우
        gate_values_static = []
        for blk in model.xhconv_blks:
            if hasattr(blk, 'alpha'):
                alpha_val = torch.relu(blk.alpha).detach().cpu().numpy()  # (2, 2)
                gate_values_static.append(alpha_val.flatten())
        
        if len(gate_values_static) > 0:
            gate_array_static = np.array(gate_values_static)  # (n_layers, 4)
            # 정적 게이트는 모든 샘플에 대해 동일
            # 최소한의 변동성 분석을 위해 층별로만 분석
            gate_per_sample = None
            gate_values_for_metrics = gate_array_static
        else:
            gate_per_sample = None
            gate_values_for_metrics = None
    else:
        # 동적 게이트 출력을 사용
        # gate_per_sample: (N, n_layers, 4)
        gate_values_for_metrics = gate_per_sample
    
    if gate_values_for_metrics is not None:
        # gate_values_for_metrics: (N, n_layers, 4) 또는 (n_layers, 4)
        is_dynamic = len(gate_values_for_metrics.shape) == 3  # 동적 게이트 여부
        
        if is_dynamic:
            # 동적 게이트: (N, n_layers, 4)
            N, n_layers, n_gates = gate_values_for_metrics.shape
            
            # 1. 시간별 게이트 변동성: 각 샘플 내에서 층별로 게이트가 얼마나 변하는지
            # 각 샘플(N개)에 대해 (n_layers, 4) 값들의 표준편차를 구하고 평균
            # shape: (N, n_layers*4) -> std along axis=1 -> (N,) -> mean
            gate_reshaped = gate_values_for_metrics.reshape(N, -1)  # (N, n_layers*4)
            gate_std_per_sample = np.nanstd(gate_reshaped, axis=1)  # (N,)
            metrics["w2_gate_variability_time"] = float(np.nanmean(gate_std_per_sample))
            
            # 2. 샘플별 게이트 변동성: 동일 층/게이트에서 샘플 간 분산
            # 각 (layer, gate) 조합에 대해 샘플 간(N개) 표준편차를 구하고 평균
            gate_std_per_gate = np.nanstd(gate_values_for_metrics, axis=0)  # (n_layers, 4)
            metrics["w2_gate_variability_sample"] = float(np.nanmean(gate_std_per_gate))
            
            # 평균 게이트 (모든 샘플과 층에 걸쳐)
            gate_flat = gate_values_for_metrics.reshape(-1)  # (N*n_layers*4,)
            gate_mean_per_sample = gate_values_for_metrics.mean(axis=(1, 2))  # (N,)
        else:
            # 정적 게이트: (n_layers, 4)
            # 층별 변동성만 측정 가능
            gate_std_layer = np.nanstd(gate_values_for_metrics)
            metrics["w2_gate_variability_time"] = float(gate_std_layer)
            metrics["w2_gate_variability_sample"] = float(np.nanstd(gate_values_for_metrics, axis=0).mean())
            
            gate_flat = gate_values_for_metrics.flatten()
            gate_mean_per_sample = None
        
        # 3. 게이트 엔트로피 (정규화 후)
        gate_flat_pos = gate_flat[gate_flat > 1e-8]  # 양수만
        if len(gate_flat_pos) > 0:
            gate_norm = gate_flat_pos / (gate_flat_pos.sum() + 1e-12)
            entropy = -np.sum(gate_norm * np.log(gate_norm + 1e-12))
            metrics["w2_gate_entropy"] = float(entropy)
        else:
            metrics["w2_gate_entropy"] = np.nan
        
        # 4. 게이트-TOD 정렬
        # TOD 벡터와 게이트 출력 간의 거리상관 계산
        if tod_vec is not None and is_dynamic:
            # 동적 게이트: 각 샘플의 평균 게이트 값과 TOD 비교
            gate_per_sample_mean = gate_values_for_metrics.mean(axis=(1, 2))  # (N,)
            # gate를 (N, 1)로, TOD는 (N, 2)
            if tod_vec.shape[0] == len(gate_per_sample_mean):
                gate_for_tod = gate_per_sample_mean.reshape(-1, 1)
                try:
                    metrics["w2_gate_tod_alignment"] = _dcor_u(gate_for_tod, tod_vec)
                except:
                    metrics["w2_gate_tod_alignment"] = np.nan
            else:
                metrics["w2_gate_tod_alignment"] = np.nan
        else:
            metrics["w2_gate_tod_alignment"] = np.nan
        
        # 5. 게이트-GRU 상태 정렬
        # GRU 상태의 norm과 게이트 강도 간의 상관
        if len(gru_states_list) > 0 and is_dynamic:
            # GRU 상태를 하나로 합침
            gru_concat = np.concatenate(gru_states_list, axis=0)  # (N, d)
            # GRU 상태의 L2 norm을 스칼라 요약값으로 사용
            gru_norm = np.linalg.norm(gru_concat, axis=1)  # (N,)
            
            # 각 샘플의 평균 게이트 값
            gate_per_sample_mean = gate_values_for_metrics.mean(axis=(1, 2))  # (N,)
            
            if len(gru_norm) == len(gate_per_sample_mean):
                # 피어슨 상관계수 계산
                try:
                    metrics["w2_gate_gru_state_alignment"] = _safe_pearson_r2(
                        gate_per_sample_mean, gru_norm
                    )
                except:
                    metrics["w2_gate_gru_state_alignment"] = np.nan
            else:
                metrics["w2_gate_gru_state_alignment"] = np.nan
        else:
            metrics["w2_gate_gru_state_alignment"] = np.nan
        
        # 6. 채널 선택도 (첨도)
        # 각 게이트 채널의 평균값에 대한 첨도
        if is_dynamic:
            gate_per_channel = gate_values_for_metrics.mean(axis=(0, 1))  # (4,)
        else:
            gate_per_channel = gate_values_for_metrics.mean(axis=0)  # (4,)
        
        if len(gate_per_channel) > 1:
            kurt = stats.kurtosis(gate_per_channel)
            metrics["w2_channel_selectivity_kurtosis"] = float(kurt) if np.isfinite(kurt) else np.nan
        else:
            metrics["w2_channel_selectivity_kurtosis"] = np.nan
        
        # 7. 채널 선택도 (희소성) - L1 norm / L2 norm
        gate_l1 = np.abs(gate_per_channel).sum()
        gate_l2 = np.sqrt((gate_per_channel ** 2).sum())
        if gate_l2 > 1e-12:
            sparsity = gate_l1 / (gate_l2 * np.sqrt(len(gate_per_channel)))
            metrics["w2_channel_selectivity_sparsity"] = float(sparsity)
        else:
            metrics["w2_channel_selectivity_sparsity"] = np.nan
    else:
        metrics["w2_gate_variability_time"] = np.nan
        metrics["w2_gate_variability_sample"] = np.nan
        metrics["w2_gate_entropy"] = np.nan
        metrics["w2_gate_tod_alignment"] = np.nan
        metrics["w2_gate_gru_state_alignment"] = np.nan
        metrics["w2_channel_selectivity_kurtosis"] = np.nan
        metrics["w2_channel_selectivity_sparsity"] = np.nan
    
    # 8. 이벤트 조건 반응
    # Conv 변화량이 큰 이벤트에서 게이트가 얼마나 반응하는지 측정
    if direct_evidence is not None and gate_values_for_metrics is not None:
        cg_event_gain = direct_evidence.get("cg_event_gain", np.nan)
        
        # direct_evidence에서 Conv 변화량 (E_list)를 사용할 수 있으면 더 정확
        # 여기서는 cg_event_gain을 활용하여 간접적으로 측정
        if is_dynamic and not np.isnan(cg_event_gain):
            # 이벤트 게인이 높으면 Conv→GRU 경로에서 이벤트 반응이 큼
            # 게이트 변동성이 높으면 이벤트에 반응한 것으로 간주
            gate_variability = metrics["w2_gate_variability_time"]
            if not np.isnan(gate_variability):
                # 정규화된 반응도: 게이트 변동성 * 이벤트 게인
                event_response = gate_variability * cg_event_gain
                metrics["w2_event_conditional_response"] = float(event_response)
            else:
                metrics["w2_event_conditional_response"] = np.nan
        else:
            # 정적 게이트이거나 이벤트 게인 정보가 없으면 NaN
            metrics["w2_event_conditional_response"] = np.nan
    else:
        metrics["w2_event_conditional_response"] = np.nan
    
    return metrics
