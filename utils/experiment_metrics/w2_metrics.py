"""
W2 실험 특화 지표: 동적 vs 정적 교차
"""

import torch
import numpy as np
from typing import Dict, Optional
from utils.metrics import _dcor_u, _multioutput_r2
from scipy import stats


def compute_w2_metrics(
    model,
    hooks_data: Optional[Dict] = None,
    tod_vec: Optional[np.ndarray] = None,
    **kwargs
) -> Dict:
    """
    W2 실험 특화 지표 계산
    
    Returns:
        dict with keys:
        - w2_gate_variability_time: 시간별 게이트 변동성
        - w2_gate_variability_sample: 샘플별 게이트 변동성
        - w2_gate_entropy: 게이트 엔트로피
        - w2_gate_tod_alignment: 게이트-TOD 정렬 (R²)
        - w2_gate_gru_state_alignment: 게이트-GRU 상태 정렬 (R²)
        - w2_event_conditional_response: 이벤트 조건 반응
        - w2_channel_selectivity_kurtosis: 채널 선택도 (첨도)
        - w2_channel_selectivity_sparsity: 채널 선택도 (희소성)
    """
    metrics = {}
    
    # 게이트 값 수집 (alpha 파라미터 또는 실제 게이트 출력)
    gate_values_list = []
    gru_states_list = []
    
    for blk in model.xhconv_blks:
        if hasattr(blk, 'alpha'):
            # alpha는 학습된 파라미터, 실제 게이트는 forward에서 계산됨
            alpha_val = torch.relu(blk.alpha).detach().cpu().numpy()  # (2, 2)
            gate_values_list.append(alpha_val.flatten())
    
    # hooks_data에서 GRU 상태 추출
    if hooks_data is not None:
        gru_states = hooks_data.get("gru_states")  # List of (B, d) arrays
        if gru_states is not None and len(gru_states) > 0:
            gru_states_list = gru_states
    
    if len(gate_values_list) > 0:
        gate_array = np.array(gate_values_list)  # (n_layers, 4)
        
        # 게이트 변동성
        metrics["w2_gate_variability_time"] = float(np.nanstd(gate_array))
        metrics["w2_gate_variability_sample"] = float(np.nanstd(gate_array, axis=0).mean())
        
        # 게이트 엔트로피 (정규화 후)
        gate_flat = gate_array.flatten()
        gate_flat = gate_flat[gate_flat > 1e-8]  # 0 제거
        if len(gate_flat) > 0:
            gate_norm = gate_flat / (gate_flat.sum() + 1e-12)
            entropy = -np.sum(gate_norm * np.log(gate_norm + 1e-12))
            metrics["w2_gate_entropy"] = float(entropy)
        else:
            metrics["w2_gate_entropy"] = np.nan
        
        # 게이트-TOD 정렬
        if tod_vec is not None and len(gate_values_list) > 0:
            # 각 층의 게이트 값을 TOD와 비교
            gate_mean = gate_array.mean(axis=0)  # (4,)
            # TOD 벡터와의 상관 (간단히 평균 게이트와 TOD의 거리상관)
            if tod_vec.shape[0] >= len(gate_values_list):
                tod_subset = tod_vec[:len(gate_values_list), :]
                # 게이트를 (N, 4)로 확장
                gate_expanded = np.tile(gate_mean, (tod_subset.shape[0], 1))
                metrics["w2_gate_tod_alignment"] = _dcor_u(gate_expanded, tod_subset)
            else:
                metrics["w2_gate_tod_alignment"] = np.nan
        else:
            metrics["w2_gate_tod_alignment"] = np.nan
        
        # 게이트-GRU 상태 정렬
        if len(gru_states_list) > 0:
            gru_concat = np.concatenate(gru_states_list, axis=0)  # (N, d)
            if gru_concat.shape[0] >= len(gate_values_list):
                gate_expanded = np.tile(gate_mean, (gru_concat.shape[0], 1))
                # 차원 맞추기
                if gate_expanded.shape[1] == gru_concat.shape[1]:
                    metrics["w2_gate_gru_state_alignment"] = _dcor_u(gate_expanded, gru_concat)
                else:
                    metrics["w2_gate_gru_state_alignment"] = np.nan
            else:
                metrics["w2_gate_gru_state_alignment"] = np.nan
        else:
            metrics["w2_gate_gru_state_alignment"] = np.nan
        
        # 채널 선택도 (첨도)
        gate_per_channel = gate_array.mean(axis=0)  # (4,)
        if len(gate_per_channel) > 0:
            kurt = stats.kurtosis(gate_per_channel)
            metrics["w2_channel_selectivity_kurtosis"] = float(kurt) if np.isfinite(kurt) else np.nan
        else:
            metrics["w2_channel_selectivity_kurtosis"] = np.nan
        
        # 채널 선택도 (희소성) - L1 norm / L2 norm
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
    
    # 이벤트 조건 반응 (TODO: 실제 구현 필요)
    metrics["w2_event_conditional_response"] = np.nan
    
    return metrics
