"""
실험별 특화 지표 계산 모듈
W1~W5 실험에 특화된 지표들을 계산
"""

import torch
import numpy as np
from typing import Dict, Optional
from utils.metrics import _safe_corr, _dcor_u, _pearson_corr, _spearman_corr


# ==================== W1 특화 지표 ====================
def compute_w1_metrics(model, hooks_data, tod_vec=None):
    """
    W1 실험 특화 지표: 층별 교차 vs 최종 결합
    
    Returns:
        dict with keys:
        - cka_similarity_cnn_gru: CKA 유사도 (CNN↔GRU)
        - cca_similarity_cnn_gru: CCA 유사도
        - layerwise_upward_improvement: 층별 상향 개선도
        - inter_path_gradient_align: 경로 간 그래디언트 정렬
    """
    metrics = {}
    
    # TODO: 실제 구현 필요
    # CKA/CCA 유사도 계산
    metrics["cka_similarity_cnn_gru"] = np.nan
    metrics["cca_similarity_cnn_gru"] = np.nan
    
    # 층별 상향 개선도
    metrics["layerwise_upward_improvement"] = np.nan
    
    # 그래디언트 정렬
    metrics["inter_path_gradient_align"] = np.nan
    
    return metrics


# ==================== W2 특화 지표 ====================
def compute_w2_metrics(model, hooks_data, tod_vec=None):
    """
    W2 실험 특화 지표: 동적 vs 정적 교차
    
    Returns:
        dict with keys:
        - gate_variability_time: 시간별 게이트 변동성
        - gate_variability_sample: 샘플별 게이트 변동성
        - gate_entropy: 게이트 엔트로피
        - gate_tod_alignment: 게이트-TOD 정렬 (R²)
        - gate_gru_state_alignment: 게이트-GRU 상태 정렬 (R²)
        - event_conditional_response: 이벤트 조건 반응
        - channel_selectivity_kurtosis: 채널 선택도 (첨도)
        - channel_selectivity_sparsity: 채널 선택도 (희소성)
    """
    metrics = {}
    
    # 게이트 값 수집 (alpha 파라미터 또는 실제 게이트 출력)
    gate_values = []
    for blk in model.xhconv_blks:
        if hasattr(blk, 'alpha'):
            # alpha는 학습된 파라미터, 실제 게이트는 forward에서 계산됨
            gate_values.append(blk.alpha.detach().cpu().numpy())
    
    if len(gate_values) > 0:
        # 게이트 변동성 계산
        gate_array = np.array(gate_values)
        metrics["gate_variability_time"] = float(np.nanstd(gate_array))
        metrics["gate_variability_sample"] = float(np.nanstd(gate_array))
        metrics["gate_entropy"] = np.nan  # TODO: 엔트로피 계산
        
        # 게이트-TOD 정렬
        if tod_vec is not None:
            # TODO: 게이트 값과 TOD 벡터 간 R² 계산
            metrics["gate_tod_alignment"] = np.nan
            metrics["gate_gru_state_alignment"] = np.nan
        else:
            metrics["gate_tod_alignment"] = np.nan
            metrics["gate_gru_state_alignment"] = np.nan
    else:
        metrics["gate_variability_time"] = np.nan
        metrics["gate_variability_sample"] = np.nan
        metrics["gate_entropy"] = np.nan
        metrics["gate_tod_alignment"] = np.nan
        metrics["gate_gru_state_alignment"] = np.nan
    
    # 이벤트 조건 반응
    metrics["event_conditional_response"] = np.nan
    
    # 채널 선택도
    metrics["channel_selectivity_kurtosis"] = np.nan
    metrics["channel_selectivity_sparsity"] = np.nan
    
    return metrics


# ==================== W3 특화 지표 ====================
def compute_w3_metrics(model, hooks_data, perturbation_type, baseline_metrics=None):
    """
    W3 실험 특화 지표: 데이터 교란 효과
    
    Args:
        perturbation_type: 'none', 'tod_shift', 'smooth'
        baseline_metrics: 교란 없음('none') 모델의 지표 (비교용)
    
    Returns:
        dict with keys:
        - intervention_effect_rmse: 개입 효과 (ΔRMSE)
        - intervention_effect_tod: 개입 효과 (ΔTOD)
        - intervention_effect_peak: 개입 효과 (Δpeak)
        - intervention_cohens_d: Cohen's d (효과 크기)
        - rank_preservation_rate: 순위 보존률
        - lag_distribution_change: 라그 분포 변화
    """
    metrics = {}
    
    if perturbation_type == "none" or baseline_metrics is None:
        # 기준선이므로 개입 효과는 0
        metrics["intervention_effect_rmse"] = 0.0
        metrics["intervention_effect_tod"] = 0.0
        metrics["intervention_effect_peak"] = 0.0
        metrics["intervention_cohens_d"] = 0.0
    else:
        # TODO: baseline과 비교하여 개입 효과 계산
        metrics["intervention_effect_rmse"] = np.nan
        metrics["intervention_effect_tod"] = np.nan
        metrics["intervention_effect_peak"] = np.nan
        metrics["intervention_cohens_d"] = np.nan
    
    metrics["rank_preservation_rate"] = np.nan
    metrics["lag_distribution_change"] = np.nan
    
    return metrics


# ==================== W4 특화 지표 ====================
def compute_w4_metrics(model, hooks_data, active_layers):
    """
    W4 실험 특화 지표: 층별 기여도 분석
    
    Args:
        active_layers: 활성화된 층 인덱스 리스트
    
    Returns:
        dict with keys:
        - layer_contribution_score: 층별 기여 점수 (NCS)
        - layerwise_gate_usage: 층별 게이트 사용률
        - layerwise_representation_similarity: 층별 표현 유사도 (CNN↔GRU CKA)
    """
    metrics = {}
    
    # 층별 기여 점수
    metrics["layer_contribution_score"] = np.nan
    
    # 층별 게이트 사용률
    gate_usage = []
    for i, blk in enumerate(model.xhconv_blks):
        if i in active_layers:
            # 활성화된 층의 게이트 사용률 계산
            if hasattr(blk, 'alpha'):
                alpha_val = torch.relu(blk.alpha).detach().cpu().numpy()
                usage = float(np.mean(alpha_val))
                gate_usage.append(usage)
    
    if len(gate_usage) > 0:
        metrics["layerwise_gate_usage"] = float(np.mean(gate_usage))
    else:
        metrics["layerwise_gate_usage"] = np.nan
    
    # 층별 표현 유사도
    metrics["layerwise_representation_similarity"] = np.nan
    
    return metrics


# ==================== W5 특화 지표 ====================
def compute_w5_metrics(model, fixed_model_metrics, dynamic_model_metrics):
    """
    W5 실험 특화 지표: 게이트 고정 효과
    
    Args:
        fixed_model_metrics: 게이트 고정 모델의 지표
        dynamic_model_metrics: 동적 게이트 모델의 지표
    
    Returns:
        dict with keys:
        - performance_degradation_ratio: 성능 저하율
        - sensitivity_gain_loss: 민감도 이득 손실
        - event_gain_loss: 이벤트 이득 손실
        - gate_event_alignment_loss: 게이트-이벤트 정렬 손실
    """
    metrics = {}
    
    if dynamic_model_metrics is None or fixed_model_metrics is None:
        metrics["performance_degradation_ratio"] = np.nan
        metrics["sensitivity_gain_loss"] = np.nan
        metrics["event_gain_loss"] = np.nan
        metrics["gate_event_alignment_loss"] = np.nan
        return metrics
    
    # 성능 저하율
    if "rmse" in dynamic_model_metrics and "rmse" in fixed_model_metrics:
        rmse_dynamic = dynamic_model_metrics["rmse"]
        rmse_fixed = fixed_model_metrics["rmse"]
        if rmse_dynamic > 0:
            metrics["performance_degradation_ratio"] = (rmse_fixed - rmse_dynamic) / rmse_dynamic
        else:
            metrics["performance_degradation_ratio"] = np.nan
    else:
        metrics["performance_degradation_ratio"] = np.nan
    
    # 민감도/이벤트 이득 손실
    metrics["sensitivity_gain_loss"] = np.nan
    metrics["event_gain_loss"] = np.nan
    metrics["gate_event_alignment_loss"] = np.nan
    
    return metrics


# ==================== 통합 함수 ====================
def compute_experiment_specific_metrics(
    experiment_type: str,
    model,
    hooks_data: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    실험 타입에 따라 적절한 특화 지표 계산
    
    Args:
        experiment_type: 'W1', 'W2', 'W3', 'W4', 'W5'
        model: 모델 인스턴스
        hooks_data: 훅에서 수집한 데이터
        **kwargs: 실험별 추가 파라미터
    
    Returns:
        실험별 특화 지표 딕셔너리
    """
    if hooks_data is None:
        hooks_data = {}
    
    if experiment_type == "W1":
        return compute_w1_metrics(model, hooks_data, kwargs.get("tod_vec"))
    elif experiment_type == "W2":
        return compute_w2_metrics(model, hooks_data, kwargs.get("tod_vec"))
    elif experiment_type == "W3":
        return compute_w3_metrics(
            model, hooks_data,
            kwargs.get("perturbation_type", "none"),
            kwargs.get("baseline_metrics")
        )
    elif experiment_type == "W4":
        return compute_w4_metrics(model, hooks_data, kwargs.get("active_layers", []))
    elif experiment_type == "W5":
        return compute_w5_metrics(
            model,
            kwargs.get("fixed_model_metrics"),
            kwargs.get("dynamic_model_metrics")
        )
    else:
        return {}
