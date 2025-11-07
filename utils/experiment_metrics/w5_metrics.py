"""
W5 실험 특화 지표: 게이트 고정 효과
"""

import numpy as np
from typing import Dict, Optional


def compute_w5_metrics(
    model,
    fixed_model_metrics: Optional[Dict] = None,
    dynamic_model_metrics: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    W5 실험 특화 지표 계산
    
    Args:
        fixed_model_metrics: 게이트 고정 모델의 지표
        dynamic_model_metrics: 동적 게이트 모델의 지표
    
    Returns:
        dict with keys:
        - w5_performance_degradation_ratio: 성능 저하율
        - w5_sensitivity_gain_loss: 민감도 이득 손실
        - w5_event_gain_loss: 이벤트 이득 손실
        - w5_gate_event_alignment_loss: 게이트-이벤트 정렬 손실
    """
    metrics = {}
    
    if dynamic_model_metrics is None or fixed_model_metrics is None:
        metrics["w5_performance_degradation_ratio"] = np.nan
        metrics["w5_sensitivity_gain_loss"] = np.nan
        metrics["w5_event_gain_loss"] = np.nan
        metrics["w5_gate_event_alignment_loss"] = np.nan
        return metrics
    
    # 성능 저하율
    rmse_dynamic = dynamic_model_metrics.get("rmse", np.nan)
    rmse_fixed = fixed_model_metrics.get("rmse", np.nan)
    
    if np.isfinite(rmse_dynamic) and np.isfinite(rmse_fixed) and rmse_dynamic > 0:
        degradation = (rmse_fixed - rmse_dynamic) / rmse_dynamic
        metrics["w5_performance_degradation_ratio"] = float(degradation)
    else:
        metrics["w5_performance_degradation_ratio"] = np.nan
    
    # 민감도 이득 손실
    tod_dynamic = dynamic_model_metrics.get("gc_kernel_tod_dcor", np.nan)
    tod_fixed = fixed_model_metrics.get("gc_kernel_tod_dcor", np.nan)
    
    if np.isfinite(tod_dynamic) and np.isfinite(tod_fixed):
        loss = tod_dynamic - tod_fixed
        metrics["w5_sensitivity_gain_loss"] = float(loss)
    else:
        metrics["w5_sensitivity_gain_loss"] = np.nan
    
    # 이벤트 이득 손실
    event_dynamic = dynamic_model_metrics.get("cg_event_gain", np.nan)
    event_fixed = fixed_model_metrics.get("cg_event_gain", np.nan)
    
    if np.isfinite(event_dynamic) and np.isfinite(event_fixed):
        loss = event_dynamic - event_fixed
        metrics["w5_event_gain_loss"] = float(loss)
    else:
        metrics["w5_event_gain_loss"] = np.nan
    
    # 게이트-이벤트 정렬 손실
    # 게이트 값과 이벤트(피크) 발생 시점의 정렬 정도
    # dynamic에서는 게이트가 이벤트에 반응하지만, fixed에서는 반응하지 않음
    event_dynamic = dynamic_model_metrics.get("cg_event_gain", np.nan)
    event_fixed = fixed_model_metrics.get("cg_event_gain", np.nan)
    
    # 게이트 변동성 (W2 지표에서 계산된 값 사용 가능)
    gate_var_dynamic = dynamic_model_metrics.get("w2_gate_variability_time", np.nan)
    gate_var_fixed = fixed_model_metrics.get("w2_gate_variability_time", np.nan)
    
    # 게이트-이벤트 정렬 = 게이트 변동성과 이벤트 게인의 상관 관계
    # dynamic에서는 게이트가 이벤트에 반응하므로 정렬이 높고,
    # fixed에서는 게이트가 고정되어 정렬이 낮음
    if (np.isfinite(event_dynamic) and np.isfinite(event_fixed) and
        np.isfinite(gate_var_dynamic) and np.isfinite(gate_var_fixed)):
        # 정렬 손실 = (dynamic의 정렬) - (fixed의 정렬)
        # 간단히: 이벤트 게인과 게이트 변동성의 곱으로 근사
        alignment_dynamic = event_dynamic * gate_var_dynamic if gate_var_dynamic > 0 else 0
        alignment_fixed = event_fixed * gate_var_fixed if gate_var_fixed > 0 else 0
        loss = alignment_dynamic - alignment_fixed
        metrics["w5_gate_event_alignment_loss"] = float(loss)
    else:
        # 대안: 이벤트 게인 차이로 근사
        if np.isfinite(event_dynamic) and np.isfinite(event_fixed):
            metrics["w5_gate_event_alignment_loss"] = float(event_dynamic - event_fixed)
        else:
            metrics["w5_gate_event_alignment_loss"] = np.nan
    
    return metrics
