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
    # 간단히 이벤트 게인과 게이트 변동성의 상관으로 근사
    # TODO: 실제 게이트-이벤트 정렬 계산 필요
    metrics["w5_gate_event_alignment_loss"] = np.nan
    
    return metrics
