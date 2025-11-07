"""
W3 실험 특화 지표: 데이터 교란 효과
"""

import numpy as np
from typing import Dict, Optional


def _cohens_d(x1: np.ndarray, x2: np.ndarray) -> float:
    """Cohen's d 효과 크기 계산"""
    if len(x1) < 2 or len(x2) < 2:
        return np.nan
    
    m1, m2 = np.mean(x1), np.mean(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    n1, n2 = len(x1), len(x2)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    
    if pooled_std < 1e-12:
        return np.nan
    
    d = (m1 - m2) / pooled_std
    return float(d) if np.isfinite(d) else np.nan


def compute_w3_metrics(
    model,
    hooks_data: Optional[Dict] = None,
    perturbation_type: str = "none",
    baseline_metrics: Optional[Dict] = None,
    **kwargs
) -> Dict:
    """
    W3 실험 특화 지표 계산
    
    Args:
        perturbation_type: 'none', 'tod_shift', 'smooth'
        baseline_metrics: 교란 없음('none') 모델의 지표 (비교용)
    
    Returns:
        dict with keys:
        - w3_intervention_effect_rmse: 개입 효과 (ΔRMSE)
        - w3_intervention_effect_tod: 개입 효과 (ΔTOD)
        - w3_intervention_effect_peak: 개입 효과 (Δpeak)
        - w3_intervention_cohens_d: Cohen's d (효과 크기)
        - w3_rank_preservation_rate: 순위 보존률
        - w3_lag_distribution_change: 라그 분포 변화
    """
    metrics = {}
    
    if perturbation_type == "none" or baseline_metrics is None:
        # 기준선이므로 개입 효과는 0
        metrics["w3_intervention_effect_rmse"] = 0.0
        metrics["w3_intervention_effect_tod"] = 0.0
        metrics["w3_intervention_effect_peak"] = 0.0
        metrics["w3_intervention_cohens_d"] = 0.0
    else:
        # baseline과 비교하여 개입 효과 계산
        current_rmse = hooks_data.get("rmse", np.nan) if hooks_data else np.nan
        baseline_rmse = baseline_metrics.get("rmse", np.nan)
        
        if np.isfinite(current_rmse) and np.isfinite(baseline_rmse):
            metrics["w3_intervention_effect_rmse"] = float(current_rmse - baseline_rmse)
        else:
            metrics["w3_intervention_effect_rmse"] = np.nan
        
        # TOD 민감도 차이
        current_tod = hooks_data.get("gc_kernel_tod_dcor", np.nan) if hooks_data else np.nan
        baseline_tod = baseline_metrics.get("gc_kernel_tod_dcor", np.nan)
        
        if np.isfinite(current_tod) and np.isfinite(baseline_tod):
            metrics["w3_intervention_effect_tod"] = float(current_tod - baseline_tod)
        else:
            metrics["w3_intervention_effect_tod"] = np.nan
        
        # 피크 반응 차이
        current_peak = hooks_data.get("cg_event_gain", np.nan) if hooks_data else np.nan
        baseline_peak = baseline_metrics.get("cg_event_gain", np.nan)
        
        if np.isfinite(current_peak) and np.isfinite(baseline_peak):
            metrics["w3_intervention_effect_peak"] = float(current_peak - baseline_peak)
        else:
            metrics["w3_intervention_effect_peak"] = np.nan
        
        # Cohen's d (효과 크기)
        # 간단히 RMSE 차이를 표준화
        if np.isfinite(metrics["w3_intervention_effect_rmse"]):
            # 가정: baseline의 표준편차를 사용
            baseline_std = baseline_metrics.get("rmse_std", 1.0)
            if baseline_std > 1e-12:
                metrics["w3_intervention_cohens_d"] = float(metrics["w3_intervention_effect_rmse"] / baseline_std)
            else:
                metrics["w3_intervention_cohens_d"] = np.nan
        else:
            metrics["w3_intervention_cohens_d"] = np.nan
    
    # 순위 보존률 (TODO: 실제 구현 필요)
    metrics["w3_rank_preservation_rate"] = np.nan
    
    # 라그 분포 변화 (TODO: 실제 구현 필요)
    metrics["w3_lag_distribution_change"] = np.nan
    
    return metrics
