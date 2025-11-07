"""
W3 실험 특화 지표: 데이터 교란 효과
"""

import numpy as np
from typing import Dict, Optional


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
        - w3_intervention_cohens_d: Cohen's d (효과 크기, 배치별 RMSE 표준편차 기준)
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
        # RMSE 차이를 baseline RMSE의 표준편차로 나눔
        if np.isfinite(metrics["w3_intervention_effect_rmse"]):
            # baseline의 표준편차를 사용
            baseline_std = baseline_metrics.get("rmse_std", 0.0)
            
            # 표준편차가 충분히 큰 경우에만 Cohen's d 계산
            if baseline_std > 1e-6:
                metrics["w3_intervention_cohens_d"] = float(metrics["w3_intervention_effect_rmse"] / baseline_std)
            else:
                # 표준편차가 0에 가까우면 대안: baseline RMSE의 비율로 계산
                # (표준편차가 없을 경우 상대적 변화량으로 효과 크기 근사)
                if baseline_rmse > 1e-6:
                    # 상대적 효과 크기 = (current - baseline) / baseline
                    metrics["w3_intervention_cohens_d"] = float(metrics["w3_intervention_effect_rmse"] / baseline_rmse)
                else:
                    metrics["w3_intervention_cohens_d"] = np.nan
        else:
            metrics["w3_intervention_cohens_d"] = np.nan
    
    return metrics
