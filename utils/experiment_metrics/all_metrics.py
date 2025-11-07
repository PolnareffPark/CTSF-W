"""
전체 실험 지표 통합 모듈
W1~W5 모든 실험의 지표를 계산하는 통합 함수
"""

from typing import Dict, Optional
from .w1_metrics import compute_w1_metrics
from .w2_metrics import compute_w2_metrics
from .w3_metrics import compute_w3_metrics
from .w4_metrics import compute_w4_metrics
from .w5_metrics import compute_w5_metrics


def compute_all_experiment_metrics(
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
        return compute_w4_metrics(
            model, hooks_data,
            kwargs.get("active_layers", [])
        )
    elif experiment_type == "W5":
        return compute_w5_metrics(
            model,
            kwargs.get("fixed_model_metrics"),
            kwargs.get("dynamic_model_metrics")
        )
    else:
        return {}
