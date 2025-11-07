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
        return compute_w2_metrics(
            model=model,
            hooks_data=hooks_data,
            tod_vec=kwargs.get("tod_vec"),
            direct_evidence=kwargs.get("direct_evidence")
        )
    elif experiment_type == "W3":
        # W3는 baseline과 perturbed 결과를 비교하므로 새로운 인터페이스 사용
        baseline_metrics = kwargs.get("baseline_metrics")
        perturb_metrics = kwargs.get("perturb_metrics")
        
        # perturbation이 "none"이면 빈 딕셔너리 반환 (baseline 자체)
        if kwargs.get("perturbation_type") == "none" or baseline_metrics is None:
            return {}
        
        # 윈도우별 오차 벡터
        win_errors_base = baseline_metrics.pop("win_errors", None) if baseline_metrics else None
        win_errors_pert = perturb_metrics.pop("win_errors", None) if perturb_metrics else None
        
        return compute_w3_metrics(
            baseline_metrics=baseline_metrics,
            perturb_metrics=perturb_metrics,
            win_errors_base=win_errors_base,
            win_errors_pert=win_errors_pert,
            dz_ci=kwargs.get("w3_dz_ci", False)
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
