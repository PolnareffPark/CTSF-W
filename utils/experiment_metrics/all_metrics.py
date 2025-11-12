# utils/experiment_metrics/all_metrics.py
"""
전체 실험 지표 통합 모듈
W1~W5 모든 실험의 지표를 계산하는 통합 함수
"""
from __future__ import annotations
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
        experiment_type: 'W1','W2','W3','W4','W5'
        model: 모델 인스턴스
        hooks_data: 훅 데이터
        **kwargs: 실험별 추가 파라미터
    Returns:
        dict
    """
    et = (experiment_type or "").upper()

    if et == "W1":
        return compute_w1_metrics(model, hooks_data, kwargs.get("tod_vec"))

    if et == "W2":
        return compute_w2_metrics(
            model=model,
            hooks_data=hooks_data,
            tod_vec=kwargs.get("tod_vec"),
            direct_evidence=kwargs.get("direct_evidence"),
        )

    if et == "W3":
        # baseline/perturb 쌍이 둘 다 있어야 함
        base = kwargs.get("baseline_metrics")
        pert = kwargs.get("perturb_metrics")
        if kwargs.get("perturbation_type") in (None, "none") or base is None or pert is None:
            return {}
        return compute_w3_metrics(
            baseline_metrics=base,
            perturb_metrics=pert,
            win_errors_base=kwargs.get("win_errors_base"),
            win_errors_pert=kwargs.get("win_errors_pert"),
            dz_ci=kwargs.get("w3_dz_ci", True),
        )

    if et == "W4":
        return compute_w4_metrics(model, hooks_data, kwargs.get("active_layers", []))

    if et == "W5":
        return compute_w5_metrics(model, kwargs.get("fixed_model_metrics"), kwargs.get("dynamic_model_metrics"))

    return {}
