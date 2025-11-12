"""
전체 실험 지표 통합 모듈
W1~W5 모든 실험의 지표를 계산하는 통합 함수
"""

from typing import Dict, Optional
from .w1_metrics import compute_w1_metrics
from .w2_metrics import compute_w2_metrics
from .w3_metrics import compute_w3_metrics   # 존재만 보장(사후 Δ계산용)
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

    주의:
      - W3는 per-run에서는 Δ를 계산하지 않음(교란쌍 매칭 필요). 여기서는 빈 dict 반환.
        Δ는 plotting 단계의 집계(rebuild_w3_perturb_delta_bar / rebuild_w3_forest_summary)에서 계산.
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
        return {}  # per-run에서 Δ 미계산
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
