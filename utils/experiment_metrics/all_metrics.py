"""
전체 실험 지표 통합 모듈 (W1~W5)
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
    실험 타입에 따라 특화 지표 계산.
    W3의 기본 Δ 집계는 그림 재작성(rebuild_*) 단계에서 수행.
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
        base = kwargs.get("baseline_metrics")
        pert = kwargs.get("perturb_metrics")
        if isinstance(base, dict) and isinstance(pert, dict):
            return compute_w3_metrics(
                baseline_metrics=base,
                perturb_metrics=pert,
                win_errors_base=kwargs.get("win_errors_base"),
                win_errors_pert=kwargs.get("win_errors_pert"),
                dz_ci=kwargs.get("w3_dz_ci", False),
            )
        # 단독 실행 시에는 페어 없음 → 빈 dict
        return {}
    if et == "W4":
        return compute_w4_metrics(model, hooks_data, kwargs.get("active_layers", []))
    if et == "W5":
        return compute_w5_metrics(
            model,
            kwargs.get("fixed_model_metrics"),
            kwargs.get("dynamic_model_metrics")
        )
    return {}
