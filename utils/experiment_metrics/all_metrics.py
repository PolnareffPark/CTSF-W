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
    if experiment_type == "W1":
        return compute_w1_metrics(model, hooks_data, kwargs.get("tod_vec"))
    if experiment_type == "W2":
        return compute_w2_metrics(
            model=model,
            hooks_data=hooks_data,
            tod_vec=kwargs.get("tod_vec"),
            direct_evidence=kwargs.get("direct_evidence")
        )
    if experiment_type == "W3":
        # per-run 단계에서는 Δ를 계산하지 않음(집계 단계 전담)
        # baseline/perturb 한 쌍을 확보해 호출하고 싶다면 아래 주석을 활성화
        # return compute_w3_metrics(
        #     baseline_metrics=kwargs.get("baseline_metrics"),
        #     perturb_metrics=kwargs.get("perturb_metrics"),
        #     win_errors_base=kwargs.get("win_errors_base"),
        #     win_errors_pert=kwargs.get("win_errors_pert"),
        #     dz_ci=kwargs.get("w3_dz_ci", False)
        # )
        return {}
    if experiment_type == "W4":
        return compute_w4_metrics(model, hooks_data, kwargs.get("active_layers", []))
    if experiment_type == "W5":
        return compute_w5_metrics(model, kwargs.get("fixed_model_metrics"), kwargs.get("dynamic_model_metrics"))
    return {}