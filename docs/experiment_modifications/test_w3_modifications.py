"""
W3 수정 사항 간단 테스트 스크립트
baseline_metrics 전달 및 W3 지표 계산 확인
"""

import numpy as np
from utils.experiment_metrics.w3_metrics import compute_w3_metrics


def test_w3_metrics_with_baseline():
    """baseline_metrics 전달 시 W3 지표 계산 테스트"""
    print("=" * 60)
    print("W3 수정 사항 테스트")
    print("=" * 60)
    
    # Baseline 지표 (교란 없음)
    baseline_metrics = {
        "rmse": 0.3456,
        "rmse_std": 0.0234,  # 새로 추가된 필드
        "gc_kernel_tod_dcor": 0.45,
        "cg_event_gain": 0.67,
        "cg_bestlag": 2.3,
    }
    
    # 교란 실험 지표
    hooks_data = {
        "rmse": 0.4123,  # baseline보다 높음 (성능 저하)
        "gc_kernel_tod_dcor": 0.38,  # baseline보다 낮음
        "cg_event_gain": 0.54,  # baseline보다 낮음
        "cg_bestlag": 2.8,  # baseline보다 높음 (lag 분포 변화)
    }
    
    # 1. Baseline 자체 (perturbation="none")
    print("\n[Test 1] Baseline (perturbation='none')")
    baseline_result = compute_w3_metrics(
        model=None,
        hooks_data=baseline_metrics,
        perturbation_type="none",
        baseline_metrics=None
    )
    print(f"  intervention_effect_rmse: {baseline_result['w3_intervention_effect_rmse']:.4f} (expected: 0.0)")
    print(f"  intervention_cohens_d: {baseline_result['w3_intervention_cohens_d']:.4f} (expected: 0.0)")
    
    # 2. 교란 실험 (perturbation="tod_shift", baseline 전달)
    print("\n[Test 2] Perturbed (perturbation='tod_shift', with baseline)")
    perturbed_result = compute_w3_metrics(
        model=None,
        hooks_data=hooks_data,
        perturbation_type="tod_shift",
        baseline_metrics=baseline_metrics
    )
    
    expected_rmse_change = hooks_data["rmse"] - baseline_metrics["rmse"]
    expected_cohens_d = expected_rmse_change / baseline_metrics["rmse_std"]
    
    print(f"  intervention_effect_rmse: {perturbed_result['w3_intervention_effect_rmse']:.4f}")
    print(f"    (expected: {expected_rmse_change:.4f})")
    print(f"  intervention_effect_tod: {perturbed_result['w3_intervention_effect_tod']:.4f}")
    print(f"    (expected: {hooks_data['gc_kernel_tod_dcor'] - baseline_metrics['gc_kernel_tod_dcor']:.4f})")
    print(f"  intervention_effect_peak: {perturbed_result['w3_intervention_effect_peak']:.4f}")
    print(f"    (expected: {hooks_data['cg_event_gain'] - baseline_metrics['cg_event_gain']:.4f})")
    print(f"  intervention_cohens_d: {perturbed_result['w3_intervention_cohens_d']:.4f}")
    print(f"    (expected: {expected_cohens_d:.4f})")
    
    # 3. 교란 실험이지만 baseline 없음 (문제 상황 - 이제는 경고 가능)
    print("\n[Test 3] Perturbed without baseline (problematic case)")
    no_baseline_result = compute_w3_metrics(
        model=None,
        hooks_data=hooks_data,
        perturbation_type="tod_shift",
        baseline_metrics=None  # 문제: baseline 없음
    )
    print(f"  intervention_effect_rmse: {no_baseline_result['w3_intervention_effect_rmse']:.4f} (expected: 0.0 - problematic!)")
    print(f"  ⚠️  이 경우 W3 지표가 모두 0으로 계산됨 (이제는 W3Experiment에서 자동으로 baseline 생성)")
    
    print("\n" + "=" * 60)
    print("✅ 테스트 완료!")
    print("=" * 60)
    
    # 검증
    assert baseline_result['w3_intervention_effect_rmse'] == 0.0, "Baseline 효과는 0이어야 함"
    assert abs(perturbed_result['w3_intervention_effect_rmse'] - expected_rmse_change) < 1e-6, "RMSE 변화량 계산 오류"
    assert abs(perturbed_result['w3_intervention_cohens_d'] - expected_cohens_d) < 1e-6, "Cohen's d 계산 오류"
    print("\n✅ 모든 검증 통과!")


def test_cohens_d_with_zero_std():
    """표준편차가 0일 때 Cohen's d 대안 계산 테스트"""
    print("\n" + "=" * 60)
    print("Cohen's d 대안 계산 테스트 (표준편차가 0인 경우)")
    print("=" * 60)
    
    baseline_metrics = {
        "rmse": 0.3456,
        "rmse_std": 0.0,  # 표준편차가 0
        "gc_kernel_tod_dcor": 0.45,
        "cg_event_gain": 0.67,
        "cg_bestlag": 2.3,
    }
    
    hooks_data = {
        "rmse": 0.4123,
        "gc_kernel_tod_dcor": 0.38,
        "cg_event_gain": 0.54,
        "cg_bestlag": 2.8,
    }
    
    result = compute_w3_metrics(
        model=None,
        hooks_data=hooks_data,
        perturbation_type="tod_shift",
        baseline_metrics=baseline_metrics
    )
    
    # 표준편차가 0이면 상대적 효과 크기로 계산
    expected_relative_effect = (hooks_data["rmse"] - baseline_metrics["rmse"]) / baseline_metrics["rmse"]
    
    print(f"  intervention_cohens_d: {result['w3_intervention_cohens_d']:.4f}")
    print(f"    (expected relative effect: {expected_relative_effect:.4f})")
    print(f"    (상대적 변화: {expected_relative_effect * 100:.2f}%)")
    
    assert abs(result['w3_intervention_cohens_d'] - expected_relative_effect) < 1e-6, "대안 Cohen's d 계산 오류"
    print("\n✅ 대안 계산 검증 통과!")


if __name__ == "__main__":
    test_w3_metrics_with_baseline()
    test_cohens_d_with_zero_std()
    print("\n" + "=" * 60)
    print("🎉 모든 테스트 통과! W3 수정 사항이 정상 작동합니다.")
    print("=" * 60)

