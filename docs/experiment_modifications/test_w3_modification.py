"""
W3 수정 사항 테스트: Paired Cohen's d_z 및 새로운 인터페이스
"""

import sys
sys.path.insert(0, '/home/himchan/proj/CTSF/CTSF-W')

import numpy as np
from utils.experiment_metrics.w3_metrics import (
    compute_w3_metrics,
    paired_cohens_dz,
    bootstrap_ci,
    _safe_div
)


def test_safe_div():
    """안전한 나눗셈 테스트"""
    print("=" * 60)
    print("Test 1: _safe_div()")
    print("=" * 60)
    
    assert abs(_safe_div(10, 2) - 5.0) < 1e-9
    assert abs(_safe_div(10, 0) - 10 / 1e-12) < 1e-6  # 0으로 나누기 방지
    print("✅ _safe_div() 테스트 통과!\n")


def test_paired_cohens_dz():
    """Paired Cohen's d_z 테스트"""
    print("=" * 60)
    print("Test 2: paired_cohens_dz()")
    print("=" * 60)
    
    # 예시: 교란으로 오차가 일관되게 증가
    diff = np.array([0.05, 0.08, 0.06, 0.07, 0.09, 0.04, 0.06, 0.08])
    dz = paired_cohens_dz(diff)
    
    expected_dz = np.mean(diff) / np.std(diff, ddof=1)
    
    print(f"  차이 벡터: {diff}")
    print(f"  평균: {np.mean(diff):.4f}")
    print(f"  표준편차: {np.std(diff, ddof=1):.4f}")
    print(f"  Cohen's d_z: {dz:.4f}")
    print(f"  예상값: {expected_dz:.4f}")
    
    assert abs(dz - expected_dz) < 1e-9
    print("✅ paired_cohens_dz() 테스트 통과!\n")


def test_bootstrap_ci():
    """부트스트랩 신뢰구간 테스트"""
    print("=" * 60)
    print("Test 3: bootstrap_ci()")
    print("=" * 60)
    
    diff = np.array([0.05, 0.08, 0.06, 0.07, 0.09, 0.04, 0.06, 0.08])
    
    stat_fn = lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else np.nan
    lo, hi = bootstrap_ci(stat_fn, diff, n_boot=100, ci=95)
    
    print(f"  차이 벡터: {diff}")
    print(f"  95% CI: [{lo:.4f}, {hi:.4f}]")
    
    # CI가 실제 d_z 값을 포함하는지 확인
    dz = paired_cohens_dz(diff)
    print(f"  실제 d_z: {dz:.4f}")
    
    assert lo < dz < hi, "CI가 실제 값을 포함해야 함"
    print("✅ bootstrap_ci() 테스트 통과!\n")


def test_compute_w3_metrics_basic():
    """compute_w3_metrics() 기본 테스트"""
    print("=" * 60)
    print("Test 4: compute_w3_metrics() - 기본 기능")
    print("=" * 60)
    
    # Baseline 지표
    baseline_metrics = {
        "rmse": 0.3456,
        "gc_kernel_tod_dcor": 0.45,
        "cg_event_gain": 0.67,
    }
    
    # 교란 지표 (성능 악화)
    perturb_metrics = {
        "rmse": 0.4123,
        "gc_kernel_tod_dcor": 0.38,
        "cg_event_gain": 0.54,
    }
    
    # 윈도우별 오차 (N=10)
    np.random.seed(42)
    win_errors_base = np.random.uniform(0.08, 0.12, 10)
    win_errors_pert = win_errors_base + np.random.uniform(0.02, 0.05, 10)  # 일관되게 증가
    
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=win_errors_base,
        win_errors_pert=win_errors_pert,
        dz_ci=False
    )
    
    print("\n📊 결과:")
    print(f"  w3_rmse_base: {result['w3_rmse_base']:.4f}")
    print(f"  w3_rmse_perturb: {result['w3_rmse_perturb']:.4f}")
    print(f"  w3_intervention_effect_rmse: {result['w3_intervention_effect_rmse']:.4f} ({result['w3_intervention_effect_rmse']*100:.2f}%)")
    print(f"  w3_tod_sens_base: {result['w3_tod_sens_base']:.4f}")
    print(f"  w3_tod_sens_perturb: {result['w3_tod_sens_perturb']:.4f}")
    print(f"  w3_intervention_effect_tod: {result['w3_intervention_effect_tod']:.4f}")
    print(f"  w3_peak_resp_base: {result['w3_peak_resp_base']:.4f}")
    print(f"  w3_peak_resp_perturb: {result['w3_peak_resp_perturb']:.4f}")
    print(f"  w3_intervention_effect_peak: {result['w3_intervention_effect_peak']:.4f}")
    print(f"  w3_intervention_cohens_d: {result['w3_intervention_cohens_d']:.4f}")
    print(f"  w3_rmse_win_rate: {result['w3_rmse_win_rate']:.4f} ({result['w3_rmse_win_rate']*100:.1f}%)")
    
    # 검증
    expected_rmse_change = (perturb_metrics['rmse'] - baseline_metrics['rmse']) / baseline_metrics['rmse']
    assert abs(result['w3_intervention_effect_rmse'] - expected_rmse_change) < 1e-9
    
    expected_tod_change = perturb_metrics['gc_kernel_tod_dcor'] - baseline_metrics['gc_kernel_tod_dcor']
    assert abs(result['w3_intervention_effect_tod'] - expected_tod_change) < 1e-9
    
    # Cohen's d_z는 양수여야 함 (교란이 오차를 증가시킴)
    assert result['w3_intervention_cohens_d'] > 0
    
    # Win-rate는 1.0이어야 함 (모든 윈도우에서 교란이 더 나쁨)
    assert result['w3_rmse_win_rate'] == 1.0
    
    print("\n✅ compute_w3_metrics() 기본 테스트 통과!\n")


def test_compute_w3_metrics_with_ci():
    """compute_w3_metrics() - 부트스트랩 CI 포함"""
    print("=" * 60)
    print("Test 5: compute_w3_metrics() - 부트스트랩 CI")
    print("=" * 60)
    
    baseline_metrics = {"rmse": 0.3456, "gc_kernel_tod_dcor": 0.45, "cg_event_gain": 0.67}
    perturb_metrics = {"rmse": 0.4123, "gc_kernel_tod_dcor": 0.38, "cg_event_gain": 0.54}
    
    np.random.seed(42)
    win_errors_base = np.random.uniform(0.08, 0.12, 20)
    win_errors_pert = win_errors_base + np.random.uniform(0.02, 0.05, 20)
    
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=win_errors_base,
        win_errors_pert=win_errors_pert,
        dz_ci=True  # CI 계산
    )
    
    print("\n📊 결과 (CI 포함):")
    print(f"  w3_intervention_cohens_d: {result['w3_intervention_cohens_d']:.4f}")
    print(f"  w3_cohens_d_ci_low: {result['w3_cohens_d_ci_low']:.4f}")
    print(f"  w3_cohens_d_ci_high: {result['w3_cohens_d_ci_high']:.4f}")
    
    # CI가 d_z를 포함하는지 확인
    dz = result['w3_intervention_cohens_d']
    ci_low = result['w3_cohens_d_ci_low']
    ci_high = result['w3_cohens_d_ci_high']
    
    assert ci_low <= dz <= ci_high, "CI가 Cohen's d_z를 포함해야 함"
    
    print("\n✅ 부트스트랩 CI 테스트 통과!\n")


def test_edge_cases():
    """엣지 케이스 테스트"""
    print("=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)
    
    baseline_metrics = {"rmse": 0.3456, "gc_kernel_tod_dcor": 0.45, "cg_event_gain": 0.67}
    perturb_metrics = {"rmse": 0.4123, "gc_kernel_tod_dcor": 0.38, "cg_event_gain": 0.54}
    
    # Case 1: win_errors 없음
    print("\n[Case 1] win_errors 없음")
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=None,
        win_errors_pert=None
    )
    assert np.isnan(result['w3_intervention_cohens_d'])
    assert np.isnan(result['w3_rmse_win_rate'])
    print("  ✅ Cohen's d와 win-rate가 NaN으로 처리됨")
    
    # Case 2: 크기가 다른 win_errors
    print("\n[Case 2] 크기가 다른 win_errors")
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=np.array([0.1, 0.2, 0.3]),
        win_errors_pert=np.array([0.15, 0.25])  # 크기 다름
    )
    assert np.isnan(result['w3_intervention_cohens_d'])
    print("  ✅ 크기 불일치 시 NaN 처리됨")
    
    # Case 3: 윈도우 수가 너무 적음 (N < 2)
    print("\n[Case 3] 윈도우 수 부족 (N=1)")
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=np.array([0.1]),
        win_errors_pert=np.array([0.15])
    )
    assert np.isnan(result['w3_intervention_cohens_d'])
    print("  ✅ 윈도우 수 부족 시 NaN 처리됨")
    
    print("\n✅ 모든 엣지 케이스 테스트 통과!\n")


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 60)
    print("W3 새로운 인터페이스 테스트 시작")
    print("=" * 60 + "\n")
    
    try:
        test_safe_div()
        test_paired_cohens_dz()
        test_bootstrap_ci()
        test_compute_w3_metrics_basic()
        test_compute_w3_metrics_with_ci()
        test_edge_cases()
        
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 통과!")
        print("=" * 60)
        print("\n✨ W3 수정 사항이 정상적으로 작동합니다.")
        print("   - Paired Cohen's d_z 계산 ✅")
        print("   - 원본 값 3쌍 저장 ✅")
        print("   - Win-rate 계산 ✅")
        print("   - 부트스트랩 CI (선택적) ✅")
        
    except AssertionError as e:
        print(f"\n❌ 테스트 실패: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ 예외 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

