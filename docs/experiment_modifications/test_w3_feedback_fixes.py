"""
W3 피드백 반영 테스트:
1. w3_rmse_win_rate → w3_rank_preservation_rate로 키 변경
2. w3_lag_distribution_change 계산 추가
"""

import sys
sys.path.insert(0, '/home/himchan/proj/CTSF/CTSF-W')

import numpy as np
from utils.experiment_metrics.w3_metrics import compute_w3_metrics


def test_key_name_change():
    """키 이름 변경 테스트: w3_rmse_win_rate → w3_rank_preservation_rate"""
    print("=" * 60)
    print("Test 1: 키 이름 변경 (w3_rank_preservation_rate)")
    print("=" * 60)
    
    baseline_metrics = {
        "rmse": 0.3456,
        "gc_kernel_tod_dcor": 0.45,
        "cg_event_gain": 0.67,
        "cg_bestlag": 2.3,
    }
    
    perturb_metrics = {
        "rmse": 0.4123,
        "gc_kernel_tod_dcor": 0.38,
        "cg_event_gain": 0.54,
        "cg_bestlag": 2.8,
    }
    
    np.random.seed(42)
    win_errors_base = np.random.uniform(0.08, 0.12, 10)
    win_errors_pert = win_errors_base + np.random.uniform(0.02, 0.05, 10)
    
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=win_errors_base,
        win_errors_pert=win_errors_pert
    )
    
    print(f"\n📊 결과 키 확인:")
    print(f"  'w3_rank_preservation_rate' in result: {'w3_rank_preservation_rate' in result}")
    print(f"  'w3_rmse_win_rate' in result: {'w3_rmse_win_rate' in result}")
    
    if 'w3_rank_preservation_rate' in result:
        print(f"  ✅ w3_rank_preservation_rate: {result['w3_rank_preservation_rate']:.4f} ({result['w3_rank_preservation_rate']*100:.1f}%)")
    
    # 검증
    assert 'w3_rank_preservation_rate' in result, "w3_rank_preservation_rate 키가 있어야 함"
    assert 'w3_rmse_win_rate' not in result, "w3_rmse_win_rate 키가 없어야 함"
    assert result['w3_rank_preservation_rate'] == 1.0, "모든 윈도우에서 교란이 더 나쁘므로 1.0"
    
    print("\n✅ 키 이름 변경 테스트 통과!\n")


def test_lag_distribution_change():
    """라그 분포 변화 계산 테스트"""
    print("=" * 60)
    print("Test 2: 라그 분포 변화 계산 (w3_lag_distribution_change)")
    print("=" * 60)
    
    baseline_metrics = {
        "rmse": 0.3456,
        "gc_kernel_tod_dcor": 0.45,
        "cg_event_gain": 0.67,
        "cg_bestlag": 2.3,  # baseline lag
    }
    
    perturb_metrics = {
        "rmse": 0.4123,
        "gc_kernel_tod_dcor": 0.38,
        "cg_event_gain": 0.54,
        "cg_bestlag": 2.8,  # perturbed lag (증가)
    }
    
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=None,
        win_errors_pert=None
    )
    
    print(f"\n📊 결과:")
    print(f"  cg_bestlag (baseline): {baseline_metrics['cg_bestlag']:.4f}")
    print(f"  cg_bestlag (perturbed): {perturb_metrics['cg_bestlag']:.4f}")
    print(f"  w3_lag_distribution_change: {result['w3_lag_distribution_change']:.4f}")
    
    expected_change = abs(perturb_metrics['cg_bestlag'] - baseline_metrics['cg_bestlag'])
    print(f"  예상값: {expected_change:.4f}")
    
    # 검증
    assert 'w3_lag_distribution_change' in result, "w3_lag_distribution_change 키가 있어야 함"
    assert abs(result['w3_lag_distribution_change'] - expected_change) < 1e-9, "라그 변화 계산 오류"
    
    print("\n✅ 라그 분포 변화 계산 테스트 통과!\n")


def test_lag_distribution_change_nan():
    """라그 분포 변화 NaN 처리 테스트"""
    print("=" * 60)
    print("Test 3: 라그 분포 변화 NaN 처리")
    print("=" * 60)
    
    # Case 1: bestlag 없음
    baseline_metrics = {"rmse": 0.3456, "gc_kernel_tod_dcor": 0.45, "cg_event_gain": 0.67}
    perturb_metrics = {"rmse": 0.4123, "gc_kernel_tod_dcor": 0.38, "cg_event_gain": 0.54}
    
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=None,
        win_errors_pert=None
    )
    
    print(f"\n[Case 1] bestlag 없음:")
    print(f"  w3_lag_distribution_change: {result['w3_lag_distribution_change']}")
    assert np.isnan(result['w3_lag_distribution_change']), "bestlag 없으면 NaN이어야 함"
    print("  ✅ NaN으로 처리됨")
    
    # Case 2: bestlag가 NaN
    baseline_metrics = {"rmse": 0.3456, "cg_bestlag": np.nan}
    perturb_metrics = {"rmse": 0.4123, "cg_bestlag": 2.8}
    
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=None,
        win_errors_pert=None
    )
    
    print(f"\n[Case 2] bestlag가 NaN:")
    print(f"  w3_lag_distribution_change: {result['w3_lag_distribution_change']}")
    assert np.isnan(result['w3_lag_distribution_change']), "bestlag가 NaN이면 NaN이어야 함"
    print("  ✅ NaN으로 처리됨")
    
    print("\n✅ NaN 처리 테스트 통과!\n")


def test_full_metrics():
    """전체 지표 포함 테스트"""
    print("=" * 60)
    print("Test 4: 전체 지표 확인")
    print("=" * 60)
    
    baseline_metrics = {
        "rmse": 0.3456,
        "gc_kernel_tod_dcor": 0.45,
        "cg_event_gain": 0.67,
        "cg_bestlag": 2.3,
    }
    
    perturb_metrics = {
        "rmse": 0.4123,
        "gc_kernel_tod_dcor": 0.38,
        "cg_event_gain": 0.54,
        "cg_bestlag": 2.8,
    }
    
    np.random.seed(42)
    win_errors_base = np.random.uniform(0.08, 0.12, 20)
    win_errors_pert = win_errors_base + np.random.uniform(0.02, 0.05, 20)
    
    result = compute_w3_metrics(
        baseline_metrics=baseline_metrics,
        perturb_metrics=perturb_metrics,
        win_errors_base=win_errors_base,
        win_errors_pert=win_errors_pert,
        dz_ci=True
    )
    
    expected_keys = [
        'w3_rmse_base', 'w3_rmse_perturb',
        'w3_tod_sens_base', 'w3_tod_sens_perturb',
        'w3_peak_resp_base', 'w3_peak_resp_perturb',
        'w3_intervention_effect_rmse',
        'w3_intervention_effect_tod',
        'w3_intervention_effect_peak',
        'w3_intervention_cohens_d',
        'w3_rank_preservation_rate',  # ← 변경된 키
        'w3_lag_distribution_change',  # ← 추가된 키
        'w3_cohens_d_ci_low',
        'w3_cohens_d_ci_high',
    ]
    
    print(f"\n📊 전체 지표 확인:")
    for key in expected_keys:
        if key in result:
            value = result[key]
            if isinstance(value, float):
                print(f"  ✅ {key}: {value:.4f}")
            else:
                print(f"  ✅ {key}: {value}")
        else:
            print(f"  ❌ {key}: MISSING")
    
    # 검증
    for key in expected_keys:
        assert key in result, f"{key} 키가 없습니다"
    
    # CSV 로거에서 정의된 W3 컬럼들이 모두 있는지 확인
    csv_w3_cols = [
        "w3_intervention_effect_rmse",
        "w3_intervention_effect_tod",
        "w3_intervention_effect_peak",
        "w3_intervention_cohens_d",
        "w3_rank_preservation_rate",
        "w3_lag_distribution_change",
    ]
    
    print(f"\n📋 CSV 컬럼 매칭 확인:")
    for col in csv_w3_cols:
        if col in result:
            print(f"  ✅ {col}: 존재")
        else:
            print(f"  ❌ {col}: 누락")
    
    for col in csv_w3_cols:
        assert col in result, f"CSV 컬럼 {col}이 결과에 없습니다"
    
    print("\n✅ 전체 지표 확인 테스트 통과!\n")


def main():
    """메인 테스트 실행"""
    print("\n" + "=" * 60)
    print("W3 피드백 반영 테스트 시작")
    print("=" * 60 + "\n")
    
    try:
        test_key_name_change()
        test_lag_distribution_change()
        test_lag_distribution_change_nan()
        test_full_metrics()
        
        print("\n" + "=" * 60)
        print("🎉 모든 테스트 통과!")
        print("=" * 60)
        print("\n✨ W3 피드백 반영 완료:")
        print("   1. w3_rmse_win_rate → w3_rank_preservation_rate ✅")
        print("   2. w3_lag_distribution_change 계산 추가 ✅")
        print("   3. CSV 컬럼명과 완벽히 매칭 ✅")
        
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

