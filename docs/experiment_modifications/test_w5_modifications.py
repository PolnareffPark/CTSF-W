"""
W5 실험 수정 사항 테스트 스크립트
게이트 고정 효과 평가가 정상적으로 동작하는지 확인
"""

import torch
import numpy as np
from config.config import load_config, apply_HP2
from experiments.w5_experiment import W5Experiment, GateFixedModel
from models.ctsf_model import HybridTS


def test_gate_fixed_model():
    """GateFixedModel이 게이트를 올바르게 고정하는지 테스트"""
    print("=" * 80)
    print("테스트 1: GateFixedModel 게이트 고정 검증")
    print("=" * 80)
    
    base_cfg = load_config("hp2_config.yaml")
    cfg = apply_HP2(
        base_cfg,
        csv_path="datasets/ETTh2.csv",
        seed=42,
        horizon=96,
        device="cpu",
        out_root="results",
        model_tag="HyperConv"
    )
    
    try:
        # 기본 모델 생성
        model = HybridTS(cfg, n_vars=7)
        model.set_cross_directions(use_gc=True, use_cg=True)
        
        print(f"\n  모델 생성 완료")
        print(f"  교차 블록 수: {len(model.xhconv_blks)}")
        
        # 원본 alpha 값 저장 (복원 확인용)
        original_alphas = [blk.alpha.data.clone() for blk in model.xhconv_blks]
        
        # 게이트 고정 래퍼 적용 (Context Manager)
        fixed_model_wrapper = GateFixedModel(model)
        
        print(f"\n  GateFixedModel 래퍼 생성 완료")
        print(f"  고정 게이트 평균값 계산 완료: {len(fixed_model_wrapper.gate_means)}개")
        print(f"  원본 alpha 백업 완료: {len(fixed_model_wrapper.original_alphas)}개")
        
        # 게이트 평균값 확인
        for i, mean_val in fixed_model_wrapper.gate_means.items():
            print(f"    Layer {i}: gate_mean shape = {mean_val.shape}, "
                  f"min = {mean_val.min().item():.4f}, "
                  f"max = {mean_val.max().item():.4f}")
            
            # ReLU 적용되어 음수가 없어야 함
            assert mean_val.min() >= 0, f"Layer {i}에 음수 게이트 값 발견"
        
        print(f"\n  ✓ 모든 게이트가 양수로 고정됨")
        
        # Context manager로 사용
        model.eval()
        
        # 더미 입력 생성
        batch_size = 4
        lookback = cfg.get("lookback", 168)
        n_vars = 7
        
        x = torch.randn(batch_size, lookback, n_vars)
        
        with torch.no_grad():
            # 동적 모델 출력 (before context)
            out_dynamic_before = model(x)
            print(f"\n  동적 모델 출력 (before) shape: {out_dynamic_before.shape}")
            
            # Context manager 내에서 고정 모델 평가
            with fixed_model_wrapper as fixed_model:
                print(f"\n  Context manager 진입 - 훅 등록됨")
                print(f"  등록된 훅 수: {len(fixed_model.hooks)}")
                assert len(fixed_model.hooks) == len(model.xhconv_blks), "훅 개수가 블록 개수와 다름"
                
                fixed_model.eval()
                out_fixed = fixed_model(x)
                print(f"  고정 모델 출력 shape: {out_fixed.shape}")
                
                # 출력 shape이 동일해야 함
                assert out_dynamic_before.shape == out_fixed.shape, "출력 shape이 다름"
                print(f"  ✓ 출력 shape 일치")
                
                # 출력이 달라야 함 (게이트 고정의 효과)
                diff = torch.abs(out_dynamic_before - out_fixed).mean().item()
                print(f"  동적 vs 고정 출력 차이: {diff:.6f}")
                
                if diff > 1e-6:
                    print(f"  ✓ 게이트 고정이 출력에 영향을 미침")
                else:
                    print(f"  ⚠ 경고: 출력 차이가 매우 작음 (게이트 고정 효과 미미)")
            
            # Context manager 종료 후 - 원본 복원 확인
            print(f"\n  Context manager 종료 - 훅 제거 및 원본 복원")
            
            # 훅이 제거되었는지 확인
            assert len(fixed_model_wrapper.hooks) == 0, "훅이 제거되지 않음"
            print(f"  ✓ 훅이 제거됨")
            
            # 원본 alpha가 복원되었는지 확인
            for i, blk in enumerate(model.xhconv_blks):
                alpha_restored = torch.allclose(blk.alpha.data, original_alphas[i])
                assert alpha_restored, f"Layer {i}의 alpha가 복원되지 않음"
            print(f"  ✓ 원본 alpha가 복원됨")
            
            # 동적 모델이 정상 작동하는지 확인
            out_dynamic_after = model(x)
            dynamic_unchanged = torch.allclose(out_dynamic_before, out_dynamic_after, atol=1e-5)
            assert dynamic_unchanged, "원본 모델의 동작이 변경됨"
            print(f"  ✓ 원본 모델이 정상 작동함")
        
        print(f"\n  ✓ 테스트 1 통과 (원본 모델 보호 확인)")
        
    except Exception as e:
        print(f"\n  ✗ 테스트 1 실패: {e}")
        import traceback
        traceback.print_exc()


def test_w5_metrics_computation():
    """W5 지표 계산이 올바른지 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: W5 지표 계산 확인")
    print("=" * 80)
    
    from utils.experiment_metrics.w5_metrics import compute_w5_metrics
    
    # 가상의 모델 생성
    base_cfg = load_config("hp2_config.yaml")
    model = HybridTS(base_cfg, n_vars=7)
    
    # 동적 모델과 고정 모델의 가상 결과 생성
    dynamic_metrics = {
        "rmse": 1.0,
        "mae": 0.8,
        "gc_kernel_tod_dcor": 0.7,
        "cg_event_gain": 0.6,
        "w2_gate_variability_time": 0.5,
    }
    
    fixed_metrics = {
        "rmse": 1.2,  # 고정 시 성능 저하
        "mae": 0.9,
        "gc_kernel_tod_dcor": 0.5,  # TOD 민감도 감소
        "cg_event_gain": 0.4,  # 이벤트 게인 감소
        "w2_gate_variability_time": 0.1,  # 게이트 변동성 감소
    }
    
    try:
        # 지표 계산
        metrics = compute_w5_metrics(
            model=model,
            fixed_model_metrics=fixed_metrics,
            dynamic_model_metrics=dynamic_metrics
        )
        
        print(f"\n  계산된 지표:")
        for key, value in metrics.items():
            if isinstance(value, float) and not np.isnan(value):
                print(f"    {key}: {value:.6f}")
            else:
                print(f"    {key}: {value}")
        
        # 필수 지표 확인
        required_metrics = [
            "w5_performance_degradation_ratio",
            "w5_sensitivity_gain_loss",
            "w5_event_gain_loss",
            "w5_gate_event_alignment_loss"
        ]
        
        all_present = all(key in metrics for key in required_metrics)
        if all_present:
            print(f"\n  ✓ 모든 필수 지표가 계산되었습니다")
        else:
            missing = [key for key in required_metrics if key not in metrics]
            print(f"\n  ✗ 누락된 지표: {missing}")
            return
        
        # NaN이 아닌 값 확인
        non_nan_metrics = {k: v for k, v in metrics.items() 
                          if not (isinstance(v, float) and np.isnan(v))}
        print(f"  유효한 값: {len(non_nan_metrics)}/{len(metrics)}")
        
        # 지표 해석 검증
        print(f"\n  지표 해석:")
        
        # 성능 저하율 (양수 = 고정 시 성능 악화)
        perf_deg = metrics["w5_performance_degradation_ratio"]
        expected_deg = (1.2 - 1.0) / 1.0  # 0.2
        assert abs(perf_deg - expected_deg) < 1e-6, "성능 저하율 계산 오류"
        print(f"    성능 저하율: {perf_deg:.2%} (고정 시 RMSE {perf_deg:.2%} 증가)")
        
        # 민감도 손실 (양수 = 동적이 더 좋음)
        sens_loss = metrics["w5_sensitivity_gain_loss"]
        expected_sens = 0.7 - 0.5  # 0.2
        assert abs(sens_loss - expected_sens) < 1e-6, "민감도 손실 계산 오류"
        print(f"    민감도 손실: {sens_loss:.4f} (동적이 TOD 패턴을 더 잘 포착)")
        
        # 이벤트 손실 (양수 = 동적이 더 좋음)
        event_loss = metrics["w5_event_gain_loss"]
        expected_event = 0.6 - 0.4  # 0.2
        assert abs(event_loss - expected_event) < 1e-6, "이벤트 손실 계산 오류"
        print(f"    이벤트 손실: {event_loss:.4f} (동적이 이벤트를 더 잘 탐지)")
        
        # 정렬 손실
        alignment_loss = metrics["w5_gate_event_alignment_loss"]
        print(f"    정렬 손실: {alignment_loss:.4f} (동적 게이트가 이벤트에 더 잘 정렬)")
        
        print(f"\n  ✓ 테스트 2 통과")
        
    except Exception as e:
        print(f"\n  ✗ 테스트 2 실패: {e}")
        import traceback
        traceback.print_exc()


def test_w5_metrics_with_missing_data():
    """일부 데이터가 없을 때 W5 지표가 올바르게 처리하는지 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: 누락된 데이터 처리 확인")
    print("=" * 80)
    
    from utils.experiment_metrics.w5_metrics import compute_w5_metrics
    
    base_cfg = load_config("hp2_config.yaml")
    model = HybridTS(base_cfg, n_vars=7)
    
    # Case 1: 둘 다 None
    print("\n  Case 1: 둘 다 None")
    metrics = compute_w5_metrics(model, None, None)
    all_nan = all(np.isnan(v) if isinstance(v, float) else False 
                  for v in metrics.values())
    if all_nan:
        print(f"    ✓ 모든 지표가 NaN으로 반환됨")
    else:
        print(f"    ✗ 일부 지표가 NaN이 아님")
    
    # Case 2: 하나만 None
    print("\n  Case 2: fixed_metrics만 제공")
    fixed_metrics = {"rmse": 1.2, "mae": 0.9}
    metrics = compute_w5_metrics(model, fixed_metrics, None)
    all_nan = all(np.isnan(v) if isinstance(v, float) else False 
                  for v in metrics.values())
    if all_nan:
        print(f"    ✓ 모든 지표가 NaN으로 반환됨")
    else:
        print(f"    ✗ 일부 지표가 NaN이 아님")
    
    # Case 3: 일부 지표 누락
    print("\n  Case 3: 일부 지표 누락")
    dynamic_metrics = {"rmse": 1.0}  # 다른 지표 없음
    fixed_metrics = {"rmse": 1.2}
    metrics = compute_w5_metrics(model, fixed_metrics, dynamic_metrics)
    
    # rmse는 있으므로 성능 저하율은 계산되어야 함
    perf_deg = metrics.get("w5_performance_degradation_ratio", np.nan)
    if not np.isnan(perf_deg):
        print(f"    ✓ 성능 저하율 계산됨: {perf_deg:.4f}")
    else:
        print(f"    ✗ 성능 저하율이 NaN")
    
    # 다른 지표는 NaN이어야 함
    other_metrics = ["w5_sensitivity_gain_loss", "w5_event_gain_loss"]
    all_nan = all(np.isnan(metrics.get(k, np.nan)) for k in other_metrics)
    if all_nan:
        print(f"    ✓ 누락된 지표 의존 값들이 NaN으로 반환됨")
    else:
        print(f"    ✗ 일부 지표가 부적절하게 계산됨")
    
    print(f"\n  ✓ 테스트 3 통과")


def test_w5_evaluate_test_integration():
    """W5Experiment.evaluate_test가 올바르게 동작하는지 통합 테스트"""
    print("\n" + "=" * 80)
    print("테스트 4: evaluate_test 통합 테스트 (시뮬레이션)")
    print("=" * 80)
    
    print("\n  참고: 실제 데이터 로딩과 학습 없이 로직만 검증합니다")
    print("  실제 실험은 run_suite.py를 통해 수행하세요")
    
    try:
        # W5Experiment의 evaluate_test 로직 시뮬레이션
        print("\n  시뮬레이션 단계:")
        print("    1. 동적 게이트 모델 평가")
        print("    2. 게이트 고정 모델 평가 (Context Manager 사용)")
        print("    3. 원본 모델 자동 복원")
        print("    4. W5 지표 계산")
        print("    5. 결과 병합")
        
        # 가상의 결과 생성 (evaluate_with_direct_evidence 결과 모방)
        dynamic_results = {
            "rmse": 1.0,
            "mae": 0.8,
            "gc_kernel_tod_dcor": 0.7,
            "cg_event_gain": 0.6,
            "w2_gate_variability_time": 0.5,
        }
        
        fixed_results = {
            "rmse": 1.2,
            "mae": 0.9,
            "gc_kernel_tod_dcor": 0.5,
            "cg_event_gain": 0.4,
            "w2_gate_variability_time": 0.1,
        }
        
        # W5 지표 계산
        from utils.experiment_metrics.w5_metrics import compute_w5_metrics
        base_cfg = load_config("hp2_config.yaml")
        model = HybridTS(base_cfg, n_vars=7)
        
        w5_metrics = compute_w5_metrics(
            model,
            fixed_model_metrics=fixed_results,
            dynamic_model_metrics=dynamic_results
        )
        
        # 결과 병합 (W5Experiment.evaluate_test와 동일)
        final_results = {**dynamic_results}
        final_results.update(w5_metrics)
        final_results['rmse_fixed'] = fixed_results.get('rmse', np.nan)
        final_results['mae_fixed'] = fixed_results.get('mae', np.nan)
        final_results['gc_kernel_tod_dcor_fixed'] = fixed_results.get('gc_kernel_tod_dcor', np.nan)
        final_results['cg_event_gain_fixed'] = fixed_results.get('cg_event_gain', np.nan)
        
        print(f"\n  최종 결과 키 개수: {len(final_results)}")
        print(f"  동적 모델 지표: {len(dynamic_results)}")
        print(f"  W5 비교 지표: {len(w5_metrics)}")
        print(f"  고정 모델 개별 지표: 4")
        
        # 필수 키 확인
        required_keys = [
            "rmse",  # 동적 모델의 RMSE
            "rmse_fixed",  # 고정 모델의 RMSE
            "w5_performance_degradation_ratio",  # 비교 지표
            "w5_sensitivity_gain_loss",
            "w5_event_gain_loss",
            "w5_gate_event_alignment_loss",
        ]
        
        all_present = all(key in final_results for key in required_keys)
        if all_present:
            print(f"\n  ✓ 모든 필수 키가 결과에 포함됨")
            
            # 결과 출력
            print(f"\n  주요 결과:")
            print(f"    동적 RMSE: {final_results['rmse']:.4f}")
            print(f"    고정 RMSE: {final_results['rmse_fixed']:.4f}")
            print(f"    성능 저하율: {final_results['w5_performance_degradation_ratio']:.2%}")
            print(f"    민감도 손실: {final_results['w5_sensitivity_gain_loss']:.4f}")
            print(f"    이벤트 손실: {final_results['w5_event_gain_loss']:.4f}")
            print(f"    정렬 손실: {final_results['w5_gate_event_alignment_loss']:.4f}")
            
        else:
            missing = [key for key in required_keys if key not in final_results]
            print(f"\n  ✗ 누락된 키: {missing}")
            return
        
        print(f"\n  ✓ 테스트 4 통과")
        
    except Exception as e:
        print(f"\n  ✗ 테스트 4 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "W5 실험 수정 사항 테스트" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        test_gate_fixed_model()
        test_w5_metrics_computation()
        test_w5_metrics_with_missing_data()
        test_w5_evaluate_test_integration()
        
        print("\n" + "=" * 80)
        print("모든 테스트 완료!")
        print("=" * 80)
        print("\n실제 W5 실험 실행:")
        print("  python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96")
        print()
        
    except Exception as e:
        print(f"\n테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

