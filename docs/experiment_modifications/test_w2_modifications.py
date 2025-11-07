"""
W2 실험 수정 사항 테스트 스크립트
게이트 출력 수집 및 W2 지표 계산이 정상적으로 동작하는지 확인
"""

import torch
import numpy as np
import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from config.config import load_config, apply_HP2
from models.ctsf_model import HybridTS
from models.experiment_variants import StaticCrossModel
from utils.hooks import GateOutputHooks
from utils.experiment_metrics.w2_metrics import compute_w2_metrics


def test_gate_output_hooks():
    """GateOutputHooks 클래스가 올바르게 동작하는지 테스트"""
    print("=" * 80)
    print("테스트 1: GateOutputHooks 클래스 기본 동작 확인")
    print("=" * 80)
    
    # 간단한 모델 생성
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
        model = HybridTS(cfg, n_vars=7)
        model.eval()
        
        # GateOutputHooks 생성
        gate_hooks = GateOutputHooks(model)
        print(f"  ✓ GateOutputHooks 생성 성공")
        print(f"    - CrossHyperConv 블록 개수: {gate_hooks.n_layers}")
        
        # 훅 등록
        gate_hooks.attach()
        print(f"  ✓ 훅 등록 성공")
        
        # 게이트 값 직접 수집 (forward pass 없이)
        B = 4
        gate_hooks.collect_gate_values(B)
        
        print(f"  ✓ 게이트 값 수집 성공")
        
        # 수집된 데이터 확인
        outputs = gate_hooks.get_outputs()
        gate_outputs = outputs.get("gate_outputs")
        
        if gate_outputs:
            print(f"  ✓ 게이트 출력 구조 확인")
            print(f"    - 배치 개수: {len(gate_outputs)}")
            print(f"    - 첫 배치 shape: {gate_outputs[0].shape}")
            expected_shape = (B, gate_hooks.n_layers, 4)
            if gate_outputs[0].shape == expected_shape:
                print(f"    - ✓ Shape 검증 통과: {expected_shape}")
            else:
                print(f"    - ✗ Shape 불일치: 예상 {expected_shape}, 실제 {gate_outputs[0].shape}")
                return False
        else:
            print(f"  ✗ 게이트 출력이 None입니다")
            return False
        
        # 훅 제거
        gate_hooks.detach()
        print(f"  ✓ 훅 제거 성공")
        
    except Exception as e:
        print(f"  ✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_w2_metrics_computation():
    """W2 지표 계산 함수가 올바르게 동작하는지 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: W2 지표 계산 함수 확인")
    print("=" * 80)
    
    # 간단한 모델 생성
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
        model = HybridTS(cfg, n_vars=7)
        model.eval()
        
        # 가상 데이터 생성
        N, n_layers, B, d = 100, 7, 4, 64
        
        # 동적 게이트 출력 시뮬레이션
        gate_outputs = [
            np.random.rand(B, n_layers, 4) for _ in range(N // B)
        ]
        gru_states = [
            np.random.randn(B, d) for _ in range(N // B)
        ]
        
        hooks_data = {
            "gate_outputs": gate_outputs,
            "gru_states": gru_states
        }
        
        # TOD 벡터
        tod_vec = np.random.randn(N, 2)
        
        # direct_evidence (cg_event_gain 등)
        direct_evidence = {
            "cg_event_gain": 0.75,
            "cg_event_hit": 0.80,
        }
        
        print("  가상 데이터 생성 완료")
        print(f"    - 샘플 수: {N}")
        print(f"    - 게이트 배치: {len(gate_outputs)}")
        print(f"    - GRU 상태 배치: {len(gru_states)}")
        
        # W2 지표 계산
        w2_metrics = compute_w2_metrics(
            model=model,
            hooks_data=hooks_data,
            tod_vec=tod_vec,
            direct_evidence=direct_evidence
        )
        
        print(f"  ✓ W2 지표 계산 성공")
        print(f"\n  계산된 지표들:")
        
        expected_metrics = [
            "w2_gate_variability_time",
            "w2_gate_variability_sample",
            "w2_gate_entropy",
            "w2_gate_tod_alignment",
            "w2_gate_gru_state_alignment",
            "w2_event_conditional_response",
            "w2_channel_selectivity_kurtosis",
            "w2_channel_selectivity_sparsity",
        ]
        
        all_present = True
        for metric_name in expected_metrics:
            if metric_name in w2_metrics:
                value = w2_metrics[metric_name]
                is_nan = np.isnan(value) if isinstance(value, (int, float)) else False
                status = "NaN" if is_nan else f"{value:.6f}"
                print(f"    ✓ {metric_name}: {status}")
            else:
                print(f"    ✗ {metric_name}: 누락")
                all_present = False
        
        if all_present:
            print(f"\n  ✓ 모든 예상 지표가 존재합니다")
        else:
            print(f"\n  ✗ 일부 지표가 누락되었습니다")
            return False
        
    except Exception as e:
        print(f"  ✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_static_vs_dynamic_mode():
    """정적 모드와 동적 모드의 alpha 파라미터 확인"""
    print("\n" + "=" * 80)
    print("테스트 3: 정적 vs 동적 모드 alpha 파라미터 비교")
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
        # 동적 모델
        print("\n  [동적 모드]")
        dynamic_model = HybridTS(cfg, n_vars=7)
        dynamic_model.eval()
        
        gate_hooks_dynamic = GateOutputHooks(dynamic_model)
        B = 4
        gate_hooks_dynamic.collect_gate_values(B)
        
        dynamic_outputs = gate_hooks_dynamic.get_outputs()
        
        print(f"    ✓ 동적 모델 alpha 파라미터 수집 완료")
        if dynamic_outputs.get("gate_outputs"):
            print(f"      - 게이트 shape: {dynamic_outputs['gate_outputs'][0].shape}")
        
        # 정적 모델
        print("\n  [정적 모드]")
        static_model = StaticCrossModel(cfg, n_vars=7)
        static_model.eval()
        
        gate_hooks_static = GateOutputHooks(static_model)
        gate_hooks_static.collect_gate_values(B)
        
        static_outputs = gate_hooks_static.get_outputs()
        
        print(f"    ✓ 정적 모델 alpha 파라미터 수집 완료")
        if static_outputs.get("gate_outputs"):
            print(f"      - 게이트 shape: {static_outputs['gate_outputs'][0].shape}")
        
        # 비교
        print("\n  [비교 결과]")
        dyn_gates = dynamic_outputs.get("gate_outputs")
        sta_gates = static_outputs.get("gate_outputs")
        
        if dyn_gates and sta_gates:
            dyn_arr = np.concatenate(dyn_gates, axis=0)
            sta_arr = np.concatenate(sta_gates, axis=0)
            
            dyn_mean = np.mean(dyn_arr, axis=0)  # (n_layers, 4)
            sta_mean = np.mean(sta_arr, axis=0)  # (n_layers, 4)
            
            print(f"    동적 모델 평균 alpha: shape {dyn_mean.shape}")
            print(f"    정적 모델 평균 alpha: shape {sta_mean.shape}")
            print(f"    ✓ Alpha 파라미터 비교 완료")
        else:
            print(f"    ✗ 게이트 데이터 수집 실패")
            return False
        
    except Exception as e:
        print(f"  ✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_integration_with_evaluate():
    """evaluate_with_direct_evidence 함수와의 통합 테스트"""
    print("\n" + "=" * 80)
    print("테스트 4: evaluate_with_direct_evidence 통합")
    print("=" * 80)
    
    try:
        from utils.direct_evidence import evaluate_with_direct_evidence
        from data.dataset import load_split_dataloaders
        
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
        cfg["batch_size"] = 8  # 작은 배치로 테스트
        
        print("  데이터 로딩 중...")
        train_loader, val_loader, test_loader, (mu_np, std_np, C) = load_split_dataloaders(cfg)
        mu = torch.tensor(mu_np, device="cpu").unsqueeze(-1)
        std = torch.tensor(std_np, device="cpu").unsqueeze(-1)
        
        print("  모델 생성 중...")
        model = HybridTS(cfg, n_vars=C)
        model.eval()
        
        # 작은 데이터셋으로 테스트 (첫 2개 배치만)
        small_dataset = torch.utils.data.Subset(
            test_loader.dataset, 
            list(range(min(16, len(test_loader.dataset))))
        )
        small_loader = torch.utils.data.DataLoader(
            small_dataset, 
            batch_size=8, 
            shuffle=False
        )
        
        print("  평가 실행 중 (collect_gate_outputs=True)...")
        result = evaluate_with_direct_evidence(
            model, small_loader, mu, std,
            tod_vec=None, device="cpu",
            collect_gate_outputs=True
        )
        
        print(f"  ✓ 평가 완료")
        print(f"\n  결과 키 확인:")
        
        # 기본 지표 확인
        basic_keys = ["mse_std", "mse_real", "rmse"]
        for key in basic_keys:
            if key in result:
                print(f"    ✓ {key}: {result[key]:.6f}")
            else:
                print(f"    ✗ {key}: 누락")
        
        # hooks_data 확인
        if "hooks_data" in result:
            print(f"    ✓ hooks_data 존재")
            hooks_data = result["hooks_data"]
            
            if hooks_data.get("gate_outputs"):
                print(f"      - gate_outputs 개수: {len(hooks_data['gate_outputs'])}")
                print(f"      - 첫 배치 shape: {hooks_data['gate_outputs'][0].shape}")
            else:
                print(f"      - ✗ gate_outputs 없음")
            
            if hooks_data.get("gru_states"):
                print(f"      - gru_states 개수: {len(hooks_data['gru_states'])}")
                print(f"      - 첫 배치 shape: {hooks_data['gru_states'][0].shape}")
            else:
                print(f"      - ✗ gru_states 없음")
        else:
            print(f"    ✗ hooks_data 누락")
            return False
        
    except Exception as e:
        print(f"  ✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """모든 테스트 실행"""
    print("\n" + "=" * 80)
    print("W2 실험 수정 사항 테스트 시작")
    print("=" * 80 + "\n")
    
    tests = [
        ("GateOutputHooks 동작", test_gate_output_hooks),
        ("W2 지표 계산", test_w2_metrics_computation),
        ("정적 vs 동적 모드", test_static_vs_dynamic_mode),
        ("evaluate 통합", test_integration_with_evaluate),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n테스트 '{test_name}' 실행 중 오류: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    for test_name, success in results:
        status = "✓ 통과" if success else "✗ 실패"
        print(f"  {status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, s in results if s)
    
    print(f"\n  총 {total}개 테스트 중 {passed}개 통과")
    
    if passed == total:
        print("\n  ✓ 모든 테스트 통과!")
        return 0
    else:
        print(f"\n  ✗ {total - passed}개 테스트 실패")
        return 1


if __name__ == "__main__":
    exit(main())

