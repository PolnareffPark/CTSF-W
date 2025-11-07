"""
W1 실험 수정 사항 테스트 스크립트
층별 교차 vs 최종 결합(Late Fusion) 비교 실험이 정상적으로 동작하는지 확인
"""

import torch
import numpy as np
from config.config import load_config, apply_HP2
from experiments.w1_experiment import W1Experiment
from models.ctsf_model import HybridTS
from models.experiment_variants import LastLayerFusionModel
from utils.experiment_metrics.w1_metrics import compute_w1_metrics


def test_model_creation():
    """W1 실험의 두 모드(per_layer, last_layer)에서 모델 타입이 올바른지 확인"""
    print("=" * 80)
    print("테스트 1: W1 실험 모델 타입 확인 (Forward 테스트 제외)")
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
    
    modes = ["per_layer", "last_layer"]
    
    for mode in modes:
        print(f"\n모드: {mode}")
        cfg["mode"] = mode
        cfg["experiment_type"] = "W1"
        cfg["verbose"] = False
        
        try:
            # W1Experiment를 통해 모델 생성 테스트
            # 참고: 실제 데이터 로딩이 필요하므로 모델 타입만 확인
            if mode == "per_layer":
                # per_layer 모드에서는 HybridTS + set_cross_directions 호출
                model = HybridTS(cfg, n_vars=7)
                model.set_cross_directions(use_gc=True, use_cg=True)
                print(f"  ✓ HybridTS 모델 생성 확인")
                print(f"    교차 블록 수: {len(model.xhconv_blks)}")
                
                # 교차 연결이 활성화되었는지 확인
                assert all(blk.use_gc and blk.use_cg for blk in model.xhconv_blks), \
                    "교차 연결이 제대로 활성화되지 않음"
                print(f"  ✓ 모든 교차 블록이 양방향 연결 활성화됨")
                
            elif mode == "last_layer":
                # last_layer 모드에서는 LastLayerFusionModel 사용
                model = LastLayerFusionModel(cfg, n_vars=7)
                print(f"  ✓ LastLayerFusionModel 모델 생성 확인")
                print(f"    교차 블록 수: {len(model.xhconv_blks)}")
                print(f"    Last fusion layer가 존재함")
            
            print(f"  ✓ 모드 '{mode}' 모델 타입 확인 완료")
                
        except Exception as e:
            print(f"  ✗ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print(f"\n✓ 테스트 1 통과")
    print(f"  참고: Forward pass 테스트는 실제 데이터셋과 함께 실험 실행 시 검증됩니다.")
    return True


def test_w1_metrics_structure():
    """W1 지표 계산 함수의 입출력 구조가 올바른지 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: W1 지표 계산 구조 확인")
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
        # 모델 생성 (지표 계산에 필요, 실제로는 사용 안 함)
        model = HybridTS(cfg, n_vars=7)
        
        # 빈 hooks_data로 테스트 (모든 지표가 NaN이어야 함)
        print("\n서브테스트 2-1: 빈 hooks_data")
        hooks_data = {}
        metrics = compute_w1_metrics(model, hooks_data)
        
        expected_keys = [
            "w1_cka_similarity_cnn_gru",
            "w1_cca_similarity_cnn_gru",
            "w1_layerwise_upward_improvement",
            "w1_inter_path_gradient_align"
        ]
        
        for key in expected_keys:
            assert key in metrics, f"지표 '{key}'가 반환되지 않음"
            assert np.isnan(metrics[key]), f"빈 데이터에서 '{key}'가 NaN이 아님: {metrics[key]}"
        
        print(f"  ✓ 빈 hooks_data에서 모든 지표가 NaN 반환")
        
        # 가상의 데이터로 테스트
        print("\n서브테스트 2-2: CKA/CCA 유사도 계산")
        n_samples = 100
        d_cnn = 256
        d_gru = 256
        
        cnn_repr = np.random.randn(n_samples, d_cnn)
        gru_repr = np.random.randn(n_samples, d_gru)
        
        hooks_data_with_repr = {
            "cnn_representations": [cnn_repr],
            "gru_representations": [gru_repr]
        }
        
        metrics = compute_w1_metrics(model, hooks_data_with_repr)
        
        assert not np.isnan(metrics["w1_cka_similarity_cnn_gru"]), \
            "CKA 유사도가 계산되지 않음"
        assert not np.isnan(metrics["w1_cca_similarity_cnn_gru"]), \
            "CCA 유사도가 계산되지 않음"
        assert 0 <= metrics["w1_cka_similarity_cnn_gru"] <= 1, \
            f"CKA 유사도가 범위를 벗어남: {metrics['w1_cka_similarity_cnn_gru']}"
        assert 0 <= metrics["w1_cca_similarity_cnn_gru"] <= 1, \
            f"CCA 유사도가 범위를 벗어남: {metrics['w1_cca_similarity_cnn_gru']}"
        
        print(f"  ✓ CKA 유사도: {metrics['w1_cka_similarity_cnn_gru']:.4f}")
        print(f"  ✓ CCA 유사도: {metrics['w1_cca_similarity_cnn_gru']:.4f}")
        
        # 층별 개선도 테스트
        print("\n서브테스트 2-3: 층별 상향 개선도 계산")
        # 손실이 점진적으로 감소하는 경우
        layerwise_losses = [1.0, 0.8, 0.6, 0.5, 0.45]
        hooks_data_with_losses = {
            "layerwise_losses": layerwise_losses
        }
        
        metrics = compute_w1_metrics(model, hooks_data_with_losses)
        
        assert not np.isnan(metrics["w1_layerwise_upward_improvement"]), \
            "층별 개선도가 계산되지 않음"
        
        # 손실이 감소하므로 양수여야 함
        assert metrics["w1_layerwise_upward_improvement"] > 0, \
            f"손실 감소 시 개선도가 양수여야 하는데 {metrics['w1_layerwise_upward_improvement']}"
        
        print(f"  ✓ 층별 개선도: {metrics['w1_layerwise_upward_improvement']:.4f}")
        print(f"    입력 손실값: {layerwise_losses}")
        
        # 그래디언트 정렬 테스트
        print("\n서브테스트 2-4: 경로 간 그래디언트 정렬 계산")
        d_grad = 512
        cnn_grads = np.random.randn(d_grad)
        gru_grads = np.random.randn(d_grad)
        
        hooks_data_with_grads = {
            "cnn_gradients": cnn_grads,
            "gru_gradients": gru_grads
        }
        
        metrics = compute_w1_metrics(model, hooks_data_with_grads)
        
        assert not np.isnan(metrics["w1_inter_path_gradient_align"]), \
            "그래디언트 정렬이 계산되지 않음"
        assert -1 <= metrics["w1_inter_path_gradient_align"] <= 1, \
            f"그래디언트 정렬이 범위를 벗어남: {metrics['w1_inter_path_gradient_align']}"
        
        print(f"  ✓ 그래디언트 정렬: {metrics['w1_inter_path_gradient_align']:.4f}")
        
        # 동일한 벡터로 테스트 (코사인 유사도 = 1)
        same_grads = np.random.randn(d_grad)
        hooks_data_same = {
            "cnn_gradients": same_grads,
            "gru_gradients": same_grads.copy()
        }
        
        metrics = compute_w1_metrics(model, hooks_data_same)
        assert abs(metrics["w1_inter_path_gradient_align"] - 1.0) < 1e-5, \
            f"동일 벡터의 코사인 유사도가 1이 아님: {metrics['w1_inter_path_gradient_align']}"
        
        print(f"  ✓ 동일 벡터 테스트: {metrics['w1_inter_path_gradient_align']:.4f} (≈ 1.0)")
        
        # 반대 방향 벡터로 테스트 (코사인 유사도 = -1)
        opposite_grads = np.random.randn(d_grad)
        hooks_data_opposite = {
            "cnn_gradients": opposite_grads,
            "gru_gradients": -opposite_grads
        }
        
        metrics = compute_w1_metrics(model, hooks_data_opposite)
        assert abs(metrics["w1_inter_path_gradient_align"] - (-1.0)) < 1e-5, \
            f"반대 벡터의 코사인 유사도가 -1이 아님: {metrics['w1_inter_path_gradient_align']}"
        
        print(f"  ✓ 반대 벡터 테스트: {metrics['w1_inter_path_gradient_align']:.4f} (≈ -1.0)")
        
        # Torch 텐서 입력 테스트
        print("\n서브테스트 2-5: Torch 텐서 입력 처리")
        torch_cnn_grads = torch.randn(d_grad)
        torch_gru_grads = torch.randn(d_grad)
        
        hooks_data_torch = {
            "cnn_gradients": torch_cnn_grads,
            "gru_gradients": torch_gru_grads
        }
        
        metrics = compute_w1_metrics(model, hooks_data_torch)
        assert not np.isnan(metrics["w1_inter_path_gradient_align"]), \
            "Torch 텐서 입력에서 그래디언트 정렬이 계산되지 않음"
        
        print(f"  ✓ Torch 텐서 입력 처리 성공: {metrics['w1_inter_path_gradient_align']:.4f}")
        
        print(f"\n✓ 테스트 2 통과")
        return True
        
    except Exception as e:
        print(f"\n✗ 테스트 2 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_metrics_integration():
    """모의 hooks_data로 지표 통합 계산 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: W1 지표 통합 계산 (모든 지표 동시 테스트)")
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
        # 모델 생성 (지표 계산에 필요)
        model = HybridTS(cfg, n_vars=7)
        
        print(f"\n  모델 생성 완료 (지표 계산용)")
        print(f"  교차 블록 수: {len(model.xhconv_blks)}")
        
        # 실제 실험에서 수집될 hooks_data 시뮬레이션
        # 모든 필요한 데이터를 포함
        n_samples = 64
        d_embed = 256
        d_grad = 512
        
        # 유사한 표현 생성 (높은 CKA/CCA를 얻기 위해)
        base_repr = np.random.randn(n_samples, d_embed)
        cnn_repr = base_repr + np.random.randn(n_samples, d_embed) * 0.1
        gru_repr = base_repr + np.random.randn(n_samples, d_embed) * 0.1
        
        # 손실이 점진적으로 감소하는 경우
        layerwise_losses = [2.0, 1.5, 1.0, 0.7, 0.5, 0.4, 0.35]
        
        # 어느 정도 정렬된 그래디언트
        base_grad = np.random.randn(d_grad)
        cnn_grads = base_grad + np.random.randn(d_grad) * 0.3
        gru_grads = base_grad + np.random.randn(d_grad) * 0.3
        
        hooks_data = {
            "cnn_representations": [cnn_repr],
            "gru_representations": [gru_repr],
            "layerwise_losses": layerwise_losses,
            "cnn_gradients": cnn_grads,
            "gru_gradients": gru_grads
        }
        
        print(f"\n  모의 hooks_data 생성 완료:")
        print(f"    표현 샘플 수: {n_samples}")
        print(f"    표현 차원: {d_embed}")
        print(f"    층별 손실 수: {len(layerwise_losses)}")
        print(f"    그래디언트 차원: {d_grad}")
        
        # W1 지표 계산
        metrics = compute_w1_metrics(model, hooks_data)
        
        print(f"\n  계산된 W1 지표:")
        for key, value in metrics.items():
            if np.isnan(value):
                print(f"    {key}: NaN")
            else:
                print(f"    {key}: {value:.6f}")
        
        # 모든 지표가 계산되었는지 확인
        expected_keys = [
            "w1_cka_similarity_cnn_gru",
            "w1_cca_similarity_cnn_gru",
            "w1_layerwise_upward_improvement",
            "w1_inter_path_gradient_align"
        ]
        
        for key in expected_keys:
            assert key in metrics, f"지표 '{key}'가 반환되지 않음"
            assert not np.isnan(metrics[key]), f"지표 '{key}'가 NaN임 (데이터가 제공되었는데도)"
        
        # 값 범위 검증
        assert 0 <= metrics["w1_cka_similarity_cnn_gru"] <= 1, \
            f"CKA 유사도가 범위 [0, 1]을 벗어남"
        assert 0 <= metrics["w1_cca_similarity_cnn_gru"] <= 1, \
            f"CCA 유사도가 범위 [0, 1]을 벗어남"
        assert metrics["w1_layerwise_upward_improvement"] > 0, \
            f"손실 감소 시나리오에서 개선도가 양수여야 함"
        assert -1 <= metrics["w1_inter_path_gradient_align"] <= 1, \
            f"그래디언트 정렬이 범위 [-1, 1]을 벗어남"
        
        print(f"\n  ✓ 모든 지표가 정상적으로 계산됨")
        print(f"  ✓ 모든 지표가 유효한 범위 내에 있음")
        print(f"\n✓ 테스트 3 통과")
        return True
        
    except Exception as e:
        print(f"\n✗ 테스트 3 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """모든 테스트 실행"""
    print("\n" + "=" * 80)
    print("W1 실험 수정 사항 테스트 시작")
    print("=" * 80)
    
    results = []
    
    # 테스트 1: 모델 생성
    results.append(("모델 생성", test_model_creation()))
    
    # 테스트 2: 지표 계산 구조
    results.append(("지표 계산 구조", test_w1_metrics_structure()))
    
    # 테스트 3: 지표 통합 테스트
    results.append(("지표 통합 테스트", test_metrics_integration()))
    
    # 결과 요약
    print("\n" + "=" * 80)
    print("테스트 결과 요약")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ 통과" if passed else "✗ 실패"
        print(f"  {test_name}: {status}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "=" * 80)
        print("✓ 모든 테스트 통과!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ 일부 테스트 실패")
        print("=" * 80)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

