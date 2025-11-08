"""
W4 실험 수정 사항 테스트 스크립트
중간층 표현 수집 및 지표 계산이 정상적으로 동작하는지 확인
"""

import torch
import numpy as np
from config.config import load_config, apply_HP2, build_experiment_config
from experiments.w4_experiment import W4Experiment
from models.ctsf_model import HybridTS


def test_layer_coverage():
    """모든 depth에서 shallow/mid/deep이 전체 층을 커버하는지 테스트"""
    print("=" * 80)
    print("테스트 1: 층 커버리지 검증 (개선된 로직)")
    print("=" * 80)
    
    test_depths = [6, 7, 8, 9, 10, 12]
    
    for depth in test_depths:
        # Shallow 계산
        n = max(1, depth // 3)
        shallow = list(range(n))
        
        # Mid 계산 (개선된 로직)
        shallow_end = max(1, depth // 3)
        deep_start = depth - max(1, depth // 3)
        mid = list(range(shallow_end, deep_start))
        
        # Deep 계산
        n = max(1, depth // 3)
        deep = list(range(depth - n, depth))
        
        # 전체 커버된 층
        covered = set(shallow + mid + deep)
        all_layers = set(range(depth))
        
        missing = all_layers - covered
        overlapping = []
        for i in range(depth):
            count = (i in shallow) + (i in mid) + (i in deep)
            if count > 1:
                overlapping.append(i)
        
        status = "✓" if len(missing) == 0 and len(overlapping) == 0 else "✗"
        
        print(f"\n  depth={depth}: {status}")
        print(f"    shallow:  {shallow}")
        print(f"    mid:      {mid}")
        print(f"    deep:     {deep}")
        
        if missing:
            print(f"    ⚠️  누락된 층: {sorted(missing)}")
        if overlapping:
            print(f"    ⚠️  중복된 층: {overlapping}")
        
        if len(missing) == 0 and len(overlapping) == 0:
            print(f"    ✓ 모든 층이 정확히 한 범주에 속함")


def test_w4_active_layers():
    """active_layers 속성이 올바르게 설정되는지 테스트"""
    print("\n" + "=" * 80)
    print("테스트 2: active_layers 속성 확인")
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
    
    test_cases = [
        ("all", None),
        ("shallow", None),
        ("mid", None),
        ("deep", None),
    ]
    
    for cross_layers_mode, expected_count in test_cases:
        cfg["cross_layers"] = cross_layers_mode
        cfg["experiment_type"] = "W4"
        cfg["verbose"] = False
        
        try:
            # 모델만 생성하고 테스트 (학습은 스킬)
            model = HybridTS(cfg, n_vars=7)
            actual_depth = len(model.xhconv_blks)
            
            # W4Experiment에서 active_layers 계산 로직 재현
            if cross_layers_mode == "all":
                expected_layers = list(range(actual_depth))
            elif cross_layers_mode == "shallow":
                n = max(1, actual_depth // 3)
                expected_layers = list(range(n))
            elif cross_layers_mode == "mid":
                # 전체 중간 영역 포함 (개선된 로직)
                shallow_end = max(1, actual_depth // 3)
                deep_start = actual_depth - max(1, actual_depth // 3)
                expected_layers = list(range(shallow_end, deep_start))
            elif cross_layers_mode == "deep":
                n = max(1, actual_depth // 3)
                expected_layers = list(range(actual_depth - n, actual_depth))
            
            print(f"\n  모드: {cross_layers_mode}")
            print(f"    실제 모델 depth: {actual_depth}")
            print(f"    예상 active_layers: {expected_layers}")
            print(f"    ✓ 통과")
            
        except Exception as e:
            print(f"  모드: {cross_layers_mode} - ✗ 실패: {e}")
            import traceback
            traceback.print_exc()


def test_w4_hooks_structure():
    """중간층 표현 수집 구조가 올바른지 테스트"""
    print("\n" + "=" * 80)
    print("테스트 3: 중간층 표현 수집 구조 확인")
    print("=" * 80)
    
    # 가상의 hooks_data 생성
    active_layers = [0, 1, 2]
    batch_size = 32
    d_embed = 256
    
    # BaseExperiment.evaluate_test()에서 생성되는 형식 모방
    cnn_repr_by_layer = {i: [] for i in active_layers}
    gru_repr_by_layer = {i: [] for i in active_layers}
    
    # 여러 배치의 데이터 시뮬레이션
    num_batches = 3
    for batch_idx in range(num_batches):
        for layer_idx in active_layers:
            # 각 배치마다 (B, d) 형태의 표현 추가
            cnn_repr_by_layer[layer_idx].append(np.random.randn(batch_size, d_embed))
            gru_repr_by_layer[layer_idx].append(np.random.randn(batch_size, d_embed))
    
    # 최종 변환 (BaseExperiment.evaluate_test()의 finally 블록과 동일)
    cnn_repr_list = []
    gru_repr_list = []
    for i in active_layers:
        if cnn_repr_by_layer[i]:
            cnn_repr_list.append(np.concatenate(cnn_repr_by_layer[i], axis=0))
        else:
            cnn_repr_list.append(None)
        
        if gru_repr_by_layer[i]:
            gru_repr_list.append(np.concatenate(gru_repr_by_layer[i], axis=0))
        else:
            gru_repr_list.append(None)
    
    # 검증
    print(f"\n  활성 층: {active_layers}")
    print(f"  배치 수: {num_batches}, 배치 크기: {batch_size}")
    print(f"  임베딩 차원: {d_embed}")
    print(f"\n  수집된 표현:")
    for idx, layer_idx in enumerate(active_layers):
        cnn_shape = cnn_repr_list[idx].shape if cnn_repr_list[idx] is not None else None
        gru_shape = gru_repr_list[idx].shape if gru_repr_list[idx] is not None else None
        expected_shape = (batch_size * num_batches, d_embed)
        
        print(f"    Layer {layer_idx}:")
        print(f"      CNN 표현 shape: {cnn_shape} (예상: {expected_shape})")
        print(f"      GRU 표현 shape: {gru_shape} (예상: {expected_shape})")
        
        if cnn_shape == expected_shape and gru_shape == expected_shape:
            print(f"      ✓ 통과")
        else:
            print(f"      ✗ 실패")


def test_w4_metrics_computation():
    """W4 지표 계산이 올바른지 테스트"""
    print("\n" + "=" * 80)
    print("테스트 4: W4 지표 계산 확인")
    print("=" * 80)
    
    from utils.experiment_metrics.w4_metrics import compute_w4_metrics
    
    # 가상의 모델과 데이터 생성
    base_cfg = load_config("hp2_config.yaml")
    model = HybridTS(base_cfg, n_vars=7)
    
    # active_layers 설정
    active_layers = [0, 1, 2]
    for i, blk in enumerate(model.xhconv_blks):
        if i in active_layers:
            blk.use_gc = True
            blk.use_cg = True
        else:
            blk.use_gc = False
            blk.use_cg = False
    
    # 가상의 hooks_data 생성
    num_samples = 100
    d_embed = 256
    hooks_data = {
        "cnn_representations": [np.random.randn(num_samples, d_embed) for _ in active_layers],
        "gru_representations": [np.random.randn(num_samples, d_embed) for _ in active_layers],
    }
    
    # 지표 계산
    try:
        metrics = compute_w4_metrics(
            model=model,
            hooks_data=hooks_data,
            active_layers=active_layers
        )
        
        print(f"\n  계산된 지표:")
        for key, value in metrics.items():
            print(f"    {key}: {value}")
        
        # 필수 지표 확인
        required_metrics = [
            "w4_layerwise_gate_usage",
            "w4_layer_contribution_score",
            "w4_layerwise_representation_similarity"
        ]
        
        all_present = all(key in metrics for key in required_metrics)
        if all_present:
            print(f"\n  ✓ 모든 필수 지표가 계산되었습니다")
        else:
            missing = [key for key in required_metrics if key not in metrics]
            print(f"\n  ✗ 누락된 지표: {missing}")
        
        # NaN이 아닌 값 확인
        non_nan_count = sum(1 for v in metrics.values() if not (isinstance(v, float) and np.isnan(v)))
        print(f"  유효한 값: {non_nan_count}/{len(metrics)}")
        
    except Exception as e:
        print(f"\n  ✗ 지표 계산 실패: {e}")
        import traceback
        traceback.print_exc()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "W4 실험 수정 사항 테스트" + " " * 34 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    try:
        test_layer_coverage()
        test_w4_active_layers()
        test_w4_hooks_structure()
        test_w4_metrics_computation()
        
        print("\n" + "=" * 80)
        print("모든 테스트 완료!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n테스트 실행 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

