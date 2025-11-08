"""
단일 실험 테스트 스크립트
각 실험(W1~W5)의 각 모드별로 최소 설정으로 테스트 실행
"""

import time
import warnings
from pathlib import Path
from config.config import load_config, apply_HP2
from experiments.w1_experiment import W1Experiment
from experiments.w2_experiment import W2Experiment
from experiments.w3_experiment import W3Experiment
from experiments.w4_experiment import W4Experiment
from experiments.w5_experiment import W5Experiment
from utils.error_logger import log_experiment_error
import traceback

# numpy 경고 억제 (표준편차가 0일 때 나누기 경고)
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in divide')
warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')


def test_single_experiment(experiment_type, mode, dataset="ETTh1", horizon=96, seed=42):
    """
    단일 실험 테스트
    
    Args:
        experiment_type: 'W1', 'W2', 'W3', 'W4', 'W5'
        mode: 실험별 모드
        dataset: 테스트용 데이터셋 (ETTh1 - 가장 빠름)
        horizon: 테스트용 horizon (96 - 가장 빠름)
        seed: 테스트용 시드 (42)
    """
    print(f"\n{'='*80}")
    print(f"테스트: {experiment_type} - {mode}")
    print(f"{'='*80}")
    
    # 설정 로드
    base_cfg = load_config("hp2_config.yaml")
    
    # 데이터셋 경로 찾기
    csv_path = None
    for p in [Path("datasets") / f"{dataset}.csv", Path("/mnt/data") / f"{dataset}.csv"]:
        if p.exists():
            csv_path = str(p)
            break
    
    if csv_path is None:
        print(f"❌ 데이터셋 {dataset}을 찾을 수 없습니다.")
        return False
    
    # HP2 설정 적용
    cfg = apply_HP2(
        base_cfg,
        csv_path=csv_path,
        seed=seed,
        horizon=horizon,
        device="cuda",
        out_root="results_test",
        model_tag="HyperConv"
    )
    
    # 테스트용으로 epochs를 줄임 (빠른 테스트)
    cfg["epochs"] = 5  # 테스트용 (원래는 100)
    cfg["early_stop_patience"] = 3  # 테스트용 (원래는 20)
    
    # 실험별 설정
    cfg["experiment_type"] = experiment_type
    cfg["mode"] = mode
    cfg["verbose"] = True
    
    if experiment_type == "W3":
        cfg["perturbation"] = mode if mode != "none" else None
        cfg["perturbation_kwargs"] = {}
    elif experiment_type == "W4":
        cfg["cross_layers"] = mode
    elif experiment_type == "W5":
        cfg["gate_fixed"] = (mode == "fixed")
    
    # 실험 실행
    start_time = time.time()
    try:
        if experiment_type == "W1":
            exp = W1Experiment(cfg)
        elif experiment_type == "W2":
            exp = W2Experiment(cfg)
        elif experiment_type == "W3":
            exp = W3Experiment(cfg)
        elif experiment_type == "W4":
            exp = W4Experiment(cfg)
        elif experiment_type == "W5":
            exp = W5Experiment(cfg)
        else:
            raise ValueError(f"Unknown experiment_type: {experiment_type}")
        
        exp.run()
        elapsed = time.time() - start_time
        
        print(f"\n✅ 성공 - 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_tb = traceback.format_exc()
        
        # 오류 로깅
        log_experiment_error(
            experiment_type=experiment_type,
            dataset=dataset,
            horizon=horizon,
            seed=seed,
            mode=mode,
            error_message=str(e),
            error_traceback=error_tb,
            results_root="results_test"
        )
        
        print(f"\n❌ 실패 - 소요 시간: {elapsed/60:.1f}분 ({elapsed:.1f}초)")
        print(f"오류: {e}")
        print(f"트레이스백:\n{error_tb}")
        return False


def format_time(seconds):
    """시간을 보기 좋게 포맷 (d일 h시간 m분)"""
    if seconds < 60:
        return f"{seconds:.0f}초"
    elif seconds < 3600:
        return f"{seconds/60:.1f}분"
    elif seconds < 86400:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}시간 {mins}분"
    else:
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        return f"{days}일 {hours}시간"


def main():
    """모든 실험 테스트"""
    print("=" * 80)
    print("CTSF-W 실험 단일 테스트")
    print("=" * 80)
    print("설정: dataset=ETTh1, horizon=96, seed=42")
    print("테스트용 epochs=5 (빠른 검증)")
    print("=" * 80)
    
    total_start = time.time()
    results = {}
    test_plan = [
        ("W1", ["per_layer", "last_layer"]),
        ("W2", ["dynamic", "static"]),
        ("W3", ["none", "tod_shift", "smooth"]),
        ("W4", ["all", "shallow", "mid", "deep"]),
        ("W5", ["dynamic", "fixed"]),
    ]
    
    total_tests = sum(len(modes) for _, modes in test_plan)
    completed_tests = 0
    
    for exp_type, modes in test_plan:
        print(f"\n[{exp_type} 실험 테스트]")
        results[exp_type] = {}
        for mode in modes:
            completed_tests += 1
            # 진행 상황 및 예상 시간 출력
            elapsed = time.time() - total_start
            if completed_tests > 1:
                avg_time = elapsed / (completed_tests - 1)
                remaining = total_tests - (completed_tests - 1)
                eta = avg_time * remaining
                eta_str = format_time(eta)
            else:
                eta_str = "계산 중..."
            
            print(f"\n[{completed_tests}/{total_tests}] {exp_type}-{mode}")
            print(f"   경과 시간: {format_time(elapsed)} | 예상 남은 시간: {eta_str}")
            results[exp_type][mode] = test_single_experiment(exp_type, mode)
            time.sleep(1)  # 간단한 대기
    
    total_time = time.time() - total_start
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("테스트 완료 요약")
    print("=" * 80)
    print(f"총 소요 시간: {total_time/3600:.2f}시간 ({total_time:.1f}초)")
    print("\n실험별 결과:")
    
    all_passed = True
    for exp_type, modes in results.items():
        passed = sum(1 for v in modes.values() if v)
        total = len(modes)
        status = "✅" if passed == total else "⚠️"
        print(f"  {status} {exp_type}: {passed}/{total} 성공")
        if passed != total:
            all_passed = False
            for mode, success in modes.items():
                if not success:
                    print(f"      - {mode}: 실패")
    
    if all_passed:
        print("\n✅ 모든 테스트가 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️  일부 테스트가 실패했습니다. results_test/errors_*.json을 확인하세요.")
    
    print("=" * 80)


if __name__ == "__main__":
    main()
