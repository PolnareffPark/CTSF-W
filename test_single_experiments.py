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
    cfg["verbose"] = False  # tqdm 억제
    
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
        return True
        
    except Exception as e:
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
        
        print(f"✗ 에러: {str(e)[:80]}")
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
    print("CTSF-W 통합 테스트 시작 (ETTh1, H=96, epochs=5)")
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
        results[exp_type] = {}
        for mode in modes:
            # 실험 실행 (내부 출력 억제됨)
            success = test_single_experiment(exp_type, mode)
            results[exp_type][mode] = success
            completed_tests += 1
            
            # 완료 후 진행률 표시 (한 줄로 업데이트)
            elapsed = time.time() - total_start
            if completed_tests > 0:
                avg_time = elapsed / completed_tests
                remaining = total_tests - completed_tests
                eta = avg_time * remaining
                eta_str = format_time(eta)
            else:
                eta_str = "계산 중..."
            
            progress_pct = (completed_tests / total_tests) * 100
            status = "✓" if success else "✗"
            
            # 완료 상태를 한 줄로 표시
            print(f"진행: {completed_tests}/{total_tests} ({progress_pct:.1f}%) | 경과: {format_time(elapsed)} | 남은시간: {eta_str} | {exp_type}-{mode} {status}", flush=True)
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("테스트 완료")
    print("=" * 80)
    
    all_passed = True
    for exp_type, modes in results.items():
        passed = sum(1 for v in modes.values() if v)
        total = len(modes)
        status = "✅" if passed == total else "❌"
        print(f"{status} {exp_type}: {passed}/{total}", end="")
        if passed != total:
            all_passed = False
            failed = [m for m, v in modes.items() if not v]
            print(f" (실패: {', '.join(failed)})")
        else:
            print()
    
    print("=" * 80)


if __name__ == "__main__":
    main()
