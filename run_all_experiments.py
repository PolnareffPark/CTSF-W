"""
전체 실험 (W1~W5) 순차 실행
각 실험을 독립적으로 실행하여 하나가 실패해도 나머지 계속 진행
"""

import argparse
import time
from run_suite import run_experiment_suite
from config.config import load_config
from utils.error_logger import read_experiment_errors


def run_all_experiments(
    seeds=(42, 2, 3, 5, 7, 11, 13, 17, 19, 23),
    horizons=(96, 192, 336, 720),
    datasets=("ETTm1", "ETTm2", "ETTh1", "ETTh2", "weather"),
    results_root="results",
    resume_mode="next",
    overwrite=False,
    max_jobs_per_exp=None,
    device="cuda",
    config_path="hp2_config.yaml",
    dry_run=False,
    verbose=False
):
    """
    모든 실험 (W1~W5) 순차 실행
    
    Args:
        verbose: False면 간단한 진행 상황만 출력
    """
    experiments = ["W1", "W2", "W3", "W4", "W5"]
    total_start = time.time()
    exp_times = {}
    
    print("=" * 80)
    print(f"전체 실험 시작: {len(experiments)}개 실험")
    print("=" * 80)
    
    for exp_type in experiments:
        exp_start = time.time()
        print(f"\n{'='*80}")
        print(f"[{exp_type}] 실험 시작")
        print(f"{'='*80}")
        
        try:
            # 각 실험별 기본 모드 설정
            if exp_type == "W1":
                modes = ["per_layer", "last_layer"]
            elif exp_type == "W2":
                modes = ["dynamic", "static"]
            elif exp_type == "W3":
                modes = ["none", "tod_shift", "smooth"]
            elif exp_type == "W4":
                modes = ["all", "shallow", "mid", "deep"]
            elif exp_type == "W5":
                modes = ["dynamic", "fixed"]
            else:
                modes = ["default"]
            
            run_experiment_suite(
                experiment_type=exp_type,
                seeds=seeds,
                horizons=horizons,
                datasets=datasets,
                modes=modes,
                results_root=results_root,
                resume_mode=resume_mode,
                overwrite=overwrite,
                max_jobs=max_jobs_per_exp,
                device=device,
                config_path=config_path,
                dry_run=dry_run,
                verbose=verbose
            )
            
            exp_time = time.time() - exp_start
            exp_times[exp_type] = exp_time
            print(f"\n[{exp_type}] 완료 - 소요 시간: {exp_time/3600:.2f}시간 ({exp_time:.1f}초)")
            
        except Exception as e:
            exp_time = time.time() - exp_start
            exp_times[exp_type] = exp_time
            print(f"\n[{exp_type}] 실패 - 소요 시간: {exp_time/3600:.2f}시간 ({exp_time:.1f}초)")
            print(f"오류: {e}")
            import traceback
            traceback.print_exc()
            # 다음 실험 계속 진행
            continue
    
    total_time = time.time() - total_start
    
    # 최종 요약
    print("\n" + "=" * 80)
    print("전체 실험 완료 요약")
    print("=" * 80)
    print(f"총 소요 시간: {total_time/3600:.2f}시간 ({total_time:.1f}초)")
    print("\n실험별 소요 시간:")
    for exp_type, exp_time in exp_times.items():
        print(f"  {exp_type}: {exp_time/3600:.2f}시간 ({exp_time:.1f}초)")
    
    # 오류 요약
    all_errors = read_experiment_errors(results_root=results_root)
    if all_errors:
        print(f"\n⚠️  실패한 실험: {len(all_errors)}개")
        print("상세 내용은 results/errors_W*.json 파일을 확인하세요.")
        for err in all_errors[-5:]:  # 최근 5개만 출력
            print(f"  - {err['experiment_type']} | {err['dataset']} | H={err['horizon']} | s={err['seed']} | {err['mode']}")
    else:
        print("\n✅ 모든 실험이 성공적으로 완료되었습니다.")
    
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTSF-W 전체 실험 실행")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2, 3, 5, 7, 11, 13, 17, 19, 23],
                        help="시드 리스트")
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720],
                        help="horizon 리스트")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["ETTm1", "ETTm2", "ETTh1", "ETTh2", "weather"],
                        help="데이터셋 리스트")
    parser.add_argument("--resume", type=str, default="next",
                        choices=["next", "fill_missing", "all"],
                        help="resume 모드")
    parser.add_argument("--overwrite", action="store_true",
                        help="완료된 실험도 재실행")
    parser.add_argument("--max_jobs_per_exp", type=int, default=None,
                        help="실험당 최대 실행 개수")
    parser.add_argument("--results_root", type=str, default="results",
                        help="결과 저장 루트")
    parser.add_argument("--config", type=str, default="hp2_config.yaml",
                        help="설정 파일 경로")
    parser.add_argument("--dry_run", action="store_true",
                        help="계획만 출력")
    parser.add_argument("--verbose", action="store_true",
                        help="상세 로그 출력")
    
    args = parser.parse_args()
    
    run_all_experiments(
        seeds=args.seeds,
        horizons=args.horizons,
        datasets=args.datasets,
        results_root=args.results_root,
        resume_mode=args.resume,
        overwrite=args.overwrite,
        max_jobs_per_exp=args.max_jobs_per_exp,
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
