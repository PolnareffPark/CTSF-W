"""
실험 스위트 실행기
여러 실험을 자동으로 순차 실행 (resume 기능 포함)
"""

import argparse
import time
import traceback
from pathlib import Path
from config.config import load_config, apply_HP2
from experiments.w1_experiment import W1Experiment
from experiments.w2_experiment import W2Experiment
from experiments.w3_experiment import W3Experiment
from experiments.w4_experiment import W4Experiment
from experiments.w5_experiment import W5Experiment
from utils.csv_logger import read_results_rows
from utils.error_logger import log_experiment_error


def _norm_str(x):
    return str(x).strip()
    

def _csv_path_or_mnt(name, root_csv="datasets"):
    p1 = Path(root_csv) / f"{name}.csv"
    p2 = Path("/mnt/data") / f"{name}.csv"
    return str(p1 if p1.exists() else p2)


def _in_filter(key, datasets, horizons, seeds, modes):
    ds, H, s, md = key
    return (ds in set(map(_norm_str, datasets)) and
            H in set(map(int, horizons)) and
            s in set(map(int, seeds)) and
            md in set(map(_norm_str, modes)))


def run_experiment_suite(
    experiment_type,
    seeds=(42, 2, 3, 5, 7, 11, 13, 17, 19, 23),
    horizons=(96, 192, 336, 720),
    datasets=("ETTm1", "ETTm2", "ETTh1", "ETTh2", "weather"),
    modes=None,  # 실험별 모드 리스트
    root_csv="datasets",
    results_root="results",
    resume_mode="next",  # 'next' | 'fill_missing' | 'all'
    overwrite=False,
    max_jobs=None,
    device="cuda",
    config_path="hp2_config.yaml",
    dry_run=False,
    verbose=True
):
    """
    실험 스위트 실행
    
    Args:
        experiment_type: 'W1', 'W2', 'W3', 'W4', 'W5'
        seeds: 시드 리스트
        horizons: horizon 리스트
        datasets: 데이터셋 리스트
        modes: 실험별 모드 리스트 (예: W1의 경우 ['per_layer', 'last_layer'])
        results_root: 결과 저장 루트
        resume_mode: resume 모드
        overwrite: 완료된 실험도 재실행할지
        max_jobs: 최대 실행 개수
        device: 디바이스
        config_path: 설정 파일 경로
        dry_run: 계획만 출력
    """
    # 기본 모드 설정
    if modes is None:
        if experiment_type == "W1":
            modes = ["per_layer", "last_layer"]
        elif experiment_type == "W2":
            modes = ["dynamic", "static"]
        elif experiment_type == "W3":
            modes = ["none", "tod_shift", "smooth"]
        elif experiment_type == "W4":
            modes = ["all", "shallow", "mid", "deep"]
        elif experiment_type == "W5":
            # W5는 한 번 실행으로 동적/고정 비교를 모두 수행함
            modes = ["dynamic"]
        else:
            modes = ["default"]
    
    # 결과 파일 읽기 (실험별로 분리)
    rows, done_set_all = read_results_rows(results_root, experiment_type=experiment_type)
    
    # 필터 적용
    datasets = tuple(map(_norm_str, datasets))
    horizons = tuple(map(int, horizons))
    seeds = tuple(map(int, seeds))
    modes = tuple(map(_norm_str, modes))
    
    grid_keys = set(
        (_norm_str(ds), int(H), int(s), _norm_str(md))
        for ds in datasets for H in horizons for s in seeds for md in modes
    )
    done_set = {k for k in done_set_all if k in grid_keys}
    
    # 마지막 완료 조합 찾기
    last_key = None
    for key in reversed(rows):
        if key in grid_keys:
            last_key = key
            break
    
    # 실행 계획 만들기
    plan = []
    
    def _append_if_needed(ds, H, s, md):
        key = (_norm_str(ds), int(H), int(s), _norm_str(md))
        if (not overwrite) and (key in done_set):
            return
        plan.append((ds, H, s, md))
    
    if resume_mode == "next":
        # 마지막 완료 조합 이후부터 시작
        if (last_key is None) or (not _in_filter(last_key, datasets, horizons, seeds, modes)):
            # 처음부터 시작
            for ds in datasets:
                for H in horizons:
                    for s in seeds:
                        for md in modes:
                            _append_if_needed(ds, H, s, md)
        else:
            # 마지막 조합 이후부터
            ds_last, H_last, s_last, md_last = last_key
            started = False
            for ds in datasets:
                for H in horizons:
                    for s in seeds:
                        for md in modes:
                            if not started:
                                if (ds == ds_last and H == H_last and s == s_last and md == md_last):
                                    started = True
                                    continue
                            if started:
                                _append_if_needed(ds, H, s, md)
    
    elif resume_mode == "fill_missing":
        # 누락된 것만 실행
        for ds in datasets:
            for H in horizons:
                for s in seeds:
                    for md in modes:
                        _append_if_needed(ds, H, s, md)
    
    elif resume_mode == "all":
        # 모두 실행
        for ds in datasets:
            for H in horizons:
                for s in seeds:
                    for md in modes:
                        plan.append((ds, H, s, md))
    
    if max_jobs is not None:
        plan = plan[:int(max_jobs)]
    
    # 계획 출력
    print(f"[plan] experiment={experiment_type} | mode={resume_mode} | overwrite={overwrite}")
    print(f"[plan] grid={len(grid_keys)} | done={len(done_set)} | to_run={len(plan)} | last={last_key}")
    
    if dry_run:
        print("[dry_run] 계획만 출력하고 종료합니다.")
        return
    
    # 설정 로드
    base_cfg = load_config(config_path)
    
    # 시간 추적
    suite_start_time = time.time()
    completed = 0
    total_jobs = len(plan)
    
    # 실행
    for job_idx, (ds, H, s, md) in enumerate(plan, 1):
        try:
            cfg = apply_HP2(
                base_cfg,
                csv_path=_csv_path_or_mnt(ds, root_csv=root_csv),
                seed=s,
                horizon=H,
                device=device if device else base_cfg.get("device", "cuda"),
                out_root=results_root,
                model_tag=base_cfg.get("model_tag", "HyperConv")
            )
            
            # 실험별 모드 설정
            cfg["mode"] = md
            cfg["experiment_type"] = experiment_type
            cfg["verbose"] = False  # tqdm 억제 (진행률은 외부에서 표시)
            if experiment_type == "W3":
                cfg["perturbation"] = md if md != "none" else None
                cfg["perturbation_kwargs"] = {}  # 필요시 추가
            elif experiment_type == "W4":
                cfg["cross_layers"] = md
            elif experiment_type == "W5":
                cfg["gate_fixed"] = (md == "fixed")
            
            # 실험 실행 (내부 출력 억제됨)
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
            completed += 1
            
            # 완료 후 진행률 표시 (한 줄로 업데이트)
            elapsed = time.time() - suite_start_time
            if completed > 0:
                avg_time = elapsed / completed
                remaining = total_jobs - completed
                eta = avg_time * remaining
                
                if eta < 60:
                    eta_str = f"{eta:.0f}초"
                elif eta < 3600:
                    eta_str = f"{eta/60:.1f}분"
                elif eta < 86400:
                    hours = int(eta // 3600)
                    mins = int((eta % 3600) // 60)
                    eta_str = f"{hours}시간 {mins}분"
                else:
                    days = int(eta // 86400)
                    hours = int((eta % 86400) // 3600)
                    eta_str = f"{days}일 {hours}시간"
                
                elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
            else:
                eta_str = "계산 중..."
                elapsed_str = "0m"
            
            progress_pct = (completed / total_jobs) * 100
            print(f"진행: {completed}/{total_jobs} ({progress_pct:.1f}%) | 경과: {elapsed_str} | 남은시간: {eta_str} | {experiment_type} {ds} H{H} s{s} {md} ✓", flush=True)
        
        except Exception as e:
            # 오류 로깅
            error_tb = traceback.format_exc()
            log_experiment_error(
                experiment_type=experiment_type,
                dataset=ds,
                horizon=H,
                seed=s,
                mode=md,
                error_message=str(e),
                error_traceback=error_tb,
                results_root=results_root
            )
            
            # 실패 상태 표시
            elapsed = time.time() - suite_start_time
            elapsed_str = f"{elapsed/3600:.1f}h" if elapsed > 3600 else f"{elapsed/60:.1f}m"
            print(f"진행: {completed}/{total_jobs} | 경과: {elapsed_str} | {experiment_type} {ds} H{H} s{s} {md} ✗ ({str(e)[:50]})", flush=True)
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTSF-W 실험 스위트 실행")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["W1", "W2", "W3", "W4", "W5"],
                        help="실험 타입")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2, 3, 5, 7, 11, 13, 17, 19, 23],
                        help="시드 리스트")
    parser.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720],
                        help="horizon 리스트")
    parser.add_argument("--datasets", type=str, nargs="+",
                        default=["ETTm1", "ETTm2", "ETTh1", "ETTh2", "weather"],
                        help="데이터셋 리스트")
    parser.add_argument("--modes", type=str, nargs="+", default=None,
                        help="실험별 모드 리스트")
    parser.add_argument("--resume", type=str, default="next",
                        choices=["next", "fill_missing", "all"],
                        help="resume 모드")
    parser.add_argument("--overwrite", action="store_true",
                        help="완료된 실험도 재실행")
    parser.add_argument("--max_jobs", type=int, default=None,
                        help="최대 실행 개수")
    parser.add_argument("--results_root", type=str, default="results",
                        help="결과 저장 루트")
    parser.add_argument("--config", type=str, default="hp2_config.yaml",
                        help="설정 파일 경로")
    parser.add_argument("--dry_run", action="store_true",
                        help="계획만 출력")
    parser.add_argument("--verbose", action="store_true", default=True,
                        help="상세 로그 출력")
    
    args = parser.parse_args()
    
    run_experiment_suite(
        experiment_type=args.experiment,
        seeds=args.seeds,
        horizons=args.horizons,
        datasets=args.datasets,
        modes=args.modes,
        results_root=args.results_root,
        resume_mode=args.resume,
        overwrite=args.overwrite,
        max_jobs=args.max_jobs,
        config_path=args.config,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
