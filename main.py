"""
CTSF-W Ablation Study 메인 실행 파일
"""

import argparse
from pathlib import Path
from config.config import load_config, apply_HP2, build_experiment_config
from experiments.w1_experiment import W1Experiment
from experiments.w2_experiment import W2Experiment
from experiments.w3_experiment import W3Experiment
from experiments.w4_experiment import W4Experiment
from experiments.w5_experiment import W5Experiment


def main():
    parser = argparse.ArgumentParser(description="CTSF-W Ablation Study")
    parser.add_argument("--experiment", type=str, required=True,
                        choices=["W1", "W2", "W3", "W4", "W5"],
                        help="실험 타입")
    parser.add_argument("--dataset", type=str, required=True,
                        help="데이터셋 이름 (예: ETTh2)")
    parser.add_argument("--horizon", type=int, required=True,
                        help="예측 horizon (예: 192)")
    parser.add_argument("--seed", type=int, required=True,
                        help="랜덤 시드")
    parser.add_argument("--mode", type=str, default=None,
                        help="실험 모드 (실험별로 다름)")
    parser.add_argument("--config", type=str, default="hp2_config.yaml",
                        help="설정 파일 경로")
    parser.add_argument("--out_root", type=str, default="results",
                        help="결과 저장 루트")
    parser.add_argument("--device", type=str, default=None,
                        help="디바이스 (cuda/cpu)")
    
    args = parser.parse_args()
    
    # 설정 로드
    base_cfg = load_config(args.config)
    
    # 데이터셋 경로 찾기
    csv_path = _find_dataset_path(args.dataset)
    
    # HP2 설정 적용
    cfg = apply_HP2(
        base_cfg,
        csv_path=csv_path,
        seed=args.seed,
        horizon=args.horizon,
        device=args.device or base_cfg.get("device", "cuda"),
        out_root=args.out_root,
        model_tag=base_cfg.get("model_tag", "HyperConv")
    )
    
    # 실험별 설정
    exp_kwargs = {}
    if args.mode:
        exp_kwargs["mode"] = args.mode
    
    cfg = build_experiment_config(cfg, args.experiment, **exp_kwargs)
    
    # 실험 실행
    if args.experiment == "W1":
        exp = W1Experiment(cfg)
    elif args.experiment == "W2":
        exp = W2Experiment(cfg)
    elif args.experiment == "W3":
        exp = W3Experiment(cfg)
    elif args.experiment == "W4":
        exp = W4Experiment(cfg)
    elif args.experiment == "W5":
        exp = W5Experiment(cfg)
    
    exp.run()


def _find_dataset_path(dataset_name):
    """데이터셋 경로 찾기"""
    paths = [
        Path("datasets") / f"{dataset_name}.csv",
        Path("/mnt/data") / f"{dataset_name}.csv",
    ]
    for p in paths:
        if p.exists():
            return str(p)
    raise FileNotFoundError(f"Dataset {dataset_name} not found in {paths}")


if __name__ == "__main__":
    main()
