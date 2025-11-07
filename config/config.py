"""
설정 관리 모듈
YAML 설정 파일 로드 및 실험별 설정 생성
"""

import yaml
from pathlib import Path
import torch
import random
import numpy as np


def load_config(config_path="hp2_config.yaml"):
    """YAML 설정 파일 로드 및 평탄화"""
    with open(config_path, 'r', encoding='utf-8') as f:
        yaml_cfg = yaml.safe_load(f)
    
    # 평탄화 (중첩 구조를 단일 딕셔너리로)
    cfg = {}
    
    # general
    if "general" in yaml_cfg:
        cfg.update(yaml_cfg["general"])
    
    # data
    if "data" in yaml_cfg:
        cfg.update(yaml_cfg["data"])
    
    # dataloader
    if "dataloader" in yaml_cfg:
        for k, v in yaml_cfg["dataloader"].items():
            cfg[k] = v
    
    # model
    if "model" in yaml_cfg:
        for k, v in yaml_cfg["model"].items():
            if k == "alpha_init":
                # alpha_init 하위 구조 처리
                if isinstance(v, dict):
                    cfg["alpha_init_diag"] = v.get("diag", 0.90)
                    cfg["alpha_init_offdiag"] = v.get("offdiag", 0.05)
                continue
            elif k == "revin":
                # revin 하위 구조 처리
                if isinstance(v, dict):
                    cfg["revin_affine"] = v.get("affine", True)
                continue
            cfg[k] = v
    
    # loss
    if "loss" in yaml_cfg:
        cfg["lam_patch"] = yaml_cfg["loss"].get("lam_patch", 0.03)
    
    # optim
    if "optim" in yaml_cfg:
        cfg["lr"] = yaml_cfg["optim"].get("lr", 1e-4)
        cfg["weight_decay"] = yaml_cfg["optim"].get("weight_decay", 1e-3)
        cfg["grad_clip"] = yaml_cfg["optim"].get("grad_clip", 0.5)
    
    # train
    if "train" in yaml_cfg:
        cfg["epochs"] = yaml_cfg["train"].get("epochs", 100)
        if "early_stop" in yaml_cfg["train"]:
            cfg["early_stop_patience"] = yaml_cfg["train"]["early_stop"].get("patience", 20)
            cfg["early_stop_min_delta"] = yaml_cfg["train"]["early_stop"].get("min_delta", 1e-3)
    
    return cfg


def apply_HP2(base_cfg, csv_path, seed, horizon, device=None, out_root="results", model_tag=None):
    """
    HP2 설정 적용 및 실험별 파라미터 설정
    
    Args:
        base_cfg: 기본 설정 딕셔너리
        csv_path: 데이터셋 CSV 경로
        seed: 랜덤 시드
        horizon: 예측 horizon
        device: 디바이스
        out_root: 결과 저장 루트
        model_tag: 모델 태그
    """
    # 설정 업데이트
    cfg = base_cfg.copy()
    
    # 일반 설정
    cfg["seed"] = int(seed)
    cfg["csv_path"] = str(csv_path)
    cfg["horizon"] = int(horizon)
    cfg["lookback"] = cfg.get("lookback", 720)
    cfg["patch_len"] = cfg.get("patch_len", 36)
    cfg["out_dir"] = str(out_root)
    cfg["model_tag"] = (model_tag or cfg.get("model_tag", "HyperConv"))
    
    if device is not None:
        cfg["device"] = device
    
    # 시드 고정
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])
    
    # 재현성 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # GPU 설정
    if cfg.get("device", "cuda") == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
        # TF32 설정 (사용자 환경에 맞춤)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('highest')
    
    return cfg


def build_experiment_config(base_cfg, experiment_type, **kwargs):
    """
    실험별 설정 빌드
    
    Args:
        base_cfg: 기본 설정
        experiment_type: 'W1', 'W2', 'W3', 'W4', 'W5'
        **kwargs: 실험별 추가 파라미터
    """
    cfg = base_cfg.copy()
    cfg["experiment_type"] = experiment_type
    
    if experiment_type == "W1":
        cfg["mode"] = kwargs.get("mode", "per_layer")  # 'per_layer' or 'last_layer'
    elif experiment_type == "W2":
        cfg["mode"] = kwargs.get("mode", "dynamic")  # 'dynamic' or 'static'
    elif experiment_type == "W3":
        cfg["perturbation"] = kwargs.get("perturbation", None)  # 'tod_shift' or 'smooth'
    elif experiment_type == "W4":
        cfg["cross_layers"] = kwargs.get("cross_layers", "all")  # 'shallow', 'mid', 'deep', 'all'
    elif experiment_type == "W5":
        cfg["gate_fixed"] = kwargs.get("gate_fixed", False)
    
    return cfg
