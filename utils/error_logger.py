"""
실험 실패 정보 로깅 모듈
"""

import json
from pathlib import Path
from datetime import datetime


def log_experiment_error(
    experiment_type,
    dataset,
    horizon,
    seed,
    mode,
    error_message,
    error_traceback,
    results_root="results"
):
    """
    실험 실패 정보를 JSON 파일에 저장
    
    Args:
        experiment_type: 실험 타입 (W1, W2, ...)
        dataset: 데이터셋 이름
        horizon: horizon 값
        seed: 시드 값
        mode: 실험 모드
        error_message: 오류 메시지
        error_traceback: 오류 트레이스백
        results_root: 결과 저장 루트
    """
    error_log_path = Path(results_root) / f"errors_{experiment_type}.json"
    
    error_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_type": experiment_type,
        "dataset": dataset,
        "horizon": int(horizon),
        "seed": int(seed),
        "mode": mode,
        "error_message": str(error_message),
        "error_traceback": str(error_traceback)
    }
    
    # 기존 오류 로그 읽기
    if error_log_path.exists():
        with open(error_log_path, 'r', encoding='utf-8') as f:
            errors = json.load(f)
    else:
        errors = []
    
    # 새 오류 추가
    errors.append(error_entry)
    
    # 저장
    with open(error_log_path, 'w', encoding='utf-8') as f:
        json.dump(errors, f, indent=2, ensure_ascii=False)
    
    return str(error_log_path)


def read_experiment_errors(results_root="results", experiment_type=None):
    """
    실험 오류 로그 읽기
    
    Args:
        results_root: 결과 루트 디렉토리
        experiment_type: 실험 타입 (None이면 모든 실험)
    
    Returns:
        오류 리스트
    """
    errors = []
    
    if experiment_type:
        files = [Path(results_root) / f"errors_{experiment_type}.json"]
    else:
        files = list(Path(results_root).glob("errors_W*.json"))
    
    for f in files:
        if f.exists():
            with open(f, 'r', encoding='utf-8') as file:
                errors.extend(json.load(file))
    
    return errors
