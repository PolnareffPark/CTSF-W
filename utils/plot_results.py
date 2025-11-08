"""
실험 결과 저장 및 보고용 그림 데이터 관리
results/results_W1/dataset/plot_type/plot_summary.csv 형식으로 저장
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional


def save_plot_data(
    experiment_type: str,
    dataset: str,
    plot_type: str,
    plot_data: Dict,
    horizon: Optional[int],
    seed: Optional[int],
    mode: Optional[str],
    results_root: str = "results"
):
    """
    보고용 그림 데이터 저장
    
    Args:
        experiment_type: 'W1', 'W2', 'W3', 'W4', 'W5'
        dataset: 데이터셋 이름 (예: 'ETTm2')
        plot_type: 그림 타입 (예: 'forest_plot', 'heatmap', 'histogram')
        plot_data: 그림 데이터 딕셔너리
        horizon: 예측 horizon
        seed: 실험 시드
        mode: 실험 모드
        results_root: 결과 루트 디렉토리
    """
    results_root = Path(results_root)
    plot_dir = results_root / f"results_{experiment_type}" / dataset / plot_type
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = plot_dir / f"{plot_type}_summary.csv"
    
    # 기존 데이터 읽기
    if csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame()
    
    # 필수 메타 컬럼 보강
    for col in ["experiment_type", "dataset", "plot_type", "horizon", "seed", "mode"]:
        if col not in df.columns:
            df[col] = pd.NA

    # 누락된 메타데이터 기본값 보정
    if not df.empty:
        df["horizon"] = df["horizon"].fillna(-1)
        df["seed"] = df["seed"].fillna(-1)
        df["mode"] = df["mode"].fillna("default").astype(str)
    
    # 새 데이터 구성
    new_row = {
        "experiment_type": experiment_type,
        "dataset": dataset,
        "plot_type": plot_type,
        "horizon": int(horizon) if horizon is not None else -1,
        "seed": int(seed) if seed is not None else -1,
        "mode": str(mode) if mode is not None else "default",
        **plot_data
    }
    new_df = pd.DataFrame([new_row])
    
    # 병합 및 중복 제거
    if df.empty:
        df = new_df
    else:
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.drop_duplicates(
            subset=["experiment_type", "dataset", "plot_type", "horizon", "seed", "mode"],
            keep="last"
        )
    
    # -1 값은 결측치로 되돌림
    df["horizon"] = df["horizon"].replace(-1, pd.NA)
    df["seed"] = df["seed"].replace(-1, pd.NA)
    
    # 열 순서 정리
    meta_cols = ["experiment_type", "dataset", "plot_type", "horizon", "seed", "mode"]
    other_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + other_cols]
    
    df.to_csv(csv_path, index=False)
    
    return str(csv_path)


def load_plot_data(
    experiment_type: str,
    dataset: str,
    plot_type: str,
    results_root: str = "results"
) -> Optional[pd.DataFrame]:
    """
    보고용 그림 데이터 로드
    
    Returns:
        DataFrame 또는 None
    """
    results_root = Path(results_root)
    csv_path = results_root / f"results_{experiment_type}" / dataset / plot_type / f"{plot_type}_summary.csv"
    
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None
