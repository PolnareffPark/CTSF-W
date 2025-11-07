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
    results_root: str = "results"
):
    """
    보고용 그림 데이터 저장
    
    Args:
        experiment_type: 'W1', 'W2', 'W3', 'W4', 'W5'
        dataset: 데이터셋 이름 (예: 'ETTm2')
        plot_type: 그림 타입 (예: 'forest_plot', 'heatmap', 'histogram')
        plot_data: 그림 데이터 딕셔너리
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
    
    # 새 데이터 추가
    new_row = {
        "experiment_type": experiment_type,
        "dataset": dataset,
        "plot_type": plot_type,
        **plot_data
    }
    
    new_df = pd.DataFrame([new_row])
    
    # 중복 제거 (experiment_type, dataset, plot_type, 추가 키로)
    if not df.empty:
        # 간단히 append (실제로는 더 정교한 중복 제거 필요)
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df
    
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
