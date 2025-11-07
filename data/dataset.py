"""
데이터 로딩 및 전처리 모듈
CTSF-V1.py의 load_split_dataloaders를 모듈화
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class MVDataset(Dataset):
    """다변량 시계열 데이터셋"""
    def __init__(self, arr, L, H):
        self.arr = arr
        self.L = L
        self.H = H

    def __len__(self):
        return len(self.arr) - self.L - self.H + 1

    def __getitem__(self, idx):
        x = self.arr[idx: idx + self.L].T  # (C, L)
        y = self.arr[idx + self.L: idx + self.L + self.H].T  # (C, H)
        return torch.from_numpy(x), torch.from_numpy(y)


def load_split_dataloaders(cfg):
    """
    데이터 로딩 및 train/val/test 분할
    
    Args:
        cfg: 설정 딕셔너리 (csv_path, value_cols, train_ratio, val_ratio, 
             lookback, horizon, batch_size 포함)
    
    Returns:
        (train_loader, val_loader, test_loader, (mu, std, C))
        - mu, std: 표준화 통계 (C, 1)
        - C: 변수 수
    """
    df = pd.read_csv(cfg["csv_path"], encoding='utf-8')

    # 변수 선택
    if cfg.get("value_cols") is None:
        feature_cols = df.select_dtypes("number").columns.tolist()
    else:
        feature_cols = cfg["value_cols"]

    # 데이터 전처리
    data = (df[feature_cols]
            .interpolate("linear", limit_direction="both")
            .to_numpy(np.float32))  # (T, C)
    
    mu = data.mean(0, keepdims=True)
    std = data.std(0, keepdims=True) + 1e-6
    data = (data - mu) / std  # 표준화
    C = data.shape[1]  # 변수 수
    L, H = cfg["lookback"], cfg["horizon"]

    # 분할 (기존 CTSF-V1.py와 동일)
    total = len(data) - L - H + 1
    n_tr = int(total * cfg["train_ratio"])
    n_val = int(total * cfg["val_ratio"])
    margin = L + H
    tr_end = n_tr
    val_st = n_tr + margin
    val_end = val_st + n_val
    te_st = val_end + margin

    # DataLoader 생성 (기존과 동일)
    def mk_dataloader(sub, shuffle):
        return DataLoader(
            MVDataset(sub, L, H),
            batch_size=cfg.get("batch_size", 128),
            shuffle=shuffle,
            drop_last=cfg.get("drop_last", True),
            num_workers=cfg.get("num_workers", 4),
            pin_memory=cfg.get("pin_memory", True)
        )

    # 기존과 동일한 인덱싱
    train_loader = mk_dataloader(data[:tr_end + L + H - 1], True)
    val_loader = mk_dataloader(data[val_st:val_end + L + H - 1], False)
    test_loader = mk_dataloader(data[te_st:], False)

    return train_loader, val_loader, test_loader, (mu, std, C)


def apply_data_perturbation(data, perturbation_type, **kwargs):
    """
    데이터 교란 적용 (W3 실험용)
    
    Args:
        data: numpy array (T, C)
        perturbation_type: 'tod_shift' 또는 'smooth'
        **kwargs: 교란 파라미터
    
    Returns:
        교란된 데이터
    """
    if perturbation_type == 'tod_shift':
        # 시간대 시프트 (A4-TOD)
        shift_hours = kwargs.get('shift_hours', 1)
        points_per_hour = kwargs.get('points_per_hour', 4)  # 15분 간격 = 4 points/hour
        shift_points = shift_hours * points_per_hour
        
        # 배치별로 랜덤 시프트 적용 (예시)
        # 실제로는 학습 시 배치 단위로 적용하는 것이 더 적절
        return np.roll(data, shift_points, axis=0)
    
    elif perturbation_type == 'smooth':
        # 평활화 (A4-PEAK)
        from scipy.signal import savgol_filter
        window_length = kwargs.get('window_length', 5)
        polyorder = kwargs.get('polyorder', 2)
        
        smoothed = np.zeros_like(data)
        for c in range(data.shape[1]):
            smoothed[:, c] = savgol_filter(
                data[:, c], window_length, polyorder, mode='nearest'
            )
        return smoothed
    
    else:
        raise ValueError(f"Unknown perturbation_type: {perturbation_type}")


def find_datetime_col(df):
    """CSV에서 datetime 컬럼 찾기"""
    for c in df.columns:
        cl = str(c).lower()
        if any(k in cl for k in ["date", "time", "datetime"]):
            return c
    return None


def build_test_tod_vector(cfg):
    """
    테스트 세트 각 샘플의 예측 시작 시점에 대한 TOD [sin,cos] 라벨 생성
    
    Returns:
        (N_test, 2) numpy array 또는 None (datetime 컬럼이 없으면)
    """
    df = pd.read_csv(cfg["csv_path"])
    col = find_datetime_col(df)
    if col is None:
        return None

    ts = pd.to_datetime(df[col])
    sod = (ts.dt.hour.astype(np.int64) * 3600 +
           ts.dt.minute.astype(np.int64) * 60 +
           ts.dt.second.astype(np.int64)).to_numpy(dtype=np.float32)
    ang = 2 * np.pi * (sod / 86400.0)  # 24h 주기
    tod_all = np.stack([np.sin(ang), np.cos(ang)], axis=1)  # (T_all, 2)

    L, H = cfg["lookback"], cfg["horizon"]
    total = len(df) - L - H + 1
    n_tr = int(total * cfg["train_ratio"])
    n_val = int(total * cfg["val_ratio"])
    margin = L + H
    te_st = n_tr + margin + n_val + margin

    N_test = (len(df) - te_st) - L - H + 1
    if N_test <= 0:
        return None

    tod_vec = np.zeros((N_test, 2), dtype=np.float32)
    for i in range(N_test):
        base = te_st + i + L  # 예측 시작(첫 step) 시각
        tod_vec[i, :] = tod_all[base, :]  # [sin, cos]
    return tod_vec
