"""
W1 실험 특화 지표: 층별 교차 vs 최종 결합
"""

import torch
import numpy as np
from typing import Dict, Optional
from utils.metrics import _dcor_u, _pearson_corr, _spearman_corr


def _cka_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Centered Kernel Alignment (CKA) 유사도 계산
    
    Args:
        X: (N, d1) 첫 번째 표현
        Y: (N, d2) 두 번째 표현
    
    Returns:
        CKA 값 (0~1)
    """
    if X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return np.nan
    
    # Gram 행렬
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    
    K = X_centered @ X_centered.T
    L = Y_centered @ Y_centered.T
    
    # HSIC (Hilbert-Schmidt Independence Criterion)
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    K_centered = H @ K @ H
    L_centered = H @ L @ H
    
    hsic_xy = np.trace(K_centered @ L_centered) / (n - 1) ** 2
    hsic_xx = np.trace(K_centered @ K_centered) / (n - 1) ** 2
    hsic_yy = np.trace(L_centered @ L_centered) / (n - 1) ** 2
    
    if hsic_xx <= 0 or hsic_yy <= 0:
        return np.nan
    
    cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy)
    return float(np.clip(cka, 0.0, 1.0))


def _cca_similarity(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Canonical Correlation Analysis (CCA) 유사도 계산
    
    Args:
        X: (N, d1) 첫 번째 표현
        Y: (N, d2) 두 번째 표현
    
    Returns:
        최대 canonical correlation (0~1)
    """
    if X.shape[0] != Y.shape[0] or X.shape[0] < 2:
        return np.nan
    
    try:
        from sklearn.cross_decomposition import CCA
        cca = CCA(n_components=min(X.shape[1], Y.shape[1], X.shape[0] - 1))
        X_c, Y_c = cca.fit_transform(X, Y)
        
        # Canonical correlations
        corrs = []
        for i in range(X_c.shape[1]):
            corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            if np.isfinite(corr):
                corrs.append(abs(corr))
        
        return float(np.mean(corrs)) if corrs else np.nan
    except ImportError:
        # sklearn이 없으면 간단한 상관계수로 근사
        if X.shape[1] == Y.shape[1]:
            corrs = []
            for i in range(min(X.shape[1], 10)):  # 최대 10개 차원만
                corr = np.corrcoef(X[:, i], Y[:, i])[0, 1]
                if np.isfinite(corr):
                    corrs.append(abs(corr))
            return float(np.mean(corrs)) if corrs else np.nan
        return np.nan


def compute_w1_metrics(
    model,
    hooks_data: Optional[Dict] = None,
    tod_vec: Optional[np.ndarray] = None,
    **kwargs
) -> Dict:
    """
    W1 실험 특화 지표 계산
    
    Returns:
        dict with keys:
        - w1_cka_similarity_cnn_gru: CKA 유사도
        - w1_cca_similarity_cnn_gru: CCA 유사도
        - w1_layerwise_upward_improvement: 층별 상향 개선도
        - w1_inter_path_gradient_align: 경로 간 그래디언트 정렬
    """
    metrics = {}
    
    # hooks_data에서 CNN/GRU 표현 추출
    if hooks_data is None:
        hooks_data = {}
    
    # CKA/CCA 유사도 계산
    cnn_repr = hooks_data.get("cnn_representations")  # List of (N, d) arrays
    gru_repr = hooks_data.get("gru_representations")  # List of (N, d) arrays
    
    if cnn_repr is not None and gru_repr is not None and len(cnn_repr) > 0:
        # 마지막 층 표현 사용
        cnn_last = np.concatenate(cnn_repr[-1:], axis=0) if isinstance(cnn_repr[-1], list) else cnn_repr[-1]
        gru_last = np.concatenate(gru_repr[-1:], axis=0) if isinstance(gru_repr[-1], list) else gru_repr[-1]
        
        if cnn_last.shape[0] == gru_last.shape[0] and cnn_last.shape[0] > 1:
            metrics["w1_cka_similarity_cnn_gru"] = _cka_similarity(cnn_last, gru_last)
            metrics["w1_cca_similarity_cnn_gru"] = _cca_similarity(cnn_last, gru_last)
        else:
            metrics["w1_cka_similarity_cnn_gru"] = np.nan
            metrics["w1_cca_similarity_cnn_gru"] = np.nan
    else:
        # hooks_data가 없으면 NaN
        metrics["w1_cka_similarity_cnn_gru"] = np.nan
        metrics["w1_cca_similarity_cnn_gru"] = np.nan
    
    # 층별 상향 개선도 (TODO: 실제 구현 필요)
    metrics["w1_layerwise_upward_improvement"] = np.nan
    
    # 경로 간 그래디언트 정렬 (TODO: 실제 구현 필요)
    metrics["w1_inter_path_gradient_align"] = np.nan
    
    return metrics
