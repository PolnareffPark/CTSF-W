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
    
    Args:
        model: 평가할 모델 (사용되지 않지만 인터페이스 일관성 유지)
        hooks_data: 후크로 수집된 데이터 딕셔너리
            - cnn_representations: CNN 경로의 층별 표현 리스트
            - gru_representations: GRU 경로의 층별 표현 리스트
            - layerwise_losses: 각 층에서의 손실값 리스트
            - cnn_gradients: CNN 경로의 그래디언트
            - gru_gradients: GRU 경로의 그래디언트
        tod_vec: Time-of-Day 벡터 (사용되지 않음)
        **kwargs: 추가 인자
    
    Returns:
        dict with keys:
        - w1_cka_similarity_cnn_gru: CKA 유사도 (0~1, 높을수록 표현이 유사)
        - w1_cca_similarity_cnn_gru: CCA 유사도 (0~1, 높을수록 표현이 유사)
        - w1_layerwise_upward_improvement: 층별 상향 개선도 (양수일수록 개선)
        - w1_inter_path_gradient_align: 경로 간 그래디언트 정렬 (-1~1, 높을수록 동일 방향)
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
    
    # 층별 상향 개선도
    layerwise_losses = hooks_data.get("layerwise_losses")  # List of losses per layer
    if layerwise_losses is not None and len(layerwise_losses) > 1:
        # 층별 손실 감소량 계산 (이전 층 대비 개선도)
        improvements = []
        for i in range(1, len(layerwise_losses)):
            if layerwise_losses[i-1] > 0:
                improvement = (layerwise_losses[i-1] - layerwise_losses[i]) / layerwise_losses[i-1]
                improvements.append(improvement)
        
        # 평균 개선도 (양수일수록 좋음)
        metrics["w1_layerwise_upward_improvement"] = float(np.mean(improvements)) if improvements else np.nan
    else:
        metrics["w1_layerwise_upward_improvement"] = np.nan
    
    # 경로 간 그래디언트 정렬
    cnn_grads = hooks_data.get("cnn_gradients")  # CNN 경로의 gradient (N, d)
    gru_grads = hooks_data.get("gru_gradients")  # GRU 경로의 gradient (N, d)
    
    if cnn_grads is not None and gru_grads is not None:
        # 두 gradient를 평탄화
        if isinstance(cnn_grads, torch.Tensor):
            cnn_grads = cnn_grads.detach().cpu().numpy()
        if isinstance(gru_grads, torch.Tensor):
            gru_grads = gru_grads.detach().cpu().numpy()
        
        # 다차원 배열인 경우 평탄화
        cnn_flat = cnn_grads.flatten()
        gru_flat = gru_grads.flatten()
        
        # 코사인 유사도 계산
        if len(cnn_flat) > 0 and len(gru_flat) > 0 and len(cnn_flat) == len(gru_flat):
            norm_cnn = np.linalg.norm(cnn_flat)
            norm_gru = np.linalg.norm(gru_flat)
            
            if norm_cnn > 1e-8 and norm_gru > 1e-8:
                cosine_sim = np.dot(cnn_flat, gru_flat) / (norm_cnn * norm_gru)
                metrics["w1_inter_path_gradient_align"] = float(np.clip(cosine_sim, -1.0, 1.0))
            else:
                metrics["w1_inter_path_gradient_align"] = np.nan
        else:
            metrics["w1_inter_path_gradient_align"] = np.nan
    else:
        metrics["w1_inter_path_gradient_align"] = np.nan
    
    return metrics
