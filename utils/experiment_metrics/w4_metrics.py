"""
W4 실험 특화 지표: 층별 기여도 분석
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from utils.metrics import _dcor_u


def compute_w4_metrics(
    model,
    hooks_data: Optional[Dict] = None,
    active_layers: List[int] = None,
    **kwargs
) -> Dict:
    """
    W4 실험 특화 지표 계산
    
    Args:
        active_layers: 활성화된 층 인덱스 리스트
    
    Returns:
        dict with keys:
        - w4_layer_contribution_score: 층별 기여 점수 (NCS)
        - w4_layerwise_gate_usage: 층별 게이트 사용률
        - w4_layerwise_representation_similarity: 층별 표현 유사도 (CNN↔GRU CKA)
    """
    metrics = {}
    
    if active_layers is None:
        active_layers = []
    
    # 층별 게이트 사용률
    gate_usage_list = []
    for i, blk in enumerate(model.xhconv_blks):
        if i in active_layers:
            if hasattr(blk, 'alpha'):
                alpha_val = torch.relu(blk.alpha).detach().cpu().numpy()
                usage = float(np.mean(alpha_val))
                gate_usage_list.append(usage)
    
    if len(gate_usage_list) > 0:
        metrics["w4_layerwise_gate_usage"] = float(np.mean(gate_usage_list))
    else:
        metrics["w4_layerwise_gate_usage"] = np.nan
    
    # 층별 기여 점수 (Normalized Contribution Score)
    # 활성화된 층의 게이트 사용률을 기반으로 계산
    if len(gate_usage_list) > 0:
        # 활성 층의 평균 사용률을 전체 층 대비로 정규화
        total_layers = len(model.xhconv_blks)
        active_ratio = len(active_layers) / total_layers if total_layers > 0 else 0
        avg_usage = np.mean(gate_usage_list)
        ncs = avg_usage * active_ratio  # 정규화된 기여 점수
        metrics["w4_layer_contribution_score"] = float(ncs)
    else:
        metrics["w4_layer_contribution_score"] = np.nan
    
    # 층별 표현 유사도 (CNN↔GRU)
    if hooks_data is not None:
        cnn_repr = hooks_data.get("cnn_representations")
        gru_repr = hooks_data.get("gru_representations")
        
        if cnn_repr is not None and gru_repr is not None:
            # 활성 층의 표현만 사용
            similarities = []
            for i in active_layers:
                if i < len(cnn_repr) and i < len(gru_repr):
                    cnn_i = cnn_repr[i] if isinstance(cnn_repr[i], np.ndarray) else np.array(cnn_repr[i])
                    gru_i = gru_repr[i] if isinstance(gru_repr[i], np.ndarray) else np.array(gru_repr[i])
                    
                    if cnn_i.shape[0] == gru_i.shape[0] and cnn_i.shape[0] > 1:
                        sim = _dcor_u(cnn_i, gru_i)
                        if np.isfinite(sim):
                            similarities.append(sim)
            
            if len(similarities) > 0:
                metrics["w4_layerwise_representation_similarity"] = float(np.mean(similarities))
            else:
                metrics["w4_layerwise_representation_similarity"] = np.nan
        else:
            metrics["w4_layerwise_representation_similarity"] = np.nan
    else:
        metrics["w4_layerwise_representation_similarity"] = np.nan
    
    return metrics
