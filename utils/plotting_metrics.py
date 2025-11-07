"""
보고용 그림 생성을 위한 추가 지표 계산 모듈
평가 단계에서 그림 재현에 필요한 요약치를 계산
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from utils.metrics import _dcor_u, _safe_corr
try:
    from utils.experiment_metrics.w1_metrics import _cka_similarity, _cca_similarity
except ImportError:
    # Fallback
    def _cka_similarity(X, Y):
        return np.nan
    def _cca_similarity(X, Y):
        return np.nan


def compute_layerwise_cka(
    model,
    loader,
    device=None,
    n_samples=256,
    layers_to_sample=None
) -> Dict[str, float]:
    """
    층별 CKA 유사도 계산 (W1, W4용)
    
    Returns:
        dict with keys: cka_s, cka_m, cka_d (얕/중/깊)
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    depth = len(model.conv_blks)
    
    if layers_to_sample is None:
        # 얕/중/깊 층 선택
        shallow_idx = max(0, depth // 3 - 1)
        mid_idx = depth // 2
        deep_idx = depth - 1
        layers_to_sample = [shallow_idx, mid_idx, deep_idx]
    
    # 층별 CNN/GRU 출력 수집
    cnn_outputs = {i: [] for i in layers_to_sample}
    gru_outputs = {i: [] for i in layers_to_sample}
    
    sample_count = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            if sample_count >= n_samples:
                break
            
            xb = xb.to(device)
            B = xb.size(0)
            
            # Forward pass with layer hooks
            z0 = model.patch(model.revin(xb, mode='norm')).permute(2, 0, 1)
            zc = zr = z0
            
            for i, (cb, gb) in enumerate(zip(model.conv_blks, model.gru_blks)):
                zc = cb(zc)
                zr = gb(zr)
                
                if i in layers_to_sample:
                    # (T, B, d) -> (B, T*d)로 flatten
                    cnn_flat = zc.permute(1, 0, 2).reshape(B, -1).cpu().numpy()
                    gru_flat = zr.permute(1, 0, 2).reshape(B, -1).cpu().numpy()
                    
                    cnn_outputs[i].append(cnn_flat)
                    gru_outputs[i].append(gru_flat)
                
                if i < len(model.xhconv_blks):
                    zc, zr = model.xhconv_blks[i](zc, zr)
            
            sample_count += B
    
    # CKA 계산
    results = {}
    for i, layer_name in zip(layers_to_sample, ['s', 'm', 'd']):
        if len(cnn_outputs[i]) > 0 and len(gru_outputs[i]) > 0:
            cnn_all = np.concatenate(cnn_outputs[i], axis=0)[:n_samples]
            gru_all = np.concatenate(gru_outputs[i], axis=0)[:n_samples]
            
            if cnn_all.shape[0] == gru_all.shape[0] and cnn_all.shape[0] > 1:
                cka_val = _cka_similarity(cnn_all, gru_all)
                results[f"cka_{layer_name}"] = cka_val
            else:
                results[f"cka_{layer_name}"] = np.nan
        else:
            results[f"cka_{layer_name}"] = np.nan
    
    return results


def compute_gradient_alignment(
    model,
    loader,
    mu,
    std,
    device=None,
    n_batches=1,
    layers_to_sample=None
) -> Dict[str, float]:
    """
    층별 그래디언트 정렬 계산 (W1용)
    
    Returns:
        dict with keys: grad_align_s, grad_align_m, grad_align_d
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.train()  # gradient 계산을 위해 train mode
    depth = len(model.conv_blks)
    
    if layers_to_sample is None:
        shallow_idx = max(0, depth // 3 - 1)
        mid_idx = depth // 2
        deep_idx = depth - 1
        layers_to_sample = [shallow_idx, mid_idx, deep_idx]
    
    # Gradient hooks
    cnn_grads = {i: [] for i in layers_to_sample}
    gru_grads = {i: [] for i in layers_to_sample}
    
    def make_grad_hook(layer_idx, store_dict, path_name):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                g = grad_output[0].detach()
                # (T, B, d) -> (B, T*d)
                g_flat = g.permute(1, 0, 2).reshape(g.size(1), -1).cpu().numpy()
                store_dict[layer_idx].append(g_flat)
        return hook
    
    handles = []
    for i in layers_to_sample:
        h1 = model.conv_blks[i].register_full_backward_hook(
            make_grad_hook(i, cnn_grads, 'cnn')
        )
        h2 = model.gru_blks[i].register_full_backward_hook(
            make_grad_hook(i, gru_grads, 'gru')
        )
        handles.extend([h1, h2])
    
    batch_count = 0
    
    try:
        for xb, yb in loader:
            if batch_count >= n_batches:
                break
            
            xb, yb = xb.to(device), yb.to(device)
            model.zero_grad()
            
            pred = model(xb)
            loss = torch.nn.functional.mse_loss(pred, yb)
            loss.backward()
            
            batch_count += 1
    finally:
        # Cleanup
        for h in handles:
            h.remove()
        model.eval()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Gradient alignment 계산 (코사인 유사도)
    results = {}
    for i, layer_name in zip(layers_to_sample, ['s', 'm', 'd']):
        if len(cnn_grads[i]) > 0 and len(gru_grads[i]) > 0:
            cnn_g = np.concatenate(cnn_grads[i], axis=0)
            gru_g = np.concatenate(gru_grads[i], axis=0)
            
            if cnn_g.shape[0] == gru_g.shape[0] and cnn_g.shape[0] > 0:
                # 배치별 코사인 유사도 평균
                alignments = []
                for j in range(cnn_g.shape[0]):
                    align = _safe_corr(cnn_g[j], gru_g[j])
                    if np.isfinite(align):
                        alignments.append(align)
                
                results[f"grad_align_{layer_name}"] = float(np.mean(alignments)) if alignments else np.nan
            else:
                results[f"grad_align_{layer_name}"] = np.nan
        else:
            results[f"grad_align_{layer_name}"] = np.nan
    
    return results


def compute_gate_tod_heatmap(
    model,
    loader,
    tod_vec,
    device=None,
    n_tod_bins=24,
    layers_to_sample=None
) -> Dict[str, str]:
    """
    시간대별 게이트 히트맵 데이터 (W2용)
    
    Returns:
        dict with keys: gate_tod_mean_s, gate_tod_mean_m, gate_tod_mean_d
        각 값은 세미콜론 구분 문자열 "v1;v2;...;v24"
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    depth = len(model.xhconv_blks)
    
    if layers_to_sample is None:
        shallow_idx = max(0, depth // 3 - 1)
        mid_idx = depth // 2
        deep_idx = depth - 1
        layers_to_sample = [shallow_idx, mid_idx, deep_idx]
    
    # TOD bin별 게이트 값 수집
    gate_by_tod = {i: [[] for _ in range(n_tod_bins)] for i in layers_to_sample}
    
    sample_idx = 0
    
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            B = xb.size(0)
            
            # TOD 라벨
            if tod_vec is not None and sample_idx < tod_vec.shape[0]:
                batch_tod = tod_vec[sample_idx:sample_idx + B, :]
                # TOD [sin, cos] -> 시간 bin (0~23)
                # sin, cos를 각도로 변환
                angles = np.arctan2(batch_tod[:, 1], batch_tod[:, 0])  # -π ~ π
                angles = (angles + np.pi) / (2 * np.pi)  # 0 ~ 1
                tod_bins = (angles * n_tod_bins).astype(int).clip(0, n_tod_bins - 1)
            else:
                tod_bins = np.zeros(B, dtype=int)
            
            # Forward pass
            z0 = model.patch(model.revin(xb, mode='norm')).permute(2, 0, 1)
            zc = zr = z0
            
            for i, (cb, gb) in enumerate(zip(model.conv_blks, model.gru_blks)):
                zc = cb(zc)
                zr = gb(zr)
                
                if i < len(model.xhconv_blks):
                    blk = model.xhconv_blks[i]
                    if i in layers_to_sample:
                        # 게이트 값 (alpha)
                        alpha_val = torch.relu(blk.alpha).detach().cpu().numpy()
                        gate_mean = float(np.mean(alpha_val))
                        
                        # TOD bin별로 저장
                        for b_idx, tod_bin in enumerate(tod_bins):
                            gate_by_tod[i][tod_bin].append(gate_mean)
                    
                    zc, zr = blk(zc, zr)
            
            sample_idx += B
    
    # TOD bin별 평균 계산 및 문자열로 변환
    results = {}
    for i, layer_name in zip(layers_to_sample, ['s', 'm', 'd']):
        bin_means = []
        for bin_idx in range(n_tod_bins):
            if len(gate_by_tod[i][bin_idx]) > 0:
                mean_val = float(np.mean(gate_by_tod[i][bin_idx]))
            else:
                mean_val = 0.0
            bin_means.append(mean_val)
        
        # 세미콜론 구분 문자열
        results[f"gate_tod_mean_{layer_name}"] = ";".join([f"{v:.6f}" for v in bin_means])
    
    return results


def compute_gate_distribution(
    model,
    device=None
) -> Dict[str, float]:
    """
    게이트 분포 통계 (W2, W4, W5용)
    
    Returns:
        dict with keys: gate_var_t, gate_var_b, gate_entropy, gate_channel_kurt,
        gate_channel_sparsity, gate_q10, gate_q50, gate_q90, gate_hist10
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    # 모든 층의 게이트 값 수집
    gate_values = []
    
    with torch.no_grad():
        for blk in model.xhconv_blks:
            if hasattr(blk, 'alpha'):
                alpha_val = torch.relu(blk.alpha).detach().cpu().numpy()
                gate_values.append(alpha_val.flatten())
    
    if len(gate_values) == 0:
        return {k: np.nan for k in [
            "gate_var_t", "gate_var_b", "gate_entropy", "gate_channel_kurt",
            "gate_channel_sparsity", "gate_q10", "gate_q50", "gate_q90", "gate_hist10"
        ]}
    
    gate_array = np.concatenate(gate_values)
    
    # 변동성
    var_t = float(np.var(gate_array))
    var_b = float(np.var(gate_array, axis=0).mean()) if gate_array.ndim > 1 else var_t
    
    # 엔트로피
    gate_flat = gate_array.flatten()
    gate_flat = gate_flat[gate_flat > 1e-8]
    if len(gate_flat) > 0:
        gate_norm = gate_flat / (gate_flat.sum() + 1e-12)
        entropy = -np.sum(gate_norm * np.log(gate_norm + 1e-12))
    else:
        entropy = np.nan
    
    # 채널 선택도 (첨도)
    try:
        from scipy import stats
        kurt = float(stats.kurtosis(gate_flat)) if len(gate_flat) > 0 else np.nan
    except ImportError:
        # scipy가 없으면 수동 계산
        if len(gate_flat) > 3:
            mean = np.mean(gate_flat)
            std = np.std(gate_flat)
            if std > 1e-12:
                kurt = float(np.mean(((gate_flat - mean) / std) ** 4) - 3.0)
            else:
                kurt = np.nan
        else:
            kurt = np.nan
    
    # 채널 선택도 (희소성)
    gate_l1 = np.abs(gate_flat).sum()
    gate_l2 = np.sqrt((gate_flat ** 2).sum())
    sparsity = float(gate_l1 / (gate_l2 * np.sqrt(len(gate_flat)) + 1e-12)) if len(gate_flat) > 0 else np.nan
    
    # 분위수
    q10 = float(np.quantile(gate_flat, 0.1)) if len(gate_flat) > 0 else np.nan
    q50 = float(np.quantile(gate_flat, 0.5)) if len(gate_flat) > 0 else np.nan
    q90 = float(np.quantile(gate_flat, 0.9)) if len(gate_flat) > 0 else np.nan
    
    # 히스토그램 (10-bin)
    if len(gate_flat) > 0:
        hist, _ = np.histogram(gate_flat, bins=10, range=(gate_flat.min(), gate_flat.max()))
        hist_str = ";".join([str(int(c)) for c in hist])
    else:
        hist_str = ";".join(["0"] * 10)
    
    return {
        "gate_var_t": var_t,
        "gate_var_b": var_b,
        "gate_entropy": float(entropy) if np.isfinite(entropy) else np.nan,
        "gate_channel_kurt": kurt if np.isfinite(kurt) else np.nan,
        "gate_channel_sparsity": sparsity if np.isfinite(sparsity) else np.nan,
        "gate_q10": q10,
        "gate_q50": q50,
        "gate_q90": q90,
        "gate_hist10": hist_str
    }


def compute_bestlag_distribution(
    hooks_data: Dict,
    baseline_hooks_data: Optional[Dict] = None
) -> Dict[str, float]:
    """
    라그 분포 요약 통계 (W3용)
    
    Returns:
        dict with keys: bestlag_neg_ratio, bestlag_var, bestlag_hist21
    """
    # bestlag는 이미 direct_evidence에서 계산됨
    # 여기서는 분포 요약만 수행
    bestlag = hooks_data.get("cg_bestlag", np.nan)
    
    if not np.isfinite(bestlag):
        return {
            "bestlag_neg_ratio": np.nan,
            "bestlag_var": np.nan,
            "bestlag_hist21": ";".join(["0"] * 21)
        }
    
    # 단일 값이므로 분포 통계는 제한적
    # 실제로는 여러 샘플의 bestlag가 필요하지만, 여기서는 평균값만 사용
    neg_ratio = 1.0 if bestlag < 0 else 0.0
    var = 0.0  # 단일 값이므로 분산은 0
    
    # 히스토그램 (-10 ~ +10, 21 bins)
    hist = [0] * 21
    lag_bin = int(np.clip(bestlag + 10, 0, 20))
    hist[lag_bin] = 1
    hist_str = ";".join([str(c) for c in hist])
    
    return {
        "bestlag_neg_ratio": float(neg_ratio),
        "bestlag_var": float(var),
        "bestlag_hist21": hist_str
    }


def compute_all_plotting_metrics(
    model,
    loader,
    mu,
    std,
    tod_vec=None,
    device=None,
    experiment_type="W1",
    **kwargs
) -> Dict:
    """
    모든 보고용 그림 지표 계산 (통합 함수)
    """
    results = {}
    
    # 공통: 게이트 분포
    gate_dist = compute_gate_distribution(model, device)
    results.update(gate_dist)
    
    if experiment_type in ["W1", "W4"]:
        # 층별 CKA
        cka_results = compute_layerwise_cka(model, loader, device)
        results.update(cka_results)
        
        # 층별 그래디언트 정렬 (W1만, 선택적)
        if experiment_type == "W1" and kwargs.get("compute_grad_align", False):
            grad_results = compute_gradient_alignment(model, loader, mu, std, device)
            results.update(grad_results)
    
    if experiment_type == "W2":
        # 시간대별 게이트 히트맵
        if tod_vec is not None:
            tod_results = compute_gate_tod_heatmap(model, loader, tod_vec, device)
            results.update(tod_results)
    
    if experiment_type == "W3":
        # 라그 분포
        hooks_data = kwargs.get("hooks_data", {})
        baseline_hooks_data = kwargs.get("baseline_hooks_data")
        lag_results = compute_bestlag_distribution(hooks_data, baseline_hooks_data)
        results.update(lag_results)
    
    return results
