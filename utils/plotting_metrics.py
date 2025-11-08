"""
보고용 그림 생성을 위한 추가 지표 계산 모듈
평가 단계에서 그림 재현에 필요한 요약치를 계산
"""

import torch
import numpy as np
from typing import Dict, Optional, List
from utils.metrics import _dcor_u, _safe_corr
try:
    from utils.experiment_metrics.w1_metrics import _cka_similarity
except ImportError:
    def _cka_similarity(X, Y):
        return np.nan


def _split_bins_auto(n_layers: int) -> List[str]:
    if n_layers <= 0:
        return []
    first_cut = int(np.floor(n_layers / 3))
    second_cut = int(np.floor(2 * n_layers / 3))
    bins = []
    for idx in range(n_layers):
        if idx < first_cut:
            bins.append("S")
        elif idx < second_cut:
            bins.append("M")
        else:
            bins.append("D")
    return bins

def compute_layerwise_cka(
    model,
    loader,
    device=None,
    n_samples: int = 256,
    layers_to_sample=None
) -> Dict[str, float]:
    """
    층별 CKA 유사도 계산. S/M/D 요약뿐 아니라 전체 CNN×GRU CKA 행렬을 반환한다.
    """
    if device is None:
        device = next(model.parameters()).device

    original_mode = model.training
    model.eval()

    depth = len(model.conv_blks)
    if depth == 0:
        return {}

    if layers_to_sample is None:
        layers_to_sample = list(range(depth))

    cnn_outputs = {i: [] for i in layers_to_sample}
    gru_outputs = {i: [] for i in layers_to_sample}

    collected = 0

    with torch.no_grad():
        for xb, yb in loader:
            if collected >= n_samples:
                break

            xb = xb.to(device)
            B = xb.size(0)

            z0 = model.patch(model.revin(xb, mode='norm')).permute(2, 0, 1)
            zc = zr = z0

            for idx, (cb, gb) in enumerate(zip(model.conv_blks, model.gru_blks)):
                zc = cb(zc)
                zr = gb(zr)

                if idx in layers_to_sample:
                    cnn_flat = zc.permute(1, 0, 2).reshape(B, -1).cpu().numpy()
                    gru_flat = zr.permute(1, 0, 2).reshape(B, -1).cpu().numpy()
                    cnn_outputs[idx].append(cnn_flat)
                    gru_outputs[idx].append(gru_flat)

                if idx < len(model.xhconv_blks):
                    zc, zr = model.xhconv_blks[idx](zc, zr)

            collected += B

    if not original_mode:
        model.eval()
    else:
        model.train()

    # 변환 및 최소 샘플 확보
    cnn_arrays = []
    gru_arrays = []
    for idx in range(depth):
        if idx in cnn_outputs and cnn_outputs[idx]:
            arr = np.concatenate(cnn_outputs[idx], axis=0)
            cnn_arrays.append(arr)
        else:
            cnn_arrays.append(None)

        if idx in gru_outputs and gru_outputs[idx]:
            arr = np.concatenate(gru_outputs[idx], axis=0)
            gru_arrays.append(arr)
        else:
            gru_arrays.append(None)

    valid_counts = [arr.shape[0] for arr in cnn_arrays if arr is not None]
    valid_counts += [arr.shape[0] for arr in gru_arrays if arr is not None]
    min_samples = min(valid_counts) if valid_counts else 0
    if min_samples <= 1:
        return {
            "cka_matrix": np.full((depth, depth), np.nan),
            "cnn_depth_bins": _split_bins_auto(depth),
            "gru_depth_bins": _split_bins_auto(depth),
            "cka_s": np.nan,
            "cka_m": np.nan,
            "cka_d": np.nan,
            "cka_diag_mean": np.nan,
            "cka_full_mean": np.nan,
        }

    min_samples = min(min_samples, n_samples)

    cka_matrix = np.full((depth, depth), np.nan)
    for i in range(depth):
        if cnn_arrays[i] is None:
            continue
        cnn_data = cnn_arrays[i][:min_samples]
        if cnn_data.shape[0] <= 1:
            continue
        for j in range(depth):
            if gru_arrays[j] is None:
                continue
            gru_data = gru_arrays[j][:min_samples]
            if gru_data.shape[0] <= 1:
                continue
            cka_matrix[i, j] = _cka_similarity(cnn_data, gru_data)

    cnn_bins = _split_bins_auto(depth)
    gru_bins = _split_bins_auto(depth)

    def _band_mean(band: str) -> float:
        idx_c = [i for i, b in enumerate(cnn_bins) if b == band]
        idx_g = [j for j, b in enumerate(gru_bins) if b == band]
        if not idx_c or not idx_g:
            return np.nan
        sub = cka_matrix[np.ix_(idx_c, idx_g)]
        return float(np.nanmean(sub)) if np.isfinite(sub).any() else np.nan

    diag_mean = float(np.nanmean(np.diag(cka_matrix))) if np.isfinite(np.diag(cka_matrix)).any() else np.nan
    full_mean = float(np.nanmean(cka_matrix)) if np.isfinite(cka_matrix).any() else np.nan

    return {
        "cka_matrix": cka_matrix,
        "cnn_depth_bins": cnn_bins,
        "gru_depth_bins": gru_bins,
        "cka_s": _band_mean("S"),
        "cka_m": _band_mean("M"),
        "cka_d": _band_mean("D"),
        "cka_diag_mean": diag_mean,
        "cka_full_mean": full_mean,
    }


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

    original_mode = model.training
    model.train()  # gradient 계산을 위해 train mode
    depth = len(model.conv_blks)

    if depth == 0:
        return {}

    if layers_to_sample is None:
        layers_to_sample = list(range(depth))

    depth_bins = _split_bins_auto(depth)

    cnn_grads = {i: [] for i in layers_to_sample}
    gru_grads = {i: [] for i in layers_to_sample}
    alignment_logs = {i: [] for i in layers_to_sample}
    
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
        h1 = model.conv_blks[i].register_full_backward_hook(make_grad_hook(i, cnn_grads, 'cnn'))
        h2 = model.gru_blks[i].register_full_backward_hook(make_grad_hook(i, gru_grads, 'gru'))
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
        for h in handles:
            h.remove()
        if original_mode:
            model.train()
        else:
            model.eval()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Gradient alignment 계산 (코사인 유사도)
    grad_align_by_layer = []
    s_vals, m_vals, d_vals = [], [], []

    for i in layers_to_sample:
        if len(cnn_grads[i]) == 0 or len(gru_grads[i]) == 0:
            grad_align_by_layer.append({
                "layer_idx": i,
                "depth_bin": depth_bins[i] if i < len(depth_bins) else "",
                "grad_align_value": np.nan,
            })
            continue

        cnn_g = np.concatenate(cnn_grads[i], axis=0)
        gru_g = np.concatenate(gru_grads[i], axis=0)

        if cnn_g.shape[0] != gru_g.shape[0] or cnn_g.shape[0] == 0:
            grad_align_by_layer.append({
                "layer_idx": i,
                "depth_bin": depth_bins[i] if i < len(depth_bins) else "",
                "grad_align_value": np.nan,
            })
            continue

        alignments = []
        for j in range(cnn_g.shape[0]):
            align = _safe_corr(cnn_g[j], gru_g[j])
            if np.isfinite(align):
                alignments.append(align)

        mean_align = float(np.mean(alignments)) if alignments else np.nan
        grad_align_by_layer.append({
            "layer_idx": i,
            "depth_bin": depth_bins[i] if i < len(depth_bins) else "",
            "grad_align_value": mean_align,
        })

        bin_label = depth_bins[i] if i < len(depth_bins) else ""
        if bin_label == "S":
            s_vals.append(mean_align)
        elif bin_label == "M":
            m_vals.append(mean_align)
        elif bin_label == "D":
            d_vals.append(mean_align)

    def _mean_or_nan(values):
        vals = [v for v in values if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    results = {
        "grad_align_by_layer": grad_align_by_layer,
        "grad_align_s": _mean_or_nan(s_vals),
        "grad_align_m": _mean_or_nan(m_vals),
        "grad_align_d": _mean_or_nan(d_vals),
    }
    all_vals = [entry["grad_align_value"] for entry in grad_align_by_layer if np.isfinite(entry["grad_align_value"])]
    results["grad_align_mean"] = float(np.mean(all_vals)) if all_vals else np.nan

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
    
    if experiment_type == "W1":
        # W1은 전용 모듈을 통해 그림 데이터를 처리하므로 여기서는 반환하지 않음
        return results

    gate_dist = compute_gate_distribution(model, device)
    results.update(gate_dist)

    if experiment_type == "W4":
        cka_results = compute_layerwise_cka(model, loader, device)
        results.update({k: v for k, v in cka_results.items() if k in ["cka_s", "cka_m", "cka_d", "cka_diag_mean", "cka_full_mean"]})
    
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
