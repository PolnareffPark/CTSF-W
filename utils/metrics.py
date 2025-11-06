"""
평가 지표 계산 모듈
CTSF-V1.py의 regression_metrics 및 직접 근거 지표들을 모듈화
"""

import numpy as np
import torch
import torch.nn.functional as F


def regression_metrics(pred, target):
    """
    기본 회귀 지표 계산
    
    Args:
        pred, target: (N, H) 또는 (B, C, H) torch tensors, real-scale
    
    Returns:
        dict with keys: rmse, mae, rrmse, r2, cc, nse, pbias
    """
    p = pred.flatten()
    t = target.flatten()
    n = t.numel()

    mse = torch.mean((p - t) ** 2)
    rmse = torch.sqrt(mse)
    mae = torch.mean(torch.abs(p - t))
    rrmse = rmse / (t.mean().clamp(min=1e-6)).abs()

    # R² (coefficient of determination)
    ss_tot = torch.sum((t - t.mean()) ** 2)
    r2 = 1 - mse * n / ss_tot

    # Pearson CC
    cov = torch.mean((p - p.mean()) * (t - t.mean()))
    cc = cov / (p.std() * t.std() + 1e-6)

    # NSE (Nash–Sutcliffe efficiency)
    nse = 1 - torch.sum((p - t) ** 2) / (ss_tot + 1e-6)

    # PBIAS (%)
    pbias = 100 * torch.sum(t - p) / (torch.sum(t) + 1e-6)

    return dict(
        rmse=rmse.item(), mae=mae.item(), rrmse=rrmse.item(),
        r2=r2.item(), cc=cc.item(), nse=nse.item(),
        pbias=pbias.item()
    )


# ==================== 안전한 상관 계산 ====================
def _safe_corr(aa, bb):
    """안전한 상관계수 계산 (0으로 나누기 방지)"""
    aa = np.asarray(aa, dtype=np.float64)
    bb = np.asarray(bb, dtype=np.float64)
    if aa.size != bb.size or aa.size < 2:
        return np.nan
    aa = aa - aa.mean()
    bb = bb - bb.mean()
    va = (aa ** 2).sum()
    vb = (bb ** 2).sum()
    if va <= 1e-12 or vb <= 1e-12:
        return np.nan
    r = float(np.dot(aa, bb) / np.sqrt(va * vb))
    if not np.isfinite(r):
        return np.nan
    # 수치 안정: 경계값 딱 1/−1로 포화되는 것을 미세하게 완화
    r = max(min(r, 1.0 - 1e-9), -1.0 + 1e-9)
    return r


def _zscore_t(a, dim=1, eps=1e-8):
    """Z-score 정규화"""
    m = a.mean(dim=dim, keepdim=True)
    s = a.std(dim=dim, keepdim=True) + eps
    return (a - m) / s


def _pearson_corr(a, b):
    """Pearson 상관계수 (B,T) -> (B,)"""
    a = _zscore_t(a, dim=1)
    b = _zscore_t(b, dim=1)
    return (a * b).mean(dim=1).clamp(-1, 1)


def _rank_2d(x_np):
    """2D 배열의 순위 계산"""
    return np.argsort(np.argsort(x_np, axis=1), axis=1).astype(np.float32)


def _spearman_corr(a, b):
    """Spearman 순위상관"""
    a_np, b_np = a.detach().cpu().numpy(), b.detach().cpu().numpy()
    ra = torch.tensor(_rank_2d(a_np), device=a.device)
    rb = torch.tensor(_rank_2d(b_np), device=b.device)
    return _pearson_corr(ra, rb)


# ==================== 시간 도함수 및 변화량 ====================
def _time_deriv_norm(x):
    """시간 도함수 노름 (T,B,d) -> (B,T-1)"""
    dx = x[1:] - x[:-1]
    n = torch.linalg.vector_norm(dx, dim=-1)
    return n.transpose(0, 1).contiguous()


# ==================== 이벤트 검출 지표 ====================
def _event_gain_and_hit(conv_d_b, gru_d_b, q_gain=0.80, q_hit=0.90):
    """
    Conv→GRU 경로의 이벤트 게인 및 적중률
    
    Args:
        conv_d_b, gru_d_b: (B,T) 변화량
        q_gain: 게인 계산용 분위수
        q_hit: 적중률 계산용 분위수
    
    Returns:
        (gain_mean, hit_mean)
    """
    gains, hits = [], []
    a_np = conv_d_b.detach().cpu().numpy()
    b_np = gru_d_b.detach().cpu().numpy()
    B, T = a_np.shape
    
    for i in range(B):
        a = a_np[i]
        b = b_np[i]
        
        # gain: Conv 상위 구간에서의 GRU 업데이트 평균 / 그 외 평균
        thr_g = np.quantile(a, q_gain)
        hi = a >= thr_g
        lo = ~hi
        if hi.any() and lo.any():
            gain = float(b[hi].mean() / (b[lo].mean() + 1e-12))
            gains.append(gain)
        
        # hit: Conv 상위 이벤트가 있을 때 ±1 step 이내에 GRU 상위 이벤트가 있었던 비율
        thr_h_a = np.quantile(a, q_hit)
        thr_h_b = np.quantile(b, q_hit)
        a_top = a >= thr_h_a
        b_top = b >= thr_h_b
        
        # GRU 톱 이벤트를 ±1 확장
        b_dil = b_top.copy()
        if T > 1:
            b_dil[:-1] |= b_top[1:]
            b_dil[1:] |= b_top[:-1]
        
        if a_top.sum() > 0:
            hit = float((a_top & b_dil).sum() / a_top.sum())
            hits.append(hit)
    
    gain_mean = float(np.mean(gains)) if gains else np.nan
    hit_mean = float(np.mean(hits)) if hits else np.nan
    return gain_mean, hit_mean


# ==================== 최대 상관 및 시차 ====================
def _maxcorr_and_lag_mean(conv_d_b, gru_d_b):
    """
    라그별 최대 상관 및 최적 시차
    
    Args:
        conv_d_b, gru_d_b: (B, N) 변화량
    
    Returns:
        (배치평균 최대상관 |r|, 배치평균 최적 라그)
    """
    a = conv_d_b.detach().cpu().numpy().astype(np.float64)
    b = gru_d_b.detach().cpu().numpy().astype(np.float64)
    Na, Nb = a.shape[1], b.shape[1]
    N = min(Na, Nb)
    if N < 5:
        return float("nan"), float("nan")

    min_overlap = max(3, int(np.sqrt(N)))  # 데이터 길이에 따른 자동 기준
    bests, lags = [], []
    
    for i in range(a.shape[0]):
        ai = a[i, :N]
        bi = b[i, :N]
        # z-score
        ai = (ai - ai.mean()) / (ai.std() + 1e-8)
        bi = (bi - bi.mean()) / (bi.std() + 1e-8)

        best_eff, best_raw, bestL = -1.0, 0.0, 0
        # |L| ≤ N - min_overlap  ⇒ 겹침 M=N-|L| ≥ min_overlap 보장
        for L in range(-(N - min_overlap), (N - min_overlap) + 1):
            if L >= 0:
                aa, bb = ai[:N - L], bi[L:]
            else:
                aa, bb = ai[-L:], bi[:N + L]
            M = aa.shape[0]
            if M < min_overlap:
                continue
            r = _safe_corr(aa, bb)
            if not np.isfinite(r):
                continue
            w = M / N
            eff = abs(r) * w
            if eff > best_eff:
                best_eff, best_raw, bestL = eff, abs(r), L
        bests.append(best_raw)
        lags.append(bestL)

    return float(np.mean(bests)), float(np.mean(lags))


# ==================== 거리상관 (dCor) ====================
def _pairwise_dmat(X: np.ndarray) -> np.ndarray:
    """쌍대 거리 행렬"""
    X = np.asarray(X, dtype=np.float64)
    XX = (X ** 2).sum(axis=1, keepdims=True)
    D2 = XX + XX.T - 2.0 * X.dot(X.T)
    np.maximum(D2, 0.0, out=D2)
    return np.sqrt(D2, dtype=np.float64)


def _dcor_u(X: np.ndarray, Y: np.ndarray) -> float:
    """U-centering 기반 거리상관"""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    n = X.shape[0]
    if n < 4 or Y.shape[0] != n:
        return np.nan
    
    A = _pairwise_dmat(X)
    B = _pairwise_dmat(Y)
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(B, 0.0)
    rowA = A.sum(axis=1) / (n - 1)
    rowB = B.sum(axis=1) / (n - 1)
    gA = A.sum() / (n * (n - 1))
    gB = B.sum() / (n * (n - 1))
    A = A - rowA[:, None] - rowA[None, :] + gA
    B = B - rowB[:, None] - rowB[None, :] + gB
    np.fill_diagonal(A, 0.0)
    np.fill_diagonal(B, 0.0)
    denom = n * (n - 3)
    if denom <= 0:
        return np.nan
    dcov2 = (A * B).sum() / denom
    dvarx = (A * A).sum() / denom
    dvary = (B * B).sum() / denom
    dcov2 = max(0.0, dcov2)
    dvarx = max(dvarx, 1e-16)
    dvary = max(dvary, 1e-16)
    val = np.sqrt(dcov2) / np.sqrt(dvarx * dvary)
    if not np.isfinite(val):
        return np.nan
    return float(min(max(val, 0.0), 1.0 - 1e-9))


def _dcor_bt_u(a_bt: torch.Tensor, b_bt: torch.Tensor) -> np.ndarray:
    """배치별 거리상관"""
    a_np = a_bt.detach().cpu().numpy().astype(np.float64)
    b_np = b_bt.detach().cpu().numpy().astype(np.float64)
    out = []
    for i in range(a_np.shape[0]):
        out.append(_dcor_u(a_np[i][:, None], b_np[i][:, None]))
    return np.array(out, dtype=np.float64)


# ==================== 다중출력 R² ====================
def _multioutput_r2(Y: np.ndarray, X: np.ndarray) -> float:
    """
    다중출력 R² (E ~ TOD[sin,cos])
    
    Args:
        Y: (N,d) 출력
        X: (N,2) 입력 (TOD [sin,cos])
    
    Returns:
        평균 R²
    """
    N = Y.shape[0]
    if N < 4 or X.shape[0] != N:
        return np.nan
    X1 = np.concatenate([X, np.ones((N, 1))], axis=1)  # bias
    # 최소제곱 해(다중출력)
    beta, *_ = np.linalg.lstsq(X1, Y, rcond=None)
    Yhat = X1 @ beta
    ss_res = ((Y - Yhat) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - (ss_res / ss_tot)
    r2 = np.clip(r2, 0.0, 1.0 - 1e-9)
    return float(np.nanmean(r2))
