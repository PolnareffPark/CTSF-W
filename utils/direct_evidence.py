"""
직접 근거 지표 평가 모듈
TOD 민감도, 피크 반응 등 직접 근거 지표 계산
"""

import torch
import numpy as np
from utils.hooks import LastBlockHooks
from utils.metrics import (
    _time_deriv_norm, _pearson_corr, _spearman_corr, _dcor_bt_u,
    _event_gain_and_hit, _maxcorr_and_lag_mean, _dcor_u, _multioutput_r2
)


@torch.no_grad()
def evaluate_with_direct_evidence(model, loader, mu, std, tod_vec=None, device=None):
    """
    직접 근거 지표 포함 평가
    
    Args:
        model: CTSF 모델
        loader: 테스트 데이터로더
        mu, std: 표준화 통계
        tod_vec: (N_test, 2) TOD [sin,cos] 벡터 또는 None
        device: 디바이스
    
    Returns:
        지표 딕셔너리
    """
    model.eval()
    if device is None:
        device = next(model.parameters()).device

    # 경로 on/off 확인
    # use_gc: GRU→Conv 경로, use_cg: Conv→GRU 경로
    # both 환경에서는 모든 블록의 use_gc=True, use_cg=True이므로 둘 다 True가 됨
    # 따라서 both 환경에서도 정상 작동함
    gc_on = any(getattr(blk, "use_gc", True) for blk in model.xhconv_blks)
    cg_on = any(getattr(blk, "use_cg", True) for blk in model.xhconv_blks)

    # 누적 버퍼
    Pz_all, Tz_all, P_all, T_all = [], [], [], []

    # Conv→GRU
    cg_p_list, cg_s_list, cg_dcor_list = [], [], []
    gain_list, hit_list, maxc_list, blag_list = [], [], [], []

    # GRU→Conv
    W_list, E_list, Y_list = [], [], []  # 생성커널, Conv변화, TOD[sin,cos]

    hooks = LastBlockHooks(model)
    hooks.attach()
    sample_idx_base = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        # TOD 라벨(예측 시작 시점)
        y_tod = None
        if tod_vec is not None:
            B = xb.size(0)
            y_tod = tod_vec[sample_idx_base: sample_idx_base + B, :]
            sample_idx_base += B

        # 예측
        pred = model(xb)
        # 표준화/실측 스케일 동시 누적
        Pz_all.append(pred.detach().cpu().numpy())
        Tz_all.append(yb.detach().cpu().numpy())
        P_all.append((pred * std + mu).detach().cpu().numpy())
        T_all.append((yb * std + mu).detach().cpu().numpy())

        # 내부 신호 (마지막 블록)
        C_pre = hooks.C_last_pre
        C_post = hooks.C_last_post
        R_post = hooks.R_last_post

        # ---- Conv→GRU: 변화량 의존 지표들 (cg_on일 때만 계산) ----
        if cg_on and (C_pre is not None) and (R_post is not None):
            conv_d = _time_deriv_norm(C_pre)  # (B,T-1)
            gru_d = _time_deriv_norm(R_post)  # (B,T-1)

            # (1) 0-lag Pearson
            cg_p_list.append(_pearson_corr(conv_d, gru_d).cpu().numpy())

            # (2) Spearman
            cg_s_list.append(_spearman_corr(conv_d, gru_d).cpu().numpy())

            # (3) 거리상관(U-centering)
            cg_dcor_list.append(_dcor_bt_u(conv_d, gru_d))

            # (4) 이벤트 지표
            g_gain, g_hit = _event_gain_and_hit(conv_d, gru_d, q_gain=0.80, q_hit=0.90)
            gain_list.append(g_gain)
            hit_list.append(g_hit)

            # (5) 최대 동행도/최적 지연
            maxcorr, bestlag = _maxcorr_and_lag_mean(conv_d, gru_d)
            maxc_list.append(maxcorr)
            blag_list.append(bestlag)

        # ---- GRU→Conv: 생성 커널/Conv 변화/TOD 수집 ----
        if len(hooks.W_gc_list) > 0:
            W_list.append(hooks.W_gc_list[-1].numpy())  # (B, d*k)
        if (C_pre is not None) and (C_post is not None):
            delta = C_post - C_pre  # (T,B,d)
            E = torch.sqrt(torch.mean(delta ** 2, dim=0))  # (B,d)
            E_list.append(E.detach().cpu().numpy())
        if y_tod is not None:
            Y_list.append(y_tod)

    hooks.detach()

    # ---- 성능 (표준화/실측) ----
    Pz = np.concatenate(Pz_all, axis=0)
    Tz = np.concatenate(Tz_all, axis=0)
    P = np.concatenate(P_all, axis=0)
    T = np.concatenate(T_all, axis=0)
    mse_std = float(((Pz - Tz) ** 2).mean())
    mse_real = float(((P - T) ** 2).mean())
    rmse = float(np.sqrt(mse_real))

    # ---- Conv→GRU 요약 (cg_on일 때만 계산, 아니면 NaN) ----
    def _m(L):
        return float(np.mean(np.concatenate(L))) if L else float('nan')

    if cg_on:
        cg_pearson_mean = _m(cg_p_list)
        cg_spearman_mean = _m(cg_s_list)
        cg_dcor_mean = _m(cg_dcor_list)
        cg_event_gain = float(np.nanmean(gain_list)) if gain_list else np.nan
        cg_event_hit = float(np.nanmean(hit_list)) if hit_list else np.nan
        cg_maxcorr = float(np.nanmean(maxc_list)) if maxc_list else np.nan
        cg_bestlag = float(np.nanmean(blag_list)) if blag_list else np.nan
    else:
        cg_pearson_mean = np.nan
        cg_spearman_mean = np.nan
        cg_dcor_mean = np.nan
        cg_event_gain = np.nan
        cg_event_hit = np.nan
        cg_maxcorr = np.nan
        cg_bestlag = np.nan

    # ---- GRU→Conv 요약 (경로 off이면 NaN) ----
    gc_kernel_tod_dcor = np.nan
    gc_feat_tod_dcor = np.nan
    gc_feat_tod_r2 = np.nan
    gc_kernel_feat_dcor = np.nan
    gc_kernel_feat_align = np.nan

    has_W = len(W_list) > 0
    has_E = len(E_list) > 0
    has_Y = len(Y_list) > 0

    if gc_on:
        if has_Y and has_W:
            X = np.concatenate(W_list, axis=0)  # (N, d*k)
            Y = np.concatenate(Y_list, axis=0)  # (N, 2) TOD
            gc_kernel_tod_dcor = _dcor_u(X, Y)
        if has_Y and has_E:
            E = np.concatenate(E_list, axis=0)  # (N, d)
            Y = np.concatenate(Y_list, axis=0)  # (N, 2)
            gc_feat_tod_dcor = _dcor_u(E, Y)
            gc_feat_tod_r2 = _multioutput_r2(E, Y)
        if has_W and has_E:
            X = np.concatenate(W_list, axis=0)  # (N, d*k)
            E = np.concatenate(E_list, axis=0)  # (N, d)
            N, DK = X.shape
            if E.shape[1] > 0 and DK % E.shape[1] == 0:
                d = E.shape[1]
                k = DK // d
                X_sum = X.reshape(N, d, k).mean(axis=2)  # (N,d) tap 요약
                gc_kernel_feat_dcor = _dcor_u(X_sum, E)
                gc_kernel_feat_align = gc_kernel_feat_dcor

    return dict(
        mse_std=mse_std, mse_real=mse_real, rmse=rmse,
        # Conv → GRU
        cg_pearson_mean=cg_pearson_mean,
        cg_spearman_mean=cg_spearman_mean,
        cg_dcor_mean=cg_dcor_mean,
        cg_event_gain=cg_event_gain,
        cg_event_hit=cg_event_hit,
        cg_maxcorr=cg_maxcorr,
        cg_bestlag=cg_bestlag,
        # GRU → Conv
        gc_kernel_tod_dcor=gc_kernel_tod_dcor,
        gc_feat_tod_dcor=gc_feat_tod_dcor,
        gc_feat_tod_r2=gc_feat_tod_r2,
        gc_kernel_feat_dcor=gc_kernel_feat_dcor,
        gc_kernel_feat_align=gc_kernel_feat_align,
    )
