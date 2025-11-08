"""
W3 실험 특화 지표: 데이터 교란 효과
"""

import numpy as np
from typing import Dict, Optional


def _safe_div(a, b, eps=1e-12):
    """안전한 나눗셈"""
    return a / (b + eps)


def paired_cohens_dz(diff_vec: np.ndarray) -> float:
    """
    Paired Cohen's d_z = mean(diff) / std(diff), ddof=1
    
    Args:
        diff_vec: (N,) 윈도우별 오차 차이 = perturbed - baseline
    
    Returns:
        Cohen's d_z 값
    """
    diff_vec = np.asarray(diff_vec, dtype=np.float64)
    if diff_vec.ndim != 1 or diff_vec.size < 2:
        return np.nan
    
    mu = np.mean(diff_vec)
    sd = np.std(diff_vec, ddof=1)
    
    if not np.isfinite(sd) or sd <= 0:
        return np.nan
    
    return float(mu / sd)


def bootstrap_ci(stat_fn, data, n_boot=500, ci=95, rng=None):
    """
    부트스트랩 신뢰구간 계산 (선택적)
    
    Args:
        stat_fn: data -> scalar 함수 (예: lambda x: np.mean(x)/np.std(x, ddof=1))
        data: 1D array
        n_boot: 부트스트랩 반복 횟수
        ci: 신뢰구간 퍼센트 (기본 95%)
        rng: 난수 생성기
    
    Returns:
        (lower, upper) 신뢰구간
    """
    if rng is None:
        rng = np.random.default_rng(0)
    
    data = np.asarray(data)
    if data.ndim != 1 or data.size < 2:
        return (np.nan, np.nan)
    
    stats = []
    n = data.size
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(stat_fn(data[idx]))
    
    lo = (100 - ci) / 2.0
    hi = 100 - lo
    return (float(np.nanpercentile(stats, lo)), float(np.nanpercentile(stats, hi)))


def compute_w3_metrics(
    baseline_metrics: Dict,
    perturb_metrics: Dict,
    win_errors_base: Optional[np.ndarray] = None,
    win_errors_pert: Optional[np.ndarray] = None,
    *,
    dz_ci: bool = False,
    **kwargs
) -> Dict:
    """
    W3 특화 지표 계산
    
    Args:
        baseline_metrics: baseline 스칼라 지표 {'rmse', 'gc_kernel_tod_dcor', 'cg_event_gain', ...}
        perturb_metrics: 교란 스칼라 지표
        win_errors_base: 테스트셋 윈도우별 오차 벡터 (N,) - baseline
        win_errors_pert: 테스트셋 윈도우별 오차 벡터 (N,) - perturbed
        dz_ci: True면 부트스트랩 CI 계산
    
    Returns:
        dict with keys:
        - w3_rmse_base, w3_rmse_perturb: 원본 RMSE 값
        - w3_tod_sens_base, w3_tod_sens_perturb: 원본 TOD 민감도
        - w3_peak_resp_base, w3_peak_resp_perturb: 원본 피크 반응
        - w3_intervention_effect_rmse: ΔRMSE (상대 변화율)
        - w3_intervention_effect_tod: ΔTOD (절대 차이)
        - w3_intervention_effect_peak: ΔPeak (절대 차이)
        - w3_intervention_cohens_d: Paired Cohen's d_z
        - w3_rank_preservation_rate: P(perturbed > baseline) - 순위 보존률
        - w3_lag_distribution_change: 라그 분포 변화
        - w3_cohens_d_ci_low, w3_cohens_d_ci_high: CI (선택적)
    """
    out = {}
    
    # 1) RMSE: 원본 값 저장 + 상대 변화율 계산
    rmse_b = float(baseline_metrics.get('rmse', np.nan))
    rmse_p = float(perturb_metrics.get('rmse', np.nan))
    
    out['w3_rmse_base'] = rmse_b
    out['w3_rmse_perturb'] = rmse_p
    
    if np.isfinite(rmse_b) and rmse_b > 0 and np.isfinite(rmse_p):
        out['w3_intervention_effect_rmse'] = _safe_div((rmse_p - rmse_b), rmse_b)
    else:
        out['w3_intervention_effect_rmse'] = np.nan
    
    # 2) TOD 민감도: 원본 값 저장 + 절대 차이 계산
    tod_b = float(baseline_metrics.get('gc_kernel_tod_dcor', np.nan))
    tod_p = float(perturb_metrics.get('gc_kernel_tod_dcor', np.nan))
    
    out['w3_tod_sens_base'] = tod_b
    out['w3_tod_sens_perturb'] = tod_p
    
    if np.isfinite(tod_b) and np.isfinite(tod_p):
        out['w3_intervention_effect_tod'] = tod_p - tod_b
    else:
        out['w3_intervention_effect_tod'] = np.nan
    
    # 3) 피크 반응: 원본 값 저장 + 절대 차이 계산
    peak_b = float(baseline_metrics.get('cg_event_gain', np.nan))
    peak_p = float(perturb_metrics.get('cg_event_gain', np.nan))
    
    out['w3_peak_resp_base'] = peak_b
    out['w3_peak_resp_perturb'] = peak_p
    
    if np.isfinite(peak_b) and np.isfinite(peak_p):
        out['w3_intervention_effect_peak'] = peak_p - peak_b
    else:
        out['w3_intervention_effect_peak'] = np.nan
    
    # 4) Paired Cohen's d_z (윈도우별 오차 벡터 사용)
    if win_errors_base is not None and win_errors_pert is not None:
        win_errors_base = np.asarray(win_errors_base).reshape(-1)
        win_errors_pert = np.asarray(win_errors_pert).reshape(-1)
        
        if win_errors_base.size >= 2 and win_errors_base.size == win_errors_pert.size:
            diff = win_errors_pert - win_errors_base
            dz = paired_cohens_dz(diff)
            out['w3_intervention_cohens_d'] = dz
            
            # 순위 보존률: P(perturbed > baseline) = 교란으로 악화된 비율
            out['w3_rank_preservation_rate'] = float(np.mean(win_errors_pert > win_errors_base))
            
            # 선택적 부트스트랩 CI
            if dz_ci:
                lo, hi = bootstrap_ci(
                    lambda x: np.mean(x) / np.std(x, ddof=1) if np.std(x, ddof=1) > 0 else np.nan,
                    diff,
                    n_boot=500,
                    ci=95
                )
                out['w3_cohens_d_ci_low'] = lo
                out['w3_cohens_d_ci_high'] = hi
        else:
            out['w3_intervention_cohens_d'] = np.nan
            out['w3_rank_preservation_rate'] = np.nan
            if dz_ci:
                out['w3_cohens_d_ci_low'] = np.nan
                out['w3_cohens_d_ci_high'] = np.nan
    else:
        # 윈도우별 오차가 없으면 NaN
        out['w3_intervention_cohens_d'] = np.nan
        out['w3_rank_preservation_rate'] = np.nan
        if dz_ci:
            out['w3_cohens_d_ci_low'] = np.nan
            out['w3_cohens_d_ci_high'] = np.nan
    
    # 5) 라그 분포 변화: baseline과 perturbed의 cg_bestlag 비교
    bestlag_b = baseline_metrics.get('cg_bestlag', np.nan)
    bestlag_p = perturb_metrics.get('cg_bestlag', np.nan)
    
    if np.isfinite(bestlag_b) and np.isfinite(bestlag_p):
        # 단일 스칼라 값인 경우: 절대 차이
        out['w3_lag_distribution_change'] = float(abs(bestlag_p - bestlag_b))
    else:
        out['w3_lag_distribution_change'] = np.nan
    
    return out
