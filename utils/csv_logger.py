"""
CSV 결과 저장 모듈
실험 결과를 CSV로 저장하고 중복 제거
"""

import pandas as pd
import numpy as np
from pathlib import Path


def save_results_unified(row: dict, out_root: str = "results", experiment_type: str = None):
    """
    실험별 CSV에 결과 저장 (업서트 방식)
    
    Args:
        row: 결과 딕셔너리 (dataset, horizon, seed, mode, model_tag, 지표들 포함)
        out_root: 결과 저장 루트 디렉토리
        experiment_type: 실험 타입 (W1, W2, W3, W4, W5) - None이면 row에서 추출
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    
    # 실험 타입 추출
    if experiment_type is None:
        experiment_type = row.get("experiment_type", "W1")
    
    fpath = out_root / f"results_{experiment_type}.csv"

    # 컬럼 순서 정의 (공통 지표 + 실험별 특화 지표)
    base_cols = [
        "dataset", "horizon", "seed", "mode", "model_tag", "experiment_type",
        "mse_std", "mse_real", "rmse", "mae",
        # Conv→GRU
        "cg_pearson_mean", "cg_spearman_mean", "cg_dcor_mean",
        "cg_event_gain", "cg_event_hit", "cg_maxcorr", "cg_bestlag",
        # GRU→Conv
        "gc_kernel_tod_dcor", "gc_feat_tod_dcor", "gc_feat_tod_r2",
        "gc_kernel_feat_dcor", "gc_kernel_feat_align",
    ]
    
    # 실험별 특화 지표 컬럼
    exp_specific_cols = {
        "W1": [
            "w1_cka_similarity_cnn_gru", "w1_cca_similarity_cnn_gru",
            "w1_layerwise_upward_improvement", "w1_inter_path_gradient_align",
            # 보고용 그림 지표
            "cka_s", "cka_m", "cka_d",  # 층별 CKA
            "grad_align_s", "grad_align_m", "grad_align_d",  # 층별 그래디언트 정렬
        ],
        "W2": [
            "w2_gate_variability_time", "w2_gate_variability_sample", "w2_gate_entropy",
            "w2_gate_tod_alignment", "w2_gate_gru_state_alignment",
            "w2_event_conditional_response",
            "w2_channel_selectivity_kurtosis", "w2_channel_selectivity_sparsity",
            # 보고용 그림 지표
            "gate_tod_mean_s", "gate_tod_mean_m", "gate_tod_mean_d",  # 시간대별 게이트
            "gate_var_t", "gate_var_b", "gate_entropy",  # 게이트 분포
            "gate_channel_kurt", "gate_channel_sparsity",  # 채널 선택도
            "gate_q10", "gate_q50", "gate_q90", "gate_hist10",  # 게이트 분위수/히스토
        ],
        "W3": [
            "w3_intervention_effect_rmse", "w3_intervention_effect_tod", "w3_intervention_effect_peak",
            "w3_intervention_cohens_d", "w3_rank_preservation_rate", "w3_lag_distribution_change",
            # 보고용 그림 지표
            "bestlag_neg_ratio", "bestlag_var", "bestlag_hist21",  # 라그 분포
        ],
        "W4": [
            "w4_layer_contribution_score", "w4_layerwise_gate_usage",
            "w4_layerwise_representation_similarity",
            # 보고용 그림 지표
            "cka_s", "cka_m", "cka_d",  # 층별 CKA
        ],
        "W5": [
            "w5_performance_degradation_ratio", "w5_sensitivity_gain_loss",
            "w5_event_gain_loss", "w5_gate_event_alignment_loss",
            # 보고용 그림 지표
            "gate_var_t", "gate_var_b", "gate_entropy",  # 게이트 분포
            "gate_q10", "gate_q50", "gate_q90", "gate_hist10",  # 게이트 분위수/히스토
        ],
    }
    
    # 실험 타입에 맞는 컬럼 선택
    exp_cols = exp_specific_cols.get(experiment_type, [])
    cols_order = base_cols + exp_cols

    # 누락된 컬럼은 NaN으로 채우기
    for c in cols_order:
        if c not in row:
            row[c] = np.nan

    new_df = pd.DataFrame([row])[cols_order]

    if fpath.exists():
        df = pd.read_csv(fpath)
        # 중복 제거 (dataset, horizon, seed, mode로 판단)
        mask = (
            (df["dataset"] == row["dataset"]) &
            (df["horizon"] == row["horizon"]) &
            (df["seed"] == row["seed"]) &
            (df["mode"] == row["mode"])
        )
        df = df[~mask]
        df = pd.concat([df, new_df], ignore_index=True)
        df.to_csv(fpath, index=False)
    else:
        new_df.to_csv(fpath, index=False)

    return str(fpath)


def read_results_rows(results_root="results", experiment_type=None):
    """
    기존 결과 읽기 (resume 기능용)
    
    Args:
        results_root: 결과 루트 디렉토리
        experiment_type: 실험 타입 (W1, W2, ...) - None이면 모든 실험 파일 검색
    
    Returns:
        (rows, done_set): 완료된 실험 조합 리스트와 집합
    """
    rows = []
    done = set()
    
    # 실험별 CSV 파일 읽기
    if experiment_type:
        files = [Path(results_root) / f"results_{experiment_type}.csv"]
    else:
        # 모든 실험 파일 검색
        files = list(Path(results_root).glob("results_W*.csv"))
    
    for f in files:
        if not f.exists():
            continue
        
        df = pd.read_csv(f)
        # 안전 변환
        if "dataset" in df.columns:
            df["dataset"] = df["dataset"].astype(str).str.strip()
        if "mode" in df.columns:
            df["mode"] = df["mode"].astype(str).str.strip()
        if "horizon" in df.columns:
            df["horizon"] = df["horizon"].apply(lambda x: int(float(x)) if pd.notna(x) else None)
        if "seed" in df.columns:
            df["seed"] = df["seed"].apply(lambda x: int(float(x)) if pd.notna(x) else None)

        for _, r in df.iterrows():
            key = (
                str(r.get("dataset", "")).strip(),
                int(float(r.get("horizon", 0))) if pd.notna(r.get("horizon")) else None,
                int(float(r.get("seed", 0))) if pd.notna(r.get("seed")) else None,
                str(r.get("mode", "")).strip()
            )
            rows.append(key)
            done.add(key)

    return rows, done
