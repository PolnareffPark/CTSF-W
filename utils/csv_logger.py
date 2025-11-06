"""
CSV 결과 저장 모듈
실험 결과를 CSV로 저장하고 중복 제거
"""

import pandas as pd
import numpy as np
from pathlib import Path


def save_results_unified(row: dict, out_root: str = "results"):
    """
    통합 CSV에 결과 저장 (업서트 방식)
    
    Args:
        row: 결과 딕셔너리 (dataset, horizon, seed, mode, model_tag, 지표들 포함)
        out_root: 결과 저장 루트 디렉토리
    """
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    fpath = out_root / "results.csv"

    # 컬럼 순서 정의
    cols_order = [
        "dataset", "horizon", "seed", "mode", "model_tag",
        "mse_std", "mse_real", "rmse", "mae",
        # Conv→GRU
        "cg_pearson_mean", "cg_spearman_mean", "cg_dcor_mean",
        "cg_event_gain", "cg_event_hit", "cg_maxcorr", "cg_bestlag",
        # GRU→Conv
        "gc_kernel_tod_dcor", "gc_feat_tod_dcor", "gc_feat_tod_r2",
        "gc_kernel_feat_dcor", "gc_kernel_feat_align",
    ]

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


def read_results_rows(results_root="results"):
    """
    기존 결과 읽기 (resume 기능용)
    
    Returns:
        (rows, done_set): 완료된 실험 조합 리스트와 집합
    """
    f = Path(results_root) / "results.csv"
    if not f.exists():
        return [], set()

    df = pd.read_csv(f)
    # 안전 변환
    if "dataset" in df.columns:
        df["dataset"] = df["dataset"].astype(str).str.strip()
    if "mode" in df.columns:
        df["mode"] = df["mode"].astype(str).str.strip()
    if "horizon" in df.columns:
        df["horizon"] = df["horizon"].apply(lambda x: int(x) if pd.notna(x) else None)
    if "seed" in df.columns:
        df["seed"] = df["seed"].apply(lambda x: int(x) if pd.notna(x) else None)

    rows = []
    done = set()
    for _, r in df.iterrows():
        key = (
            str(r.get("dataset", "")).strip(),
            int(r.get("horizon", 0)) if pd.notna(r.get("horizon")) else None,
            int(r.get("seed", 0)) if pd.notna(r.get("seed")) else None,
            str(r.get("mode", "")).strip()
        )
        rows.append(key)
        done.add(key)

    return rows, done
