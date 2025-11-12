# ============================================================
# experiments/w3_experiment.py
#   - W3: 데이터 구조 원인 확인 (none / tod_shift / smooth)
#   - per-run: 원자료 저장 (results_W3.csv, gate_* CSV들) → 사후 집계에서 Δ
# ============================================================

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .base_experiment import BaseExperiment
from data.dataset import build_test_tod_vector
from models.ctsf_model import HybridTS

# (W3는 모델 구조 동일: 동적 교차 모델. 데이터 교란은 loader/전처리에 의해 이뤄짐)
# W1/W2에 영향 없도록 W3에서만 사용
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics

# W3 전용 그림 저장 유틸(요약/상세)
from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    save_w3_forest_detail_row,
    rebuild_w3_forest_summary,
    compute_and_save_w3_gate_tod_heatmap,
    compute_and_save_w3_gate_distribution,
    rebuild_w3_perturb_delta_bar,   # Δ 지표 집계
)

# (간단한 게이트 패키징 유틸 – W3 전용)
from utils.experiment_plotting_metrics.all_plotting_metrics import (
    infer_tod_bins_by_dataset,
)

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"
MODES = ("none", "tod_shift", "smooth")  # README/계획과 동일


# ---------------------------
# 공통 유틸
# ---------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


_CG_GC_KEYS = [
    # conv->gru 상호작용(cg_*), gru->conv 민감도(gc_*) 핵심 지표
    "cg_pearson_mean", "cg_spearman_mean", "cg_dcor_mean",
    "cg_event_gain", "cg_event_hit", "cg_maxcorr", "cg_bestlag",
    "gc_kernel_tod_dcor", "gc_feat_tod_dcor", "gc_feat_tod_r2",
    "gc_kernel_feat_dcor", "gc_kernel_feat_align",
    # 분산 등 로그형 지표(있으면 흡수)
    "mse_std","mae_std"
]

def _merge_cg_gc(row: dict, de: Optional[dict]) -> dict:
    if not isinstance(de, dict):
        return row
    for k in _CG_GC_KEYS:
        if k in de:
            try:
                row[k] = float(de[k])
            except Exception:
                row[k] = de[k]
    return row


# ============================================================
# W3 Experiment
# ============================================================
class W3Experiment(BaseExperiment):
    """
    W3: 데이터 구조 원인 확인 (교란 시험)
      - mode ∈ {'none','tod_shift','smooth'}
      - per-run: 결과 원자료 저장(results_W3.csv, gate_* CSV들)
      - 사후: forest Δ, perturb_delta_bar Δ 집계
    """

    def __init__(self, cfg: Dict[str, Any]):
        cfg_local = copy.deepcopy(cfg or {})
        cfg_local.setdefault("experiment_type", EXP_TAG)
        # ★ BaseExperiment를 '상속'하여 초기화: 내부에서 BaseExperiment(...)를 별도 생성하면 안 됨
        super().__init__(cfg_local)
        self._last_hooks_data: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}

    # ------------------------------
    # BaseExperiment 구현부
    # ------------------------------
    def _create_model(self):
        """
        - W3 모델: HybridTS(동적 교차). 교란은 데이터 레벨에서 처리.
        - set_cross_directions(use_gc=True, use_cg=True) 시도(있으면).
        """
        model = HybridTS(self.cfg, self.n_vars)
        if hasattr(model, "set_cross_directions"):
            try:
                model.set_cross_directions(use_gc=True, use_cg=True)
            except Exception:
                pass
        return model

    # (선택) 교란 모드별 train_loader 래핑을 여기서 처리하고 싶다면
    # BaseExperiment의 loader 생성 이후(self.train_loader) 교체하는 방식으로 구현할 수 있습니다.
    # 단, W1/W2에 영향 주지 않기 위해 W3 내부에서만 건드리세요.
    # 본 패치는 교란이 없어도 안전하게 작동하도록 설계(Δ≈0)되어 있습니다.

    def evaluate_test(self):
        """
        - 게이트/GRU 상태 후킹을 최대한 흡수(키 다양성 허용).
        - W3는 per-run 원자료 저장이 목적이므로, 여기서는 direct_evidence를 풍부히 보강만.
        """
        tod_vec = build_test_tod_vector(self.cfg)
        try:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True,  # gate_outputs, gates_by_stage, ...
                collect_gru_states=True,    # gru_states (가능 시)
            )
        except TypeError:
            # 구버전 호환
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True,
            )

        # hooks_data 흡수(루트/서브 모든 위치 허용)
        hooks_data: Dict[str, Any] = {}
        w_hooks = direct.pop("hooks_data", None)
        if isinstance(w_hooks, dict) and w_hooks:
            hooks_data.update(w_hooks)
        for k in ("gate_outputs","gru_states","gates_by_stage","gate_by_stage","gate_logs_by_stage"):
            if k in direct and k not in hooks_data:
                hooks_data[k] = direct[k]

        # (선택) 공통 특화 지표가 있다면 병합 – W3는 per-run 단계에서 Δ를 계산하지 않음
        try:
            extra = compute_all_experiment_metrics(
                experiment_type=self.experiment_type,
                model=self.model,
                hooks_data=hooks_data if hooks_data else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=self.cfg.get("mode", "none"),
                active_layers=[],
            )
            if isinstance(extra, dict):
                for k, v in extra.items():
                    direct[k] = v
        except Exception:
            pass

        self._last_hooks_data = copy.deepcopy(hooks_data)
        self._last_direct = copy.deepcopy(direct)
        return direct

    # ------------------------------
    # 내부: 게이트 패키징(시간대 평균/분포)
    # ------------------------------
    def _extract_gate_raw(
        self, hooks: Dict[str, Any]
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """
        반환:
          gates_by_stage: dict('S'|'M'|'D' -> np.ndarray)
          gates_all:      np.ndarray (전 채널/스테이지 평탄화 벡터; 분포 요약용)
        """
        # 1) stage dict 최우선
        for key in ["gates_by_stage", "gate_by_stage", "gate_logs_by_stage"]:
            if key in hooks and isinstance(hooks[key], dict):
                gbs = {}
                for s in ("S","M","D","s","m","d"):
                    if s in hooks[key]:
                        gbs[s.upper()] = np.asarray(hooks[key][s])
                if gbs:
                    ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                    return gbs, ga

        # 2) gate_outputs 목록이 있으면 평균으로 근사
        if "gate_outputs" in hooks and hooks["gate_outputs"]:
            arrs = hooks["gate_outputs"]  # list of arrays
            try:
                cat = np.concatenate([np.asarray(a) for a in arrs], axis=0)  # (N, ...)
            except Exception:
                cat = np.asarray(arrs[0])
            # 스테이지 구분 불가 시 전체를 'M'으로 뭉텅이 처리
            gbs = {"M": np.asarray(cat, dtype=float)}
            ga = np.ravel(cat.astype(float))
            return gbs, ga

        # 3) 아무것도 없으면 빈값
        return {}, np.array([], dtype=float)

    # ------------------------------
    # 저장/그림 파이프 (per-run)
    # ------------------------------
    def _after_eval_save(
        self,
        dataset: str, horizon: int, seed: int, mode: str,
        direct: Dict[str, Any], hooks: Dict[str, Any],
        model_tag: str, run_tag: str
    ) -> None:
        """
        per-run에서 다음을 저장:
          - forest_plot_detail.csv (RMSE/MAE)
          - gate_tod_heatmap_(summary|detail).csv
          - gate_distribution_(summary|detail).csv
          - results_W3.csv (원자료 + cg_*/gc_* + gate 요약 스칼라)
        사후 집계:
          - forest Δ (none vs perturbed)
          - perturb_delta_bar Δ (TOD/PEAK 핵심지표 Δ)
        """
        ctx = {
            "experiment_type": EXP_TAG, "dataset": str(dataset), "horizon": int(horizon),
            "seed": int(seed), "mode": str(mode), "model_tag": str(model_tag), "run_tag": str(run_tag)
        }

        # 1) forest raw(detail)
        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))
        fdir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        fdir.mkdir(parents=True, exist_ok=True)
        save_w3_forest_detail_row({**ctx, "plot_type": "forest_plot"},
                                  rmse_real=rmse_real, mae_real=mae_real,
                                  detail_csv=fdir / "forest_plot_detail.csv")

        # 2) gate raw → Heatmap/Distribution
        gbs, gall = self._extract_gate_raw(hooks)
        tod_bins = infer_tod_bins_by_dataset(dataset)
        max_T = max(
            len(gbs.get("S", [])) if isinstance(gbs.get("S", []), (list, np.ndarray)) else 0,
            len(gbs.get("M", [])) if isinstance(gbs.get("M", []), (list, np.ndarray)) else 0,
            len(gbs.get("D", [])) if isinstance(gbs.get("D", []), (list, np.ndarray)) else 0,
            tod_bins,
        )
        if max_T <= 0: max_T = tod_bins
        tod_labels = list(range(max_T))

        # Heatmap (S/M/D 시간대 평균)
        tdir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
        tdir.mkdir(parents=True, exist_ok=True)
        compute_and_save_w3_gate_tod_heatmap(
            ctx={**ctx, "plot_type": "gate_tod_heatmap"},
            gates_by_stage=gbs, tod_labels=tod_labels,
            out_dir=tdir, amp_method="range"
        )

        # Distribution (전역 분포/분위수/채널통계)
        ddir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
        ddir.mkdir(parents=True, exist_ok=True)
        compute_and_save_w3_gate_distribution(
            ctx={**ctx, "plot_type": "gate_distribution"},
            gates_by_stage=gbs, gates_all=gall,
            out_dir=ddir
        )

        # gate 요약 스칼라(결과 CSV용; Heatmap/Distribution 단계에서 계산된 summary와 일치)
        # 간단히 detail/summary 파일을 읽지 않고도 scalar를 재계산 가능하도록 최소 통계만 취합
        gate_mean = float(np.nanmean(gall)) if gall.size > 0 and np.isfinite(gall).any() else np.nan
        gate_std  = float(np.nanstd(gall))  if gall.size > 0 and np.isfinite(gall).any() else np.nan
        gate_entropy = np.nan
        if gall.size > 0:
            pos = gall[np.isfinite(gall) & (gall > 1e-8)]
            if pos.size > 0:
                p = pos / (pos.sum() + 1e-12)
                gate_entropy = float(-np.sum(p * np.log(p + 1e-12)))

        # TOD 평균(스테이지별 평균의 평균) 및 민감도(range 평균)
        def _stage_mean(stage: str) -> float:
            if stage in gbs and isinstance(gbs[stage], np.ndarray) and gbs[stage].size > 0:
                a = np.asarray(gbs[stage], float)
                # 시간축만 남기고 평균(다른 축은 평균)
                if a.ndim > 1:
                    axes = tuple(i for i in range(a.ndim - 1))
                    a = np.nanmean(a, axis=axes)
                return float(np.nanmean(a))
            return np.nan

        gate_tod_mean_s = _stage_mean("S")
        gate_tod_mean_m = _stage_mean("M")
        gate_tod_mean_d = _stage_mean("D")

        def _amp(x: np.ndarray) -> float:
            if x.size == 0 or not np.isfinite(x).any(): return np.nan
            return float(np.nanmax(x) - np.nanmin(x))

        amps = []
        for st in ("S","M","D"):
            if st in gbs and isinstance(gbs[st], np.ndarray) and gbs[st].size > 0:
                a = np.asarray(gbs[st], float)
                # 시간축만 남기고 평균
                if a.ndim > 1:
                    axes = tuple(i for i in range(a.ndim - 1))
                    a = np.nanmean(a, axis=axes)
                amps.append(_amp(a))
        tod_amp_mean = float(np.nanmean([v for v in amps if np.isfinite(v)])) if amps else np.nan

        # 분위수/상위 히스토그램
        gate_q10 = gate_q50 = gate_q90 = gate_hist10 = np.nan
        if gall.size > 0 and np.isfinite(gall).any():
            fin = gall[np.isfinite(gall)]
            if fin.size >= 5:
                gate_q10, gate_q50, gate_q90 = np.quantile(fin, [0.10, 0.50, 0.90]).tolist()
                gate_hist10 = float((fin > gate_q90).mean())

        # 3) results_W3.csv (per-run 원자료 1행)
        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed, "mode": mode,
            "model_tag": model_tag, "experiment_type": EXP_TAG,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": rmse_real, "mae": mae_real,

            # 게이트 전역 요약
            "gate_mean": gate_mean, "gate_std": gate_std, "gate_entropy": gate_entropy,
            "gate_q10": gate_q10, "gate_q50": gate_q50, "gate_q90": gate_q90, "gate_hist10": gate_hist10,

            # TOD 요약
            "gate_tod_mean_s": gate_tod_mean_s,
            "gate_tod_mean_m": gate_tod_mean_m,
            "gate_tod_mean_d": gate_tod_mean_d,
            "tod_amp_mean": tod_amp_mean,
        }
        row = _merge_cg_gc(row, direct)  # cg_*/gc_* 병합

        csv_path = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        self._append_or_update_results(csv_path, row,
            subset=["dataset","horizon","seed","mode","model_tag","experiment_type"])

        # 4) 사후 집계(Δ): baseline('none') ↔ perturbed('tod_shift'/'smooth') 매칭
        #    - 누적적으로 안전(idempotent)
        rebuild_w3_forest_summary(
            detail_csv=fdir / "forest_plot_detail.csv",
            summary_csv=fdir / "forest_plot_summary.csv",
            gate_tod_summary_csv=tdir / "gate_tod_heatmap_summary.csv",
            gate_dist_summary_csv=ddir / "gate_distribution_summary.csv",
            baseline_mode="none",
            perturbed_modes=("tod_shift","smooth"),
            group_keys=("experiment_type","dataset","horizon")
        )
        # 원인 지표 Δ bar (TOD/PEAK)
        rebuild_w3_perturb_delta_bar(
            results_csv=RESULTS_ROOT / f"results_{EXP_TAG}.csv",
            out_dir=RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar",
            baseline_mode="none"
        )

    def _append_or_update_results(self, csv_path: Path, row: Dict[str, Any], subset: List[str]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame([row])
        if csv_path.exists():
            df_old = pd.read_csv(csv_path)
            for c in df_new.columns:
                if c not in df_old.columns:
                    df_old[c] = np.nan
            for c in df_old.columns:
                if c not in df_new.columns:
                    df_new[c] = np.nan
            df = pd.concat([df_old, df_new], ignore_index=True)
            df.drop_duplicates(subset=subset, keep="last", inplace=True)
            tmp = csv_path.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(csv_path)
        else:
            df_new.to_csv(csv_path, index=False)

    # ------------------------------
    # 메인 엔트리
    # ------------------------------
    def run(self) -> bool:
        try:
            self.train()
            direct = self.evaluate_test()
            hooks = copy.deepcopy(getattr(self, "_last_hooks_data", {}))

            dataset = str(getattr(self, "dataset_tag", self.cfg.get("dataset", "UNKNOWN")))
            horizon = int(self.cfg.get("horizon", self.cfg.get("pred_len", 96)))
            seed = int(self.cfg.get("seed", 42))
            mode = str(self.cfg.get("mode", "none"))
            model_tag = str(self.cfg.get("model_tag", getattr(self.model, "model_tag", "HyperConv")))
            run_tag = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}"

            self._after_eval_save(
                dataset=dataset, horizon=horizon, seed=seed, mode=mode,
                direct=direct, hooks=hooks, model_tag=model_tag, run_tag=run_tag
            )
            return True
        except Exception:
            return False
