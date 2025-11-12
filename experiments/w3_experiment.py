# experiments/w3_experiment.py
# ============================================================
# W3: 데이터 구조 원인 확인 (A4‑TOD / A4‑PEAK)
#  - perturbation ∈ {"none","tod_shift","smooth"}
#  - baseline(none) 저장 후, perturb 모드에서 baseline을 자동 로드해 Δ 계산
# ============================================================

from __future__ import annotations
import copy, time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from data.dataset import build_test_tod_vector
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics
from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    save_w3_run_detail_row,
    save_w3_perturb_delta_bar,
    save_w3_rank_preservation,
    save_w3_lag_shift,
    rebuild_w3_perturb_summary,
)

from .base_experiment import BaseExperiment

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"
PERT_MODES = ("none","tod_shift","smooth")

_W3_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed",
    "perturbation","model_tag","run_tag"
]

_CG_GC_KEYS = [
    "mse_std","mae_std",
    "cg_pearson_mean","cg_spearman_mean","cg_dcor_mean",
    "cg_event_gain","cg_event_hit","cg_maxcorr","cg_bestlag",
    "gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2",
    "gc_kernel_feat_dcor","gc_kernel_feat_align"
]

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time()*1000)}")
    df.to_csv(tmp, index=False); tmp.replace(path)

def _append_or_update_results(csv_path: Path, row: Dict[str,Any], subset: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if csv_path.exists():
        old = pd.read_csv(csv_path)
        for c in new.columns:
            if c not in old.columns: old[c] = np.nan
        for c in old.columns:
            if c not in new.columns: new[c] = np.nan
        df = pd.concat([old, new], ignore_index=True)
        df.drop_duplicates(subset=subset, keep="last", inplace=True)
        tmp = csv_path.with_suffix(".csv.tmp")
        df.to_csv(tmp, index=False); tmp.replace(csv_path)
    else:
        new.to_csv(csv_path, index=False)

def _merge_cg_gc(row: dict, direct: dict | None) -> dict:
    if not isinstance(direct, dict): return row
    for k in _CG_GC_KEYS:
        if k in direct:
            try: row[k] = float(direct[k])
            except Exception: row[k] = direct[k]
    return row

class W3Experiment(BaseExperiment):
    def __init__(self, cfg: Dict[str, Any]):
        cfg_local = copy.deepcopy(cfg or {})
        cfg_local.setdefault("experiment_type", EXP_TAG)
        super().__init__(cfg_local)
        self._last_hooks_data: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}

    # ------------------------------
    # 컨텍스트 & 유틸
    # ------------------------------
    @staticmethod
    def _get(obj: Any, keys: List[str], default=None):
        for k in keys:
            if isinstance(obj, dict) and k in obj: return obj[k]
            if hasattr(obj, k): return getattr(obj, k)
        return default

    def _ctx(self, dataset: str, horizon: int, seed: int, perturbation: str, model_tag: str, run_tag: str, plot_type: str="") -> Dict[str,Any]:
        return {
            "experiment_type": EXP_TAG,
            "dataset": dataset, "plot_type": plot_type,
            "horizon": int(horizon), "seed": int(seed),
            "perturbation": str(perturbation),
            "model_tag": str(model_tag), "run_tag": str(run_tag),
        }

    # ------------------------------
    # 평가
    # ------------------------------
    def evaluate_test(self):
        tod_vec = build_test_tod_vector(self.cfg)
        # GRU 상태/게이트 모두 후킹 (가능 시)
        try:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True, collect_gru_states=True,
            )
        except TypeError:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True,
            )

        hooks: Dict[str, Any] = {}
        h = direct.pop("hooks_data", None)
        if isinstance(h, dict) and len(h) > 0:
            hooks.update(h)

        # 사양 통일: 루트에도 출력이 있을 수 있어 보강 병합
        for k in ("gate_outputs","gru_states","gates_by_stage","gate_by_stage"):
            if k in direct and k not in hooks:
                hooks[k] = direct[k]

        # 공통 지표(필요 시) 계산/병합
        try:
            exp_specific = compute_all_experiment_metrics(
                experiment_type=self.experiment_type,
                model=self.model,
                hooks_data=hooks if hooks else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=self.cfg.get("perturbation"),
            )
            if isinstance(exp_specific, dict):
                for k, v in exp_specific.items():
                    direct[k] = v
        except ImportError:
            pass

        self._last_hooks_data = copy.deepcopy(hooks)
        self._last_direct     = copy.deepcopy(direct)
        return direct

    # ------------------------------
    # 저장 & Δ 집계
    # ------------------------------
    def _save_run_and_effects(self, direct: Dict[str,Any]) -> None:
        dataset = str(self._get(self.cfg, ["dataset","dataset_tag","data_name"], self.dataset_tag))
        horizon = int(self._get(self.cfg, ["horizon","pred_len"], self.cfg.get("horizon", 96)))
        seed    = int(self._get(self.cfg, ["seed"], self.cfg.get("seed", 42)))
        perturb = str(self.cfg.get("perturbation","none")).lower()
        model_tag = str(self._get(self.model, ["model_tag"], "HybridTS"))
        run_tag = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{perturb}"

        ctx_base = self._ctx(dataset, horizon, seed, perturb, model_tag, run_tag)

        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))

        # 1) results_W3.csv (실행 단위 raw)
        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed,
            "perturbation": perturb, "model_tag": model_tag, "experiment_type": EXP_TAG,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": rmse_real, "mae": mae_real,
        }
        row = _merge_cg_gc(row, direct)
        results_csv = RESULTS_ROOT / "results_W3.csv"
        _append_or_update_results(results_csv, row,
            subset=["dataset","horizon","seed","perturbation","model_tag","experiment_type"])

        # 2) 실행 raw detail (재현용)
        run_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "run_detail"
        run_dir.mkdir(parents=True, exist_ok=True)
        save_w3_run_detail_row({**ctx_base, "plot_type": "w3_run"},
                               rmse_real, mae_real, direct,
                               detail_csv=run_dir / "w3_run_detail.csv")

        # 3) baseline이 있으면 개입효과 계산 & 그림 데이터 저장
        if perturb != "none":
            # baseline 로드
            try:
                base_df = pd.read_csv(results_csv)
                base_row = base_df[
                    (base_df["dataset"]==dataset) &
                    (base_df["horizon"]==horizon) &
                    (base_df["seed"]==seed) &
                    (base_df["perturbation"]=="none")
                ].sort_values("rmse").tail(1)  # 최근 갱신 1행
                if len(base_row)==0:
                    return
                base_row = base_row.iloc[0].to_dict()
            except Exception:
                return

            # compute_all_experiment_metrics를 통해 Δ 계산
            #  - win_errors가 direct에 들어오는 경우만 dz/순위 계산됨(없으면 NaN)
            base_win = self._last_direct.get("win_errors_base", None) or base_row.get("win_errors", None)
            pert_win = self._last_direct.get("win_errors", None)

            eff = compute_all_experiment_metrics(
                experiment_type=EXP_TAG,
                model=self.model,
                hooks_data=None,
                baseline_metrics=base_row,
                perturb_metrics=row,
                win_errors_base=base_win,
                win_errors_pert=pert_win,
                w3_dz_ci=True
            )

            # 저장 (Δ 막대)
            delta_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar"
            delta_dir.mkdir(parents=True, exist_ok=True)
            save_w3_perturb_delta_bar({**ctx_base, "plot_type": "perturb_delta_bar"}, eff, out_dir=delta_dir)

            # 저장 (순위 보존)
            rank_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "rank_preservation"
            rank_dir.mkdir(parents=True, exist_ok=True)
            save_w3_rank_preservation({**ctx_base, "plot_type": "rank_preservation"}, eff, out_dir=rank_dir)

            # 저장 (라그 변화: PEAK 전용)
            if "cg_bestlag" in row or "cg_bestlag" in base_row:
                lag_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "lag_shift"
                lag_dir.mkdir(parents=True, exist_ok=True)
                save_w3_lag_shift({**ctx_base, "plot_type": "lag_shift"},
                                  base_bestlag=base_row.get("cg_bestlag", np.nan),
                                  pert_bestlag=row.get("cg_bestlag", np.nan),
                                  out_dir=lag_dir)

            # (선택) 그룹 요약 재작성
            try:
                rebuild_w3_perturb_summary(
                    delta_detail_csv=delta_dir/"perturb_delta_bar_detail.csv",
                    delta_summary_csv=delta_dir/"perturb_delta_bar_summary_by_group.csv",
                    group_keys=("experiment_type","dataset","horizon","perturbation"),
                )
            except Exception:
                pass

    # ------------------------------
    # 실행
    # ------------------------------
    def run(self) -> bool:
        try:
            self.train()
            direct = self.evaluate_test()
            self._save_run_and_effects(direct)
            return True
        finally:
            self.cleanup()
