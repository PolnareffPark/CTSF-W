# ============================================================
# experiments/w3_experiment.py
#   - W3: 원인 교란(perturbation) 전/후를 한 실험에서 쌍으로 평가/저장
#   - 결과/그림: W1/W2와 동일 스키마 준수
# ============================================================

from __future__ import annotations
import copy, time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from .base_experiment import BaseExperiment
from data.dataset import build_test_tod_vector
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics
from utils.experiment_metrics.w3_metrics import (
    compute_w3_single_metrics, compute_w3_pair_metrics
)

from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    save_w3_forest_detail_row,
    rebuild_w3_forest_summary,
    save_w3_perturb_delta_bar,
)

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"

_W3_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed","mode","model_tag",
    "run_tag","perturb_type","condition"  # 'baseline' or 'perturbed'
]

def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time()*1000)}")
    df.to_csv(tmp, index=False); tmp.replace(path)

class W3Experiment(BaseExperiment):
    """
    - 한 번의 run 안에서 baseline → perturbed(예: 'tod_shuffle', 'peak_weakening') 순서로 평가
    - 각 조건의 결과를 results_W3.csv 에 기록
    - perturbed 행에는 baseline 대비 Δ지표를 병합하여 기록
    - Forest raw(detail) 2행 저장 → summary 재작성
    """
    def __init__(self, cfg: Dict[str, Any]):
        cfg_local = copy.deepcopy(cfg or {})
        cfg_local.setdefault("experiment_type", EXP_TAG)
        super().__init__(cfg_local)
        self._last_hooks: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}

    # ------------------------------
    # 컨텍스트/도우미
    # ------------------------------
    def _ctx(self, dataset: str, horizon: int, seed: int, mode: str,
             model_tag: str, run_tag: str, perturb_type: str, condition: str,
             plot_type: str = "") -> Dict[str,Any]:
        return {
            "experiment_type": EXP_TAG,
            "dataset": dataset,
            "plot_type": plot_type,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": str(mode),
            "model_tag": str(model_tag),
            "run_tag": str(run_tag),
            "perturb_type": str(perturb_type),
            "condition": str(condition),
        }

    def _append_or_update_results(self, csv_path: Path, row: Dict[str,Any], subset: List[str]) -> None:
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

    # ------------------------------
    # 평가: 한 조건(baseline/perturbed)
    # ------------------------------
    def _evaluate_condition(
        self,
        condition: str,                     # 'baseline' or 'perturbed'
        perturb_type: Optional[str],        # None or 'tod_shuffle'/'peak_weakening'/...
    ) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        """
        evaluate_with_direct_evidence 호출 시, perturbed이면 perturbation_type을 전달.
        direct + hooks_data 반환.
        """
        tod_vec = build_test_tod_vector(self.cfg)
        kwargs = dict(
            model=self.model,
            data_loader=self.test_loader,
            mu=self.mu, std=self.std,
            tod_vec=tod_vec,
            device=self.device,
            collect_gate_outputs=True,
        )
        try:
            # 새 시그니처(키워드명 다를 수 있어 kwargs로 안전 전달)
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True,
                perturbation_type=perturb_type if condition=="perturbed" else None
            )
        except TypeError:
            # 구버전 시그니처 호환(perturbation_type 미지원)
            direct = evaluate_with_direct_evidence(**kwargs)

        hooks_data: Dict[str,Any] = {}
        w_hooks = direct.pop("hooks_data", None)
        if isinstance(w_hooks, dict) and len(w_hooks)>0:
            hooks_data.update(w_hooks)

        # 공통 추가 지표(있는 경우에만): W1/W2와 동일 호출 패턴
        try:
            extra = compute_all_experiment_metrics(
                experiment_type=self.experiment_type,
                model=self.model,
                hooks_data=hooks_data if hooks_data else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=perturb_type if condition=="perturbed" else None,
                active_layers=[]
            )
            if isinstance(extra, dict):
                direct.update(extra)
        except ImportError:
            pass

        # 단일 조건 보조(게이트 분포)
        try:
            single = compute_w3_single_metrics(hooks_data)
            if isinstance(single, dict):
                direct.update(single)
        except Exception:
            pass

        return direct, hooks_data

    # ------------------------------
    # 저장: 한 조건(baseline/perturbed)
    # ------------------------------
    def _save_condition(
        self,
        dataset: str, horizon: int, seed: int, mode: str, model_tag: str,
        perturb_type: str, condition: str, direct: Dict[str,Any], hooks: Dict[str,Any]
    ) -> Dict[str,Any]:
        run_tag = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}_{condition}_{perturb_type}"
        ctx = self._ctx(dataset, horizon, seed, mode, model_tag, run_tag, perturb_type, condition)

        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))

        # Forest raw(detail)
        fdir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        fdir.mkdir(parents=True, exist_ok=True)
        save_w3_forest_detail_row({**ctx, "plot_type": "forest_plot"}, rmse_real=rmse_real, mae_real=mae_real,
                                  detail_csv=fdir/"forest_plot_detail.csv")

        # results_W3.csv 1행 저장
        row = {
            "experiment_type": EXP_TAG,
            "dataset": dataset, "horizon": int(horizon), "seed": int(seed),
            "mode": mode, "model_tag": model_tag, "perturb_type": perturb_type, "condition": condition,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": rmse_real, "mae": mae_real,
            # 보조 통계(있으면)
            "gate_mean": float(direct.get("gate_mean", np.nan)),
            "gate_std":  float(direct.get("gate_std",  np.nan)),
            "gate_entropy": float(direct.get("gate_entropy", np.nan)),
            "gate_kurtosis": float(direct.get("gate_kurtosis", np.nan)),
            "gate_sparsity": float(direct.get("gate_sparsity", np.nan)),
        }
        res_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        self._append_or_update_results(res_csv, row,
            subset=["experiment_type","dataset","horizon","seed","mode","model_tag","perturb_type","condition"])
        return row

    # ------------------------------
    # 메인 엔트리
    # ------------------------------
    def run(self) -> bool:
        try:
            self.train()

            dataset = str(getattr(self, "dataset_tag", self.cfg.get("dataset", self.cfg.get("dataset_tag","Unknown"))))
            horizon = int(self.cfg.get("horizon", self.cfg.get("pred_len", 96)))
            seed    = int(self.cfg.get("seed", 42))
            mode    = str(self.cfg.get("mode", "dynamic"))
            model_tag = str(self.cfg.get("model_tag", getattr(self.model, "model_tag", "HyperConv")))
            perturb  = str(self.cfg.get("perturbation", "tod_shuffle"))  # 예: 'tod_shuffle' | 'peak_weakening' 등

            # 1) baseline
            base_direct, base_hooks = self._evaluate_condition(condition="baseline",  perturb_type=None)
            base_row = self._save_condition(dataset, horizon, seed, mode, model_tag, perturb, "baseline", base_direct, base_hooks)

            # 2) perturbed
            pert_direct, pert_hooks = self._evaluate_condition(condition="perturbed", perturb_type=perturb)
            pert_row = self._save_condition(dataset, horizon, seed, mode, model_tag, perturb, "perturbed", pert_direct, pert_hooks)

            # 3) Δ지표 계산 → perturbed 행 갱신(merge)
            tod_vec = build_test_tod_vector(self.cfg)
            pair = compute_w3_pair_metrics(base_direct, pert_direct, base_hooks, pert_hooks, tod_vec)
            if isinstance(pair, dict):
                # results_W3.csv 의 perturbed 행을 Δ로 보강
                res_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
                row = {
                    **pert_row,
                    **{k: float(v) if v is not None else np.nan for k,v in pair.items()}
                }
                self._append_or_update_results(
                    res_csv, row,
                    subset=["experiment_type","dataset","horizon","seed","mode","model_tag","perturb_type","condition"]
                )

                # Perturb ΔBar 저장
                pdir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar"
                pdir.mkdir(parents=True, exist_ok=True)
                ctx_bar = {
                    "experiment_type": EXP_TAG, "dataset": dataset, "plot_type": "perturb_delta_bar",
                    "horizon": horizon, "seed": seed, "mode": mode, "model_tag": model_tag,
                    "run_tag": f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}_PAIR_{perturb}",
                    "perturb_type": perturb, "condition": "pair"
                }
                save_w3_perturb_delta_bar(ctx_bar, pair, out_dir=pdir)

            # 4) Forest summary 재작성
            fdir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
            rebuild_w3_forest_summary(
                detail_csv=fdir/"forest_plot_detail.csv",
                summary_csv=fdir/"forest_plot_summary.csv",
                per_cond="perturbed", last_cond="baseline",
                group_keys=("experiment_type","dataset","horizon","perturb_type")
            )
            return True
        finally:
            self.cleanup()
