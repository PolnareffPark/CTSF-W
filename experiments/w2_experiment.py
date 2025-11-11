# ============================================================
# experiments/w2_experiment.py
#   - W2 동적/정적 교차 비교 오케스트레이터 (완성본)
# ============================================================

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from data.dataset import build_test_tod_vector
from models.ctsf_model import HybridTS
from models.experiment_variants import StaticCrossModel
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics
from utils.experiment_metrics.w2_metrics import compute_w2_metrics
from utils.experiment_plotting_metrics.all_plotting_metrics import (
    infer_tod_bins_by_dataset,
    package_w2_gate_metrics,
)
from utils.experiment_plotting_metrics.w2_plotting_metrics import (
    rebuild_w2_forest_summary,
    save_w2_forest_detail_row,
    save_w2_gate_distribution,
    save_w2_gate_tod_heatmap,
)
from .base_experiment import BaseExperiment

RESULTS_ROOT = Path("results")
EXP_TAG = "W2"
MODES = ("dynamic", "static")

_W2_KEYS = [
    "experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag",
]

_CG_GC_KEYS = [
    "mse_std","mae_std",
    "cg_pearson_mean","cg_spearman_mean","cg_dcor_mean",
    "cg_event_gain","cg_event_hit","cg_maxcorr","cg_bestlag",
    "gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2",
    "gc_kernel_feat_dcor","gc_kernel_feat_align",
]

# ---------------------------
# 공통 유틸 (W1과 동일 정책)
# ---------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); return p

def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time()*1000)}")
    df.to_csv(tmp, index=False); tmp.replace(path)

def _append_or_update_rows(csv_path: str | Path, rows: List[Dict[str, Any]], subset_keys: List[str]) -> None:
    if not rows: return
    path = _ensure_dir(csv_path)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        for c in new.columns:
            if c not in old.columns: old[c] = np.nan
        for c in old.columns:
            if c not in new.columns: new[c] = np.nan
        df = pd.concat([old, new], ignore_index=True)
        df.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(df, path)
    else:
        new.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(new, path)

def _amp_range(x) -> float:
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any(): return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))

def _merge_cg_gc_from_direct_evidence(row: dict, direct_evidence: dict | None) -> dict:
    if not isinstance(direct_evidence, dict): return row
    for k in _CG_GC_KEYS:
        if k in direct_evidence:
            try: row[k] = float(direct_evidence[k])
            except Exception: row[k] = direct_evidence[k]
    return row


class W2Experiment(BaseExperiment):
    """
    W2 = 동적 vs 정적 교차 비교 오케스트레이터
    """

    def __init__(self, cfg: Dict[str, Any]):
        cfg_local = copy.deepcopy(cfg or {})
        cfg_local.setdefault("experiment_type", EXP_TAG)
        super().__init__(cfg_local)
        self._last_hooks_data: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}

    # ------------------------------
    # 안전 getter
    # ------------------------------
    @staticmethod
    def _get(obj: Any, keys: List[str], default=None):
        for k in keys:
            if isinstance(obj, dict) and k in obj: return obj[k]
            if hasattr(obj, k): return getattr(obj, k)
        return default

    def _ctx(
        self,
        dataset: str, horizon: int, seed: int, mode: str,
        model_tag: str, run_tag: str, plot_type: str = "",
    ) -> Dict[str, Any]:
        return {
            "experiment_type": EXP_TAG, "dataset": dataset, "plot_type": plot_type,
            "horizon": int(horizon), "seed": int(seed), "mode": str(mode),
            "model_tag": str(model_tag), "run_tag": str(run_tag),
        }

    def _extract_gate_raw(self, hooks: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # 1) hooks에 stage dict로 있을 때
        for key in ["gates_by_stage","gate_by_stage","gate_logs_by_stage"]:
            if key in hooks and isinstance(hooks[key], dict):
                gbs = {}
                for s in ["S","M","D","s","m","d"]:
                    if s in hooks[key]:
                        gbs[s.upper()] = np.asarray(hooks[key][s])
                if gbs:
                    ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                    return gbs, ga

        # 2) last_step_logs에서 보강
        last = getattr(self, "last_step_logs", {})
        if isinstance(last, dict):
            for key in ["gates_by_stage","gate_by_stage"]:
                if key in last and isinstance(last[key], dict):
                    gbs = {}
                    for s in ["S","M","D","s","m","d"]:
                        if s in last[key]: gbs[s.upper()] = np.asarray(last[key][s])
                    if gbs:
                        ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                        return gbs, ga
            cand = {}
            for s in ["S","M","D","s","m","d"]:
                if s in last: cand[s.upper()] = np.asarray(last[s])
            if cand:
                ga = np.concatenate([np.ravel(v) for v in cand.values() if v is not None], axis=0)
                return cand, ga

        # 3) 모델에서 직접 스캔
        gbs: Dict[str, np.ndarray] = {}
        m = getattr(self, "model", None)
        if m is not None:
            layer_arrays: List[np.ndarray] = []
            for attr in ["xhconv_blks","xhc_blks","blocks","modules","layers"]:
                blks = getattr(m, attr, None)
                if blks is None: continue
                try: it = list(blks)
                except Exception:
                    try: it = list(blks.values())
                    except Exception: it = []
                for b in it:
                    arr = None
                    for k in ["cross_gate","gate","alpha","g","w","gate_log","gate_values"]:
                        if hasattr(b, k):
                            v = getattr(b, k)
                            if hasattr(v, "detach"):
                                try: v = v.detach().cpu().numpy()
                                except Exception: pass
                            arr = np.asarray(v, dtype=float); break
                    if arr is not None: layer_arrays.append(arr)
            L = max(1, len(layer_arrays))
            if L > 0:
                bins = [(0, L//3), (L//3, (2*L)//3), ((2*L)//3, L)]
                stage_map = {0:"S",1:"M",2:"D"}
                gdict = {"S":[], "M":[], "D":[]}
                for i, arr in enumerate(layer_arrays):
                    st = stage_map[0 if i < bins[0][1] else (1 if i < bins[1][1] else 2)]
                    gdict[st].append(arr)
                for st in ["S","M","D"]:
                    if gdict[st]:
                        gbs[st] = np.nanmean(np.stack([np.asarray(x) for x in gdict[st]], axis=0), axis=0)
        if gbs:
            ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
            return gbs, ga
        return {}, np.array([], dtype=float)

    # ---------------------------------------------------------
    # 평가 + 훅 수집
    # ---------------------------------------------------------
    def evaluate_test(self):
        """
        - 게이트 출력과 GRU 상태를 모두 후킹하여 hooks_data에 담고,
        compute_all_experiment_metrics(...)로 direct_evidence를 보강한 뒤,
        self._last_hooks_data / self._last_direct에 저장합니다.
        """
        tod_vec = build_test_tod_vector(self.cfg)

        # 1) direct evidence 실행: 게이트 + (가능하면) GRU 상태까지 후킹
        try:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True,
                collect_gru_states=True,   # ★ GRU 상태 후킹
            )
        except TypeError:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True,
            )

        # 2) hooks_data 구성
        hooks_data: Dict[str, Any] = {}
        w2_hooks = direct.pop("hooks_data", None)
        if isinstance(w2_hooks, dict) and len(w2_hooks) > 0:
            hooks_data.update(w2_hooks)

        for k in ("gate_outputs","gru_states","gates_by_stage","gate_by_stage"):
            if k in direct and k not in hooks_data:
                hooks_data[k] = direct[k]

        # 3) 실험 공통 지표 병합(있으면)
        try:
            exp_specific = compute_all_experiment_metrics(
                experiment_type=self.experiment_type,
                model=self.model,
                hooks_data=hooks_data if hooks_data else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=self.cfg.get("perturbation"),
                active_layers=[],
            )
            if isinstance(exp_specific, dict):
                direct.update(exp_specific)
        except ImportError:
            pass

        # 4) 캐싱
        self._plotting_payload = {}
        self._last_hooks_data = copy.deepcopy(hooks_data)
        self._last_direct = copy.deepcopy(direct)
        return direct

    # ---------------------------------------------------------
    # 사후 저장: 그림/표 파일 + results_W2.csv
    # ---------------------------------------------------------
    def _after_eval_save(
        self,
        dataset: str, horizon: int, seed: int, mode: str,
        direct: Dict[str, Any], hooks: Dict[str, Any],
        model_tag: str, run_tag: str,
    ) -> None:
        ctx_base = self._ctx(dataset, horizon, seed, mode, model_tag, run_tag)

        # 1) Forest raw(detail)
        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))
        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        forest_dir.mkdir(parents=True, exist_ok=True)
        save_w2_forest_detail_row(
            {**ctx_base, "plot_type": "forest_plot"},
            rmse_real=rmse_real, mae_real=mae_real,
            detail_csv=forest_dir / "forest_plot_detail.csv"
        )

        # 2) Gate 요약 및 그림
        gates_by_stage, gates_all = self._extract_gate_raw(hooks)
        try:
            tod_vec = build_test_tod_vector(self.cfg)
        except Exception:
            tod_vec = None

        gate_metrics: Dict[str, Any] = {}
        if gates_by_stage or (isinstance(gates_all, np.ndarray) and gates_all.size > 0):
            gate_metrics = package_w2_gate_metrics(
                gates_by_stage=gates_by_stage,
                gates_all=gates_all if isinstance(gates_all, np.ndarray) else None,
                dataset=dataset,
                tod_labels=None,
                tod_vec=tod_vec
            )

        # 그림 저장
        tod_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
        dist_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
        if any(k in gate_metrics for k in ["gate_tod_mean_s","gate_tod_mean_m","gate_tod_mean_d"]):
            tod_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_tod_heatmap({**ctx_base, "plot_type":"gate_tod_heatmap"},
                                    gate_metrics, out_dir=tod_dir, amp_method="range")
        if any(k in gate_metrics for k in ["gate_mean","gate_std","gate_entropy","gate_sparsity","gate_kurtosis","w2_gate_variability_time"]):
            dist_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_distribution({**ctx_base, "plot_type":"gate_distribution"},
                                    gate_metrics, out_dir=dist_dir)

        # 3) 분위수/히스토 폴백
        gate_q10 = gate_q50 = gate_q90 = gate_hist10 = np.nan
        if isinstance(gates_all, np.ndarray) and gates_all.size > 0 and np.isfinite(gates_all).any():
            finite = gates_all[np.isfinite(gates_all)]
            if finite.size >= 5:
                gate_q10, gate_q50, gate_q90 = np.quantile(finite, [0.10,0.50,0.90]).tolist()
                gate_hist10 = float((finite > gate_q90).mean())

        # 4) W2 특화 지표(동적/정적)
        w2m = {}
        try:
            w2m = compute_w2_metrics(self.model, hooks_data=hooks, tod_vec=tod_vec, direct_evidence=direct)
        except Exception:
            w2m = {}

        # static 정의값: TOD/GRU 정렬은 0.0으로 강제
        if str(mode).lower() == "static":
            for k in ("w2_gate_tod_alignment","w2_gate_gru_state_alignment"):
                v = w2m.get(k, np.nan)
                if not (isinstance(v,(int,float)) and np.isfinite(v)):
                    w2m[k] = 0.0

        # 채널 선택도 별칭/폴백
        def _pick_finite(*vals, default=np.nan):
            for v in vals:
                if isinstance(v, (int,float)) and np.isfinite(v):
                    return float(v)
            return float(default)

        w2_kurt = w2m.get("w2_channel_selectivity_kurtosis", np.nan)
        w2_spa  = w2m.get("w2_channel_selectivity_sparsity", np.nan)
        g_kurt  = gate_metrics.get("gate_kurtosis", np.nan)
        g_spa   = gate_metrics.get("gate_sparsity", np.nan)

        gate_channel_kurt     = _pick_finite(w2_kurt, g_kurt, default=0.0)   # 보고서 호환/폴백
        gate_channel_sparsity = _pick_finite(w2_spa,  g_spa,  default=0.0)

        # 5) 결과행
        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed, "mode": mode,
            "model_tag": model_tag, "experiment_type": EXP_TAG,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": rmse_real, "mae": mae_real,

            # Gate 전역 요약
            "gate_mean": float(gate_metrics.get("gate_mean", np.nan)),
            "gate_std":  float(gate_metrics.get("gate_std",  np.nan)),
            "gate_entropy": float(gate_metrics.get("gate_entropy", np.nan)),
            "gate_sparsity": float(gate_metrics.get("gate_sparsity", np.nan)),
            "gate_q10": gate_q10, "gate_q50": gate_q50, "gate_q90": gate_q90,
            "gate_hist10": gate_hist10,

            # TOD 요약
            "gate_tod_mean_s": float(np.nanmean(gate_metrics.get("gate_tod_mean_s", [])) if isinstance(gate_metrics.get("gate_tod_mean_s"), (list,tuple,np.ndarray)) else np.nan),
            "gate_tod_mean_m": float(np.nanmean(gate_metrics.get("gate_tod_mean_m", [])) if isinstance(gate_metrics.get("gate_tod_mean_m"), (list,tuple,np.ndarray)) else np.nan),
            "gate_tod_mean_d": float(np.nanmean(gate_metrics.get("gate_tod_mean_d", [])) if isinstance(gate_metrics.get("gate_tod_mean_d"), (list,tuple,np.ndarray)) else np.nan),
            "tod_amp_mean": float(np.nanmean([np.nanmax(v)-np.nanmin(v) for v in [
                np.asarray(gate_metrics.get("gate_tod_mean_s", []), float),
                np.asarray(gate_metrics.get("gate_tod_mean_m", []), float),
                np.asarray(gate_metrics.get("gate_tod_mean_d", []), float)
            ] if v.size>0 and np.isfinite(v).any()])) if any(isinstance(gate_metrics.get(k),(list,tuple,np.ndarray)) for k in ("gate_tod_mean_s","gate_tod_mean_m","gate_tod_mean_d")) else np.nan,

            # W2 전용 (그대로 주입)
            **w2m,

            # 별칭(보고서 호환)
            "gate_var_t": w2m.get("w2_gate_variability_time", gate_metrics.get("w2_gate_variability_time", np.nan)),
            "gate_var_b": w2m.get("w2_gate_variability_sample", np.nan),
            "gate_channel_kurt": gate_channel_kurt,
            "gate_channel_sparsity": gate_channel_sparsity,
        }

        # cg_*/gc_* 병합
        for k in ("mse_std","mae_std","cg_pearson_mean","cg_spearman_mean","cg_dcor_mean",
                "cg_event_gain","cg_event_hit","cg_maxcorr","cg_bestlag",
                "gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2",
                "gc_kernel_feat_dcor","gc_kernel_feat_align"):
            if k in direct:
                try: row[k] = float(direct[k])
                except Exception: row[k] = direct[k]

        # 저장
        results_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        self._append_or_update_results(
            results_csv, row,
            subset=["dataset","horizon","seed","mode","model_tag","experiment_type"],
        )

    def _append_or_update_results(self, csv_path: Path, row: Dict[str, Any], subset: List[str]) -> None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_new = pd.DataFrame([row])
        if csv_path.exists():
            df_old = pd.read_csv(csv_path)
            for c in df_new.columns:
                if c not in df_old.columns: df_old[c] = np.nan
            for c in df_old.columns:
                if c not in df_new.columns: df_new[c] = np.nan
            df = pd.concat([df_old, df_new], ignore_index=True)
            df.drop_duplicates(subset=subset, keep="last", inplace=True)
            tmp = csv_path.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False); tmp.replace(csv_path)
        else:
            df_new.to_csv(csv_path, index=False)

    # ------------------------------
    # 메인 엔트리
    # ------------------------------
    def _create_model(self):
        mode = str(self.cfg.get("mode", "dynamic")).lower()
        if mode == "static":
            model = StaticCrossModel(self.cfg, self.n_vars)
        else:
            model = HybridTS(self.cfg, self.n_vars)
            if hasattr(model, "set_cross_directions"):
                try: model.set_cross_directions(use_gc=True, use_cg=True)
                except Exception: pass
        return model

    def run(self) -> bool:
        try:
            self.train()
            direct = self.evaluate_test()
            hooks = copy.deepcopy(getattr(self, "_last_hooks_data", {}))
            self.save_results(direct)

            dataset = str(self._get(self.cfg, ["dataset","dataset_tag","data_name"], self.dataset_tag))
            horizon = int(self._get(self.cfg, ["horizon","pred_len"], self.cfg.get("horizon", 96)))
            seed    = int(self._get(self.cfg, ["seed"], self.cfg.get("seed", 42)))
            mode    = str(self.cfg.get("mode", "dynamic"))
            model_tag = str(self.cfg.get("model_tag", getattr(self.model, "model_tag", "HyperConv")))
            run_tag   = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}"

            self._after_eval_save(dataset=dataset, horizon=horizon, seed=seed, mode=mode,
                                  direct=direct, hooks=hooks, model_tag=model_tag, run_tag=run_tag)

            forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
            tod_dir    = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
            dist_dir   = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
            rebuild_w2_forest_summary(
                detail_csv=forest_dir / "forest_plot_detail.csv",
                summary_csv=forest_dir / "forest_plot_summary.csv",
                gate_tod_summary_csv=tod_dir / "gate_tod_heatmap_summary.csv",
                gate_dist_summary_csv=dist_dir / "gate_distribution_summary.csv",
                per_mode="dynamic", last_mode="static",
                group_keys=("experiment_type","dataset","horizon"),
            )
            return True
        finally:
            self.cleanup()
