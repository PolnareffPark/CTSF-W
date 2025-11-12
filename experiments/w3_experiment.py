# ============================================================
# experiments/w3_experiment.py
#   - W3: 데이터 구조 원인 확인 (none vs tod_shift/smooth)
#   - 결과 저장: results/results_W3.csv + per-dataset 그림(summary/detail)
# ============================================================

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from data.dataset import build_test_tod_vector
from models.ctsf_model import HybridTS
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics

from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    save_w3_forest_detail_row,
    rebuild_w3_forest_summary,
    rebuild_w3_perturb_delta_summary,
    save_w3_gate_tod_heatmap,
    save_w3_gate_distribution,
)

from utils.experiment_plotting_metrics.all_plotting_metrics import (
    infer_tod_bins_by_dataset, package_w2_gate_metrics
)
from utils.experiment_plotting_metrics.w2_plotting_metrics import (
    save_w2_gate_tod_heatmap, save_w2_gate_distribution
)
from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    save_w3_perturb_delta_detail
)

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"
MODES = ("none","tod_shift","smooth")  # README/CTSF-W3-plan 기준

_W3_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag"]

_CG_GC_KEYS = [
    "mse_std","mae_std",
    "cg_pearson_mean","cg_spearman_mean","cg_dcor_mean",
    "cg_event_gain","cg_event_hit","cg_maxcorr","cg_bestlag",
    "gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2",
    "gc_kernel_feat_dcor","gc_kernel_feat_align"
]

class W3Experiment:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = copy.deepcopy(cfg or {})
        self.cfg.setdefault("experiment_type", EXP_TAG)
        self.experiment_type = EXP_TAG
        # BaseExperiment와 동일 인터페이스 필드
        from experiments.base_experiment import BaseExperiment
        # BaseExperiment를 구성원으로 활용(코드 공용) — W1/W2에 영향 없음
        self.base = BaseExperiment(self.cfg)
        self.device = self.base.device
        self.n_vars = self.base.n_vars
        self.dataset_tag = getattr(self.base, "dataset_tag", str(self.cfg.get("dataset")))
        self.model = self._create_model()
        self.train = self.base.train
        self.cleanup = self.base.cleanup
        self.test_loader = self.base.test_loader
        self.mu = self.base.mu
        self.std = self.base.std

        self._last_hooks_data: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}

    # ------------------------------
    # 모델 생성 (BaseExperiment의 NotImplementedError 회피)
    # ------------------------------
    def _create_model(self):
        """
        W3은 '데이터 교란' 실험이므로 모델은 고정(CTSF/HybridTS).
        - 교란 모드: cfg['mode'] ∈ {'none','tod_shift','smooth'}
        - 모델 구조는 동일, 데이터/특징 쪽 훅에서 교란이 적용됨.
        """
        model = HybridTS(self.cfg, self.n_vars)
        # (선택) 양방향 교차를 명시적으로 켜두고 시작 — W2와 동일 패턴
        if hasattr(model, "set_cross_directions"):
            try:
                model.set_cross_directions(use_gc=True, use_cg=True)
            except Exception:
                pass
        return model

    # ------------------------------
    # 안전 getter
    # ------------------------------
    @staticmethod
    def _get(obj: Any, keys: List[str], default=None):
        for k in keys:
            if isinstance(obj, dict) and k in obj: return obj[k]
            if hasattr(obj, k): return getattr(obj, k)
        return default

    def _ctx(self, dataset, horizon, seed, mode, model_tag, run_tag, plot_type=""):
        return {
            "experiment_type": EXP_TAG,
            "dataset": str(dataset),
            "plot_type": plot_type,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": str(mode),
            "model_tag": str(model_tag),
            "run_tag": str(run_tag),
        }

    # ------------------------------
    # 게이트 원시 텐서 추출 (W2 패턴 준용)
    # ------------------------------
    def _extract_gate_raw(self, hooks: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # 1) hooks에 stage dict가 실려오는 경우
        for key in ["gates_by_stage","gate_by_stage","gate_logs_by_stage"]:
            if key in hooks and isinstance(hooks[key], dict):
                gbs = {}
                for s in ["S","M","D","s","m","d"]:
                    if s in hooks[key]:
                        gbs[s.upper()] = np.asarray(hooks[key][s])
                if gbs:
                    ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                    return gbs, ga

        # 2) hooks 루트 단위에 S/M/D가 흩어져 실리는 경우
        cand = {}
        for s in ["S","M","D","s","m","d"]:
            if s in hooks:
                cand[s.upper()] = np.asarray(hooks[s])
        if cand:
            ga = np.concatenate([np.ravel(v) for v in cand.values() if v is not None], axis=0)
            return cand, ga

        # 3) 모델 내부 스캔(레이어별 게이트 파라미터 평균)
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
                    if arr is not None:
                        layer_arrays.append(arr)
            L = max(1, len(layer_arrays))
            if L>0:
                bins = [(0, L//3), (L//3, (2*L)//3), ((2*L)//3, L)]
                stage_map = {0:"S",1:"M",2:"D"}
                gdict = {"S":[], "M":[], "D":[]}
                for i, arr in enumerate(layer_arrays):
                    st = stage_map[0 if i<bins[0][1] else (1 if i<bins[1][1] else 2)]
                    gdict[st].append(arr)
                for st in ["S","M","D"]:
                    if gdict[st]:
                        gbs[st] = np.nanmean(np.stack([np.asarray(x) for x in gdict[st]], axis=0), axis=0)
            if gbs:
                ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                return gbs, ga

        return {}, np.array([], dtype=float)

    # ------------------------------
    # 평가 + direct evidence + 훅 수집
    # ------------------------------
    def evaluate_test(self):
        """
        - 게이트 출력/GRU 상태를 후킹하여 hooks_data에 싣고,
        - compute_all_experiment_metrics(...)로 (필요 시) 공통/보조 지표를 병합.
        - W3에서는 per-run(=baseline 또는 perturbed) 개별 실행의 raw만 저장하고,
        Δ 집계는 plotting 단계(rebuild)에서 수행.
        """
        tod_vec = build_test_tod_vector(self.cfg)

        # 게이트/GRU 상태 모두 시도 (구버전 호환 포함)
        try:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True, collect_gru_states=True
            )
        except TypeError:
            direct = evaluate_with_direct_evidence(
                self.model, self.test_loader, self.mu, self.std,
                tod_vec=tod_vec, device=self.device,
                collect_gate_outputs=True
            )

        # hooks_data 구성
        hooks_data = {}
        w_hooks = direct.pop("hooks_data", None)
        if isinstance(w_hooks, dict) and len(w_hooks) > 0:
            hooks_data.update(w_hooks)
        # 루트에 흘러온 훅 산출물 보강
        for k in ("gate_outputs", "gru_states", "gates_by_stage", "gate_by_stage"):
            if k in direct and k not in hooks_data:
                hooks_data[k] = direct[k]

        # 실험 공통 지표(필요 시) — W3는 per-run에서는 빈 dict를 돌려도 무방
        try:
            exp_specific = compute_all_experiment_metrics(
                experiment_type=self.experiment_type,
                model=self.model,
                hooks_data=hooks_data if hooks_data else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=self.cfg.get("mode") or self.cfg.get("perturbation"),
                active_layers=[],
            )
            if isinstance(exp_specific, dict):
                for k, v in exp_specific.items():
                    direct[k] = v
        except ImportError:
            pass

        self._plotting_payload = {}
        self._last_hooks_data = copy.deepcopy(hooks_data)
        self._last_direct = copy.deepcopy(direct)
        return direct
    # ------------------------------
    # run 후 저장 (+ 그림 재작성)
    # ------------------------------
    def _merge_cg_gc_from_direct_evidence(row: dict, direct_evidence: dict | None) -> dict:
        if not isinstance(direct_evidence, dict):
            return row
        for k in _CG_GC_KEYS:
            if k in direct_evidence:
                try: row[k] = float(direct_evidence[k])
                except Exception: row[k] = direct_evidence[k]
        return row

    def _amp_range(x) -> float:
        arr = np.asarray(x, dtype=float)
        if arr.size == 0 or not np.isfinite(arr).any():
            return float("nan")
        return float(np.nanmax(arr) - np.nanmin(arr))

    def _perturbation_tag(mode: str) -> str:
        m = (mode or "").lower()
        if m == "tod_shift": return "TOD"
        if m == "smooth":    return "PEAK"
        return "NONE"

    def _ctx_w3(self, dataset, horizon, seed, mode, model_tag, run_tag, plot_type=""):
        return {
            "experiment_type": EXP_TAG,
            "dataset": str(dataset),
            "plot_type": plot_type,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": str(mode),
            "perturbation": _perturbation_tag(mode),
            "model_tag": str(model_tag),
            "run_tag": str(run_tag),
        }

    def _after_eval_save(
        self,
        dataset: str, horizon: int, seed: int, mode: str,
        direct: dict, hooks: dict,
        model_tag: str, run_tag: str,
    ) -> None:
        ctx_base = _ctx_w3(self, dataset, horizon, seed, mode, model_tag, run_tag)

        # ============ 1) Forest raw(detail) ============
        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))
        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        forest_dir.mkdir(parents=True, exist_ok=True)
        save_w3_forest_detail_row(
            {**ctx_base, "plot_type": "forest_plot"},
            rmse_real=rmse_real, mae_real=mae_real,
            detail_csv=forest_dir / "forest_plot_detail.csv"
        )

        # ============ 2) Gate raw → 요약/그림 ============
        # (W2의 게이트 패키저 재사용)
        gates_by_stage = {}
        gates_all = np.array([], dtype=float)

        # hooks에서 stage별/전역 수집
        for key in ["gates_by_stage","gate_by_stage","gate_logs_by_stage"]:
            if key in hooks and isinstance(hooks[key], dict):
                gbs = {}
                for s in ["S","M","D","s","m","d"]:
                    if s in hooks[key]:
                        gbs[s.upper()] = np.asarray(hooks[key][s])
                if gbs:
                    gates_by_stage = gbs
                    gates_all = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                    break
        if gates_all.size == 0 and "gate_outputs" in hooks:
            try:
                bags = [np.asarray(a, float) for a in hooks["gate_outputs"] if a is not None]
                if bags:
                    gates_all = np.concatenate([b.reshape(-1) for b in bags], axis=0)
            except Exception:
                pass

        tod_bins = infer_tod_bins_by_dataset(dataset)
        T = max(tod_bins, 1)
        tod_labels = list(range(T))
        gate_metrics = {}
        if gates_by_stage or gates_all.size > 0:
            gate_metrics = package_w2_gate_metrics(
                gates_by_stage=gates_by_stage,
                gates_all=gates_all if gates_all.size > 0 else np.zeros((T,), dtype=float),
                dataset=dataset,
                tod_labels=tod_labels,
            )

        # 그림 저장 (W2 유틸 재사용; 경로만 W3로)
        if any(k in gate_metrics for k in ["gate_tod_mean_s","gate_tod_mean_m","gate_tod_mean_d"]):
            tod_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
            tod_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_tod_heatmap({**ctx_base, "plot_type":"gate_tod_heatmap"}, gate_metrics, out_dir=tod_dir, amp_method="range")

        if any(k in gate_metrics for k in ["gate_mean","gate_std","gate_entropy","gate_sparsity","gate_kurtosis","w2_gate_variability_time"]):
            dist_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
            dist_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_distribution({**ctx_base, "plot_type":"gate_distribution"}, gate_metrics, out_dir=dist_dir)

        # ============ 3) results_W3.csv 행 구성 ============
        # 게이트 요약 스칼라
        gate_mean    = float(gate_metrics.get("gate_mean", np.nan))
        gate_std     = float(gate_metrics.get("gate_std", np.nan))
        gate_entropy = float(gate_metrics.get("gate_entropy", np.nan))
        gate_sparsity = float(gate_metrics.get("gate_sparsity", np.nan)) if "gate_sparsity" in gate_metrics else np.nan

        def _safe_mean_key(k):
            v = gate_metrics.get(k, None)
            try:    return float(np.nanmean(np.asarray(v, dtype=float))) if v is not None else np.nan
            except: return np.nan
        gate_tod_mean_s = _safe_mean_key("gate_tod_mean_s")
        gate_tod_mean_m = _safe_mean_key("gate_tod_mean_m")
        gate_tod_mean_d = _safe_mean_key("gate_tod_mean_d")

        # TOD 민감도 평균(range 기준)
        tod_amp_mean = np.nan
        amps = []
        for k in ["gate_tod_mean_s","gate_tod_mean_m","gate_tod_mean_d"]:
            if k in gate_metrics and gate_metrics[k] is not None:
                amps.append(_amp_range(gate_metrics[k]))
        if len([a for a in amps if np.isfinite(a)])>0:
            tod_amp_mean = float(np.nanmean([a for a in amps if np.isfinite(a)]))

        # 분위수/상위 히스토그램
        gate_q10 = gate_q50 = gate_q90 = gate_hist10 = np.nan
        if gates_all.size > 0 and np.isfinite(gates_all).any():
            finite = gates_all[np.isfinite(gates_all)]
            if finite.size >= 5:
                gate_q10, gate_q50, gate_q90 = np.quantile(finite, [0.10,0.50,0.90]).tolist()
                gate_hist10 = float((finite > gate_q90).mean())

        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed, "mode": mode,
            "perturbation": _perturbation_tag(mode),
            "model_tag": model_tag, "experiment_type": EXP_TAG,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": rmse_real, "mae": mae_real,

            # Gate 전역 요약
            "gate_mean": gate_mean, "gate_std": gate_std,
            "gate_entropy": gate_entropy, "gate_sparsity": gate_sparsity,
            "gate_q10": gate_q10, "gate_q50": gate_q50, "gate_q90": gate_q90,
            "gate_hist10": gate_hist10,

            # TOD 요약
            "gate_tod_mean_s": gate_tod_mean_s,
            "gate_tod_mean_m": gate_tod_mean_m,
            "gate_tod_mean_d": gate_tod_mean_d,
            "tod_amp_mean": tod_amp_mean,
        }
        row = _merge_cg_gc_from_direct_evidence(row, direct)

        results_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        self._append_or_update_results(
            results_csv, row,
            subset=["dataset","horizon","seed","mode","model_tag","experiment_type"]
        )

        # ============ 4) 사후 집계(Δ/Bar) ============
        # (1) forest Δ 요약( baseline=none ↔ perturbed={tod_shift|smooth} )
        rebuild_w3_forest_summary(
            detail_csv=forest_dir / "forest_plot_detail.csv",
            summary_csv=forest_dir / "forest_plot_summary.csv",
            group_keys=("experiment_type","dataset","horizon","perturbation")
        )

        # (2) perturb delta bar (원인 지표의 Δ)
        #    - TOD: gc_* 3종
        #    - PEAK: cg_* 3종( + abs(cg_bestlag) )
        pbar_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar"
        pbar_dir.mkdir(parents=True, exist_ok=True)
        save_w3_perturb_delta_detail(
            results_csv=results_csv,
            out_detail_csv=pbar_dir / "perturb_delta_bar_detail.csv",
            group_keys=("experiment_type","dataset","horizon","perturbation")
        )
        rebuild_w3_perturb_delta_summary(
            detail_csv=pbar_dir / "perturb_delta_bar_detail.csv",
            summary_csv=pbar_dir / "perturb_delta_bar_summary.csv",
            group_keys=("experiment_type","dataset","horizon","perturbation")
        )

    def _append_or_update_results(self, csv_path: Path, row: Dict[str, Any], subset: List[str]) -> None:
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
            df.to_csv(tmp, index=False)
            tmp.replace(csv_path)
        else:
            new.to_csv(csv_path, index=False)

    # ------------------------------
    # 외부 호출 진입점
    # ------------------------------
    def run(self) -> bool:
        try:
            # 학습 + 평가
            self.train()
            direct = self.evaluate_test()
            hooks  = copy.deepcopy(self._last_hooks_data)

            # 컨텍스트
            dataset = str(self._get(self.cfg, ["dataset","dataset_tag","data_name"], self.dataset_tag))
            horizon = int(self._get(self.cfg, ["horizon","pred_len"], self.cfg.get("horizon", 96)))
            seed    = int(self._get(self.cfg, ["seed"], self.cfg.get("seed", 42)))
            mode    = str(self.cfg.get("mode","none"))
            model_tag = str(self.cfg.get("model_tag", getattr(self.model,"model_tag","HyperConv")))
            run_tag   = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}"

            # 저장
            self._after_eval_save(
                dataset=dataset, horizon=horizon, seed=seed, mode=mode,
                direct=direct, hooks=hooks, model_tag=model_tag, run_tag=run_tag
            )
            return True
        finally:
            self.cleanup()
