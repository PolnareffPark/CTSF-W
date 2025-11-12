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
from models.experiment_variants import StaticCrossModel  # 안전 차원: 사용 안 함
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics

from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    save_w3_forest_detail_row,
    rebuild_w3_forest_summary,
    rebuild_w3_perturb_delta_summary,
    save_w3_gate_tod_heatmap,
    save_w3_gate_distribution,
)

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"
MODES = ("none","tod_shift","smooth")  # README/CTSF-W3-plan 기준

_W3_KEYS = ["experiment_type","dataset","plot_type","horizon","seed","mode","model_tag","run_tag"]

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
        # 교란 실험이라도 모델은 W2 동적 경로와 동일(CTSF 본형)
        model = HybridTS(self.cfg, self.n_vars)
        if hasattr(model, "set_cross_directions"):
            try: model.set_cross_directions(use_gc=True, use_cg=True)
            except Exception: pass
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
        tod_vec = build_test_tod_vector(self.cfg)
        # 게이트 출력까지 수집
        direct = evaluate_with_direct_evidence(
            self.model,
            self.test_loader,
            self.mu, self.std,
            tod_vec=tod_vec,
            device=self.device,
            collect_gate_outputs=True,
        )
        hooks_data: Dict[str, Any] = {}
        w_hooks = direct.pop("hooks_data", None)
        if isinstance(w_hooks, dict) and len(w_hooks)>0:
            hooks_data.update(w_hooks)
        # (옵션) 루트 키로 온 훅도 흡수
        for k in ("gate_outputs","gates_by_stage","gate_by_stage"):
            if k in direct and k not in hooks_data:
                hooks_data[k] = direct[k]

        # 실험 특화 지표(필요시; W3는 기본 {})
        try:
            exp_specific = compute_all_experiment_metrics(
                experiment_type=self.experiment_type,
                model=self.model,
                hooks_data=hooks_data if hooks_data else None,
                tod_vec=tod_vec,
                direct_evidence=direct,
                perturbation_type=self.cfg.get("mode"),
            )
            if isinstance(exp_specific, dict):
                for k,v in exp_specific.items():
                    direct[k] = v
        except ImportError:
            pass

        self._last_hooks_data = copy.deepcopy(hooks_data)
        self._last_direct     = copy.deepcopy(direct)
        return direct

    # ------------------------------
    # run 후 저장 (+ 그림 재작성)
    # ------------------------------
    def _after_eval_save(
        self,
        dataset: str, horizon: int, seed: int, mode: str,
        direct: Dict[str, Any], hooks: Dict[str, Any],
        model_tag: str, run_tag: str,
    ) -> None:
        ctx_base = self._ctx(dataset, horizon, seed, mode, model_tag, run_tag)

        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))

        # 1) Forest raw(detail)
        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        forest_dir.mkdir(parents=True, exist_ok=True)
        save_w3_forest_detail_row(
            {**ctx_base, "plot_type": "forest_plot"},
            rmse_real=rmse_real, mae_real=mae_real,
            detail_csv=forest_dir/"forest_plot_detail.csv"
        )

        # 2) Gate Heatmap / Distribution (가능하면)
        gbs, gall = self._extract_gate_raw(hooks)
        if gbs or gall.size>0:
            # heatmap
            tod_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
            tod_dir.mkdir(parents=True, exist_ok=True)
            save_w3_gate_tod_heatmap({**ctx_base, "plot_type":"gate_tod_heatmap"}, gbs, tod_dir, amp_method="range")
            # distribution
            dist_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
            dist_dir.mkdir(parents=True, exist_ok=True)
            save_w3_gate_distribution({**ctx_base, "plot_type":"gate_distribution"}, gall, dist_dir)

        # 3) results_W3.csv(row)
        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed, "mode": mode,
            "model_tag": model_tag, "experiment_type": EXP_TAG,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": rmse_real, "mae": mae_real,
            # 필요한 direct evidence(있으면 저장)
            "cg_pearson_mean":  direct.get("cg_pearson_mean", np.nan),
            "cg_spearman_mean": direct.get("cg_spearman_mean", np.nan),
            "cg_dcor_mean":     direct.get("cg_dcor_mean", np.nan),
            "cg_event_gain":    direct.get("cg_event_gain", np.nan),
            "cg_event_hit":     direct.get("cg_event_hit", np.nan),
            "cg_maxcorr":       direct.get("cg_maxcorr", np.nan),
            "cg_bestlag":       direct.get("cg_bestlag", np.nan),
            "gc_kernel_tod_dcor": direct.get("gc_kernel_tod_dcor", np.nan),
            "gc_feat_tod_dcor":   direct.get("gc_feat_tod_dcor", np.nan),
            "gc_feat_tod_r2":     direct.get("gc_feat_tod_r2", np.nan),
        }
        results_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        self._append_or_update_results(results_csv, row,
            subset=["dataset","horizon","seed","mode","model_tag","experiment_type"])

        # 4) 그림 summary 재작성
        #    - Forest(Δ) 요약(heatmap/distribution summary와 조인)
        forest_summary = forest_dir/"forest_plot_summary.csv"
        tod_sum = (RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap" / "gate_tod_heatmap_summary.csv")
        dist_sum= (RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"  / "gate_distribution_summary.csv")
        rebuild_w3_forest_summary(
            detail_csv=forest_dir/"forest_plot_detail.csv",
            summary_csv=forest_summary,
            gate_tod_summary_csv=tod_sum if tod_sum.exists() else None,
            gate_dist_summary_csv=dist_sum if dist_sum.exists() else None,
            group_keys=("experiment_type","dataset","horizon","perturbation"),
        )

        #    - Perturb Delta Bar (원인 지표 δ)
        pdb_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar"
        pdb_dir.mkdir(parents=True, exist_ok=True)
        rebuild_w3_perturb_delta_summary(
            results_w3_csv=results_csv,
            out_dir=pdb_dir,
            group_keys=("experiment_type","dataset","horizon","perturbation"),
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
