# ============================================================
# experiments/w2_experiment.py
#   - W2 동적/정적 교차 비교 오케스트레이터
# ============================================================

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    "experiment_type",
    "dataset",
    "plot_type",
    "horizon",
    "seed",
    "mode",
    "model_tag",
    "run_tag",
]

_CG_GC_KEYS = [
    "mse_std","mae_std",
    "cg_pearson_mean","cg_spearman_mean","cg_dcor_mean",
    "cg_event_gain","cg_event_hit","cg_maxcorr","cg_bestlag",
    "gc_kernel_tod_dcor","gc_feat_tod_dcor","gc_feat_tod_r2",
    "gc_kernel_feat_dcor","gc_kernel_feat_align"
]

def _merge_cg_gc_from_direct_evidence(row: dict, direct_evidence: dict | None) -> dict:
    """direct_evidence dict에 있으면 해당 키들을 row에 주입"""
    if not isinstance(direct_evidence, dict):
        return row
    for k in _CG_GC_KEYS:
        if k in direct_evidence:
            try: row[k] = float(direct_evidence[k])
            except Exception: row[k] = direct_evidence[k]
    return row

def _amp_range(x) -> float:
    """TOD 민감도(range) 계산"""
    import numpy as np
    arr = np.asarray(x, dtype=float)
    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))

# ---------------------------
# 공통 유틸 (W1과 동일 정책)
# ---------------------------
def _ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _atomic_write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = _ensure_dir(path)
    tmp = path.with_suffix(path.suffix + f".tmp{int(time.time() * 1000)}")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _append_or_update_rows(csv_path: str | Path, rows: List[Dict[str, Any]], subset_keys: List[str]) -> None:
    if not rows:
        return
    path = _ensure_dir(csv_path)
    new = pd.DataFrame(rows)
    if path.exists():
        old = pd.read_csv(path)
        # 컬럼 union
        for c in new.columns:
            if c not in old.columns:
                old[c] = np.nan
        for c in old.columns:
            if c not in new.columns:
                new[c] = np.nan
        df = pd.concat([old, new], ignore_index=True)
        df.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(df, path)
    else:
        new.drop_duplicates(subset=subset_keys, keep="last", inplace=True)
        _atomic_write_csv(new, path)


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
            if isinstance(obj, dict) and k in obj:
                return obj[k]
            if hasattr(obj, k):
                return getattr(obj, k)
        return default

    def _ctx(
        self,
        dataset: str,
        horizon: int,
        seed: int,
        mode: str,
        model_tag: str,
        run_tag: str,
        plot_type: str = "",
    ) -> Dict[str, Any]:
        return {
            "experiment_type": EXP_TAG,
            "dataset": dataset,
            "plot_type": plot_type,
            "horizon": int(horizon),
            "seed": int(seed),
            "mode": str(mode),
            "model_tag": str(model_tag),
            "run_tag": str(run_tag),
        }

    def _extract_gate_raw(
        self,
        hooks: Dict[str, Any],
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        # 1) hooks에서 stage별로 이미 제공된 경우
        for key in ["gates_by_stage", "gate_by_stage", "gate_logs_by_stage"]:
            if key in hooks and isinstance(hooks[key], dict):
                gbs = {}
                for s in ["S", "M", "D", "s", "m", "d"]:
                    if s in hooks[key]:
                        gbs[s.upper()] = np.asarray(hooks[key][s])
                if gbs:
                    ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                    return gbs, ga

        # 2) last_step_logs 탐색
        last = getattr(self, "last_step_logs", {})
        if isinstance(last, dict):
            for key in ["gates_by_stage", "gate_by_stage"]:
                if key in last and isinstance(last[key], dict):
                    gbs = {}
                    for s in ["S", "M", "D", "s", "m", "d"]:
                        if s in last[key]:
                            gbs[s.upper()] = np.asarray(last[key][s])
                    if gbs:
                        ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                        return gbs, ga
            cand = {}
            for s in ["S", "M", "D", "s", "m", "d"]:
                if s in last:
                    cand[s.upper()] = np.asarray(last[s])
            if cand:
                ga = np.concatenate([np.ravel(v) for v in cand.values() if v is not None], axis=0)
                return cand, ga

        # 3) 모델 내부 스캔
        gbs: Dict[str, np.ndarray] = {}
        m = getattr(self, "model", None)
        if m is not None:
            layer_arrays: List[np.ndarray] = []
            for attr in ["xhconv_blks", "xhc_blks", "blocks", "modules", "layers"]:
                blks = getattr(m, attr, None)
                if blks is None:
                    continue
                try:
                    it = list(blks)
                except Exception:
                    try:
                        it = list(blks.values())
                    except Exception:
                        it = []
                for b in it:
                    arr = None
                    for k in ["cross_gate", "gate", "alpha", "g", "w", "gate_log", "gate_values"]:
                        if hasattr(b, k):
                            v = getattr(b, k)
                            if hasattr(v, "detach"):
                                try:
                                    v = v.detach().cpu().numpy()
                                except Exception:
                                    pass
                            arr = np.asarray(v, dtype=float)
                            break
                    if arr is not None:
                        layer_arrays.append(arr)
                L = max(1, len(layer_arrays))
                if L > 0:
                    bins = [(0, L // 3), (L // 3, (2 * L) // 3), ((2 * L) // 3, L)]
                    stage_map = {0: "S", 1: "M", 2: "D"}
                    gdict = {"S": [], "M": [], "D": []}
                    for i, arr in enumerate(layer_arrays):
                        st = stage_map[0 if i < bins[0][1] else (1 if i < bins[1][1] else 2)]
                        gdict[st].append(arr)
                    for st in ["S", "M", "D"]:
                        if gdict[st]:
                            gbs[st] = np.nanmean(
                                np.stack([np.asarray(x) for x in gdict[st]], axis=0),
                                axis=0,
                            )
            if gbs:
                ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                return gbs, ga

        return {}, np.array([], dtype=float)

    def _after_eval_save(
        self,
        dataset: str,
        horizon: int,
        seed: int,
        mode: str,
        direct: Dict[str, Any],
        hooks: Dict[str, Any],
        model_tag: str,
        run_tag: str,
    ) -> None:
        ctx_base = self._ctx(dataset, horizon, seed, mode, model_tag, run_tag)

        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real = float(direct.get("mae") or direct.get("mae_real", np.nan))

        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        forest_dir.mkdir(parents=True, exist_ok=True)
        save_w2_forest_detail_row(
            {**ctx_base, "plot_type": "forest_plot"},
            rmse_real=rmse_real,
            mae_real=mae_real,
            detail_csv=forest_dir / "forest_plot_detail.csv",
        )

        gates_by_stage, gates_all = self._extract_gate_raw(hooks)
        tod_bins = infer_tod_bins_by_dataset(dataset)
        max_T = max(
            len(gates_by_stage.get("S", [])) if isinstance(gates_by_stage.get("S", []), (list, np.ndarray)) else 0,
            len(gates_by_stage.get("M", [])) if isinstance(gates_by_stage.get("M", []), (list, np.ndarray)) else 0,
            len(gates_by_stage.get("D", [])) if isinstance(gates_by_stage.get("D", []), (list, np.ndarray)) else 0,
            tod_bins,
        )
        if max_T <= 0:
            max_T = tod_bins
        tod_labels = list(range(max_T))

        gate_metrics: Dict[str, Any] = {}
        if gates_by_stage or gates_all.size > 0:
            gate_metrics = package_w2_gate_metrics(
                gates_by_stage=gates_by_stage,
                gates_all=gates_all if gates_all.size > 0 else np.zeros((max_T,), dtype=float),
                dataset=dataset,
                tod_labels=tod_labels,
            )

        if any(k in gate_metrics for k in ["gate_tod_mean_s", "gate_tod_mean_m", "gate_tod_mean_d"]):
            tod_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
            tod_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_tod_heatmap(
                {**ctx_base, "plot_type": "gate_tod_heatmap"},
                gate_metrics,
                out_dir=tod_dir,
                amp_method="range",
            )

        if any(
            k in gate_metrics
            for k in [
                "gate_mean",
                "gate_std",
                "gate_entropy",
                "gate_sparsity",
                "gate_kurtosis",
                "w2_gate_variability_time",
            ]
        ):
            dist_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
            dist_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_distribution(
                {**ctx_base, "plot_type": "gate_distribution"},
                gate_metrics,
                out_dir=dist_dir,
            )

        detail_rows: List[Dict[str, Any]] = []
        gates_detail = gate_metrics.get("gate_distribution_detail")
        if isinstance(gates_detail, list):
            detail_rows.extend(gates_detail)
        if detail_rows:
            detail_csv = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution" / "gate_distribution_per_channel_detail.csv"
            _append_or_update_rows(
                detail_csv,
                [
                    {
                        **{k: ctx_base[k] for k in ctx_base if k in _W2_KEYS},
                        **row,
                    }
                    for row in detail_rows
                ],
                subset_keys=_W2_KEYS + ["channel"],
            )

        results_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        
        # 1) TOD 민감도/게이트 통계 계산/보강
        #    - gate_metrics에는 gate_mean/std/entropy, gate_tod_mean_s/m/d 등이 있을 수 있음
        gate_mean   = float(gate_metrics.get("gate_mean", np.nan))
        gate_std    = float(gate_metrics.get("gate_std", np.nan))
        gate_entropy= float(gate_metrics.get("gate_entropy", np.nan))
        gate_sparsity = float(gate_metrics.get("gate_sparsity", np.nan)) if "gate_sparsity" in gate_metrics else np.nan

        # TOD 평균(스테이지별 평균값) 및 민감도(range)
        def _safe_mean_key(k):
            v = gate_metrics.get(k, None)
            try:
                return float(np.nanmean(np.asarray(v, dtype=float))) if v is not None else np.nan
            except Exception:
                return np.nan
        gate_tod_mean_s = _safe_mean_key("gate_tod_mean_s")
        gate_tod_mean_m = _safe_mean_key("gate_tod_mean_m")
        gate_tod_mean_d = _safe_mean_key("gate_tod_mean_d")
        tod_amp_mean = np.nan
        amps = []
        for k in ["gate_tod_mean_s","gate_tod_mean_m","gate_tod_mean_d"]:
            if k in gate_metrics and gate_metrics[k] is not None:
                amps.append(_amp_range(gate_metrics[k]))
        if len([a for a in amps if np.isfinite(a)])>0:
            tod_amp_mean = float(np.nanmean([a for a in amps if np.isfinite(a)]))

        # 분위수/상위 히스토그램 지표 (gate_values로 직접 산출; 없으면 NaN)
        gate_q10 = gate_q50 = gate_q90 = gate_hist10 = np.nan
        if gates_all.size > 0 and np.isfinite(gates_all).any():
            finite = gates_all[np.isfinite(gates_all)]
            if finite.size >= 5:
                gate_q10, gate_q50, gate_q90 = np.quantile(finite, [0.10, 0.50, 0.90]).tolist()
                gate_hist10 = float((finite > gate_q90).mean())

        # 2) W2 전용 지표 계산(가능 시 hooks/tod_vec/ direct_evidence와 함께)
        try:
            from data.dataset import build_test_tod_vector  # 상단에 이미 import되어 있음
            tod_vec = build_test_tod_vector(self.cfg)
        except Exception:
            tod_vec = None

        try:
            w2m = compute_w2_metrics(
                self.model,
                hooks_data=hooks,                 # evaluate_test에서 취합한 hooks
                tod_vec=tod_vec,
                direct_evidence=direct            # evaluate_with_direct_evidence + compute_all_experiment_metrics의 결과
            )
        except Exception:
            w2m = {}

        # alias 맵핑(보고서/표 호환)
        gate_var_t = w2m.get("w2_gate_variability_time", gate_metrics.get("w2_gate_variability_time", np.nan))
        gate_var_b = w2m.get("w2_gate_variability_sample", np.nan)
        gate_channel_kurt     = w2m.get("w2_channel_selectivity_kurtosis", np.nan)
        gate_channel_sparsity = w2m.get("w2_channel_selectivity_sparsity", np.nan)

        # 3) row 구성
        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed, "mode": mode,
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

            # W2 전용 지표
            **w2m,

            # 별칭(보고서 호환 컬럼명)
            "gate_var_t": gate_var_t,
            "gate_var_b": gate_var_b,
            "gate_channel_kurt": gate_channel_kurt,
            "gate_channel_sparsity": gate_channel_sparsity,
        }

        # 4) direct_evidence에서 cg_*/gc_* 등 있으면 병합
        row = _merge_cg_gc_from_direct_evidence(row, direct)

        # 5) append+dedupe 저장
        results_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        self._append_or_update_results(
            results_csv,
            row,
            subset=["dataset","horizon","seed","mode","model_tag","experiment_type"],
        )

    def _append_or_update_results(
        self,
        csv_path: Path,
        row: Dict[str, Any],
        subset: List[str],
    ) -> None:
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
    def _create_model(self):
        mode = str(self.cfg.get("mode", "dynamic")).lower()
        if mode == "static":
            model = StaticCrossModel(self.cfg, self.n_vars)
        else:
            model = HybridTS(self.cfg, self.n_vars)
            if hasattr(model, "set_cross_directions"):
                try:
                    model.set_cross_directions(use_gc=True, use_cg=True)
                except Exception:
                    pass
        return model

def evaluate_test(self):
    """
    - 게이트 출력과 GRU 상태를 모두 후킹하여 hooks_data에 담고,
      compute_all_experiment_metrics(...)로 direct_evidence를 보강한 뒤,
      self._last_hooks_data / self._last_direct에 저장합니다.
    - 이로써 동적 모드에서 w2_gate_tod_alignment / w2_gate_gru_state_alignment가
      정상 계산되어 results_W2.csv에 기록됩니다.
    """
    # 1) TOD 벡터 구성 (실험 설정 기반)
    tod_vec = build_test_tod_vector(self.cfg)

    # 2) 직접증거 평가 호출: 게이트/GRU 상태 모두 수집 시도
    try:
        direct = evaluate_with_direct_evidence(
            self.model,
            self.test_loader,
            self.mu,
            self.std,
            tod_vec=tod_vec,
            device=self.device,
            collect_gate_outputs=True,
            collect_gru_states=True,   # ★ 핵심: GRU 상태 후킹
        )
    except TypeError:
        # 구버전 시그니처 호환(collect_gru_states 미지원 시 게이트만)
        direct = evaluate_with_direct_evidence(
            self.model,
            self.test_loader,
            self.mu,
            self.std,
            tod_vec=tod_vec,
            device=self.device,
            collect_gate_outputs=True,
        )

    # 3) hooks_data 구성: 내부 hooks_data + 루트에 실려온 키를 보강 흡수
    hooks_data: Dict[str, Any] = {}
    w2_hooks = direct.pop("hooks_data", None)
    if isinstance(w2_hooks, dict) and len(w2_hooks) > 0:
        hooks_data.update(w2_hooks)

    # 루트에 있을 수도 있는 후킹 산출물을 안전하게 병합
    for k in ("gate_outputs", "gru_states", "gates_by_stage", "gate_by_stage"):
        if k in direct and k not in hooks_data:
            hooks_data[k] = direct[k]

    # 4) 실험 공통 지표 계산(있으면 direct에 병합)
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
            for k, v in exp_specific.items():
                direct[k] = v
    except ImportError:
        # 선택적 모듈이 없는 환경 호환
        pass

    # 5) 최종 캐싱(그림/표 단계에서 사용)
    self._plotting_payload = {}
    self._last_hooks_data = copy.deepcopy(hooks_data)
    self._last_direct = copy.deepcopy(direct)
    return direct

    def run(self) -> bool:
        try:
            self.train()
            direct = self.evaluate_test()
            hooks = copy.deepcopy(getattr(self, "_last_hooks_data", {}))
            self.save_results(direct)

            dataset = str(
                self._get(self.cfg, ["dataset", "dataset_tag", "data_name"], self.dataset_tag)
            )
            horizon = int(self._get(self.cfg, ["horizon", "pred_len"], self.cfg.get("horizon", 96)))
            seed = int(self._get(self.cfg, ["seed"], self.cfg.get("seed", 42)))
            mode = str(self.cfg.get("mode", "dynamic"))
            model_tag = str(self.cfg.get("model_tag", getattr(self.model, "model_tag", "HyperConv")))
            run_tag = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}"

            self._after_eval_save(
                dataset=dataset,
                horizon=horizon,
                seed=seed,
                mode=mode,
                direct=direct,
                hooks=hooks,
                model_tag=model_tag,
                run_tag=run_tag,
            )

            forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
            tod_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
            dist_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
            rebuild_w2_forest_summary(
                detail_csv=forest_dir / "forest_plot_detail.csv",
                summary_csv=forest_dir / "forest_plot_summary.csv",
                gate_tod_summary_csv=tod_dir / "gate_tod_heatmap_summary.csv",
                gate_dist_summary_csv=dist_dir / "gate_distribution_summary.csv",
                per_mode="dynamic",
                last_mode="static",
                group_keys=("experiment_type", "dataset", "horizon"),
            )
            return True
        finally:
            self.cleanup()
