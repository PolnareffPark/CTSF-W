# ============================================================
# experiments/w2_experiment.py  (완전 동작형)
# ============================================================

from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd

from .base_experiment import BaseExperiment
from utils.experiment_plotting_metrics.w2_plotting_metrics import (
    save_w2_gate_tod_heatmap, save_w2_gate_distribution,
    save_w2_forest_detail_row, rebuild_w2_forest_summary
)
from utils.experiment_plotting_metrics.all_plotting_metrics import (
    package_w2_gate_metrics, infer_tod_bins_by_dataset
)

RESULTS_ROOT = Path("results")              # 루트(=W1과 동일)
EXP_TAG = "W2"                              # experiment_type 고정
MODES = ("dynamic", "static")               # 동적/정적 비교

class W2Experiment(BaseExperiment):
    """
    W2 = A3+B1: 동적 vs 정적 교차(+용량 매칭)
    - 모델은 절대 수정하지 않고, 실험 스위치/설정만 변경.
    """

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

    def _ctx(self, dataset: str, horizon: int, seed: int, mode: str, model_tag: str, run_tag: str, plot_type: str="") -> Dict[str,Any]:
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

    # ------------------------------
    # 모드 스위칭 (가능한 모든 스위치 대응)
    # ------------------------------
    def _set_w2_mode(self, mode: str):
        m = getattr(self, "model", None)
        if m is None: return
        # 메서드 우선
        if hasattr(m, "set_gate_mode"):
            try: m.set_gate_mode(mode)
            except Exception: pass
        if hasattr(m, "set_dynamic"):
            try: m.set_dynamic(mode.lower()=="dynamic")
            except Exception: pass
        # 속성 기반
        for attr in ["w2_mode", "gate_mode", "is_dynamic", "use_dynamic_gate"]:
            if hasattr(m, attr):
                try:
                    val = (mode.lower()=="dynamic") if isinstance(getattr(m,attr), (bool,np.bool_)) else mode
                    setattr(m, attr, val)
                except Exception: pass

    # ------------------------------
    # 학습/검증 러너 (후보 메서드 자동 탐색)
    # ------------------------------
    def _run_once_and_collect(self, mode: str) -> Tuple[Dict[str,Any], Dict[str,Any]]:
        """
        반환: (direct_metrics, hooks_data)
        direct_metrics는 최소 'mse_real' 또는 'rmse','mae'를 포함.
        hooks_data는 게이트 원시 로그를 포함(가능하면).
        """
        self._set_w2_mode(mode)

        # 후보 러너 목록(레포마다 다를 수 있음)
        candidates = [
            "train_and_validate",  # (direct, hooks)
            "train_validate",      # (direct, hooks)
            "run_one",             # (direct, hooks)
            "evaluate_validation", # direct만
            "evaluate",            # direct만
        ]
        direct, hooks = None, None
        for name in candidates:
            fn = getattr(self, name, None)
            if callable(fn):
                try:
                    out = fn(mode=mode) if name in ("train_and_validate","train_validate","run_one") else fn()
                except TypeError:
                    out = fn()
                if isinstance(out, tuple) and len(out)>=2:
                    direct, hooks = out[0], (out[1] or {})
                elif isinstance(out, dict):
                    direct, hooks = out, {}
                if isinstance(direct, dict):
                    break

        if not isinstance(direct, dict):
            raise RuntimeError("W2: evaluation runner not found. Provide one of train_and_validate/train_validate/run_one/evaluate* that returns metrics dict.")

        # 정규화
        if "rmse" not in direct and "mse_real" in direct:
            try:
                direct["rmse"] = float(np.sqrt(float(direct["mse_real"])))
            except Exception:
                pass
        if "mae" not in direct and "mae_real" in direct:
            direct["mae"] = direct["mae_real"]

        return direct, (hooks or {})

    # ------------------------------
    # 게이트 원시 로그 추출 (hooks / last_step_logs / model)
    # ------------------------------
    def _extract_gate_raw(self, hooks: Dict[str,Any]) -> Tuple[Dict[str,np.ndarray], np.ndarray]:
        """
        반환:
          gates_by_stage = {"S": arr_S, "M": arr_M, "D": arr_D}
          gates_all      = 전체 게이트 시계열(합쳐서) 1D
        """
        # 1) hooks에서 stage별로 이미 제공된 경우
        for key in ["gates_by_stage","gate_by_stage","gate_logs_by_stage"]:
            if key in hooks and isinstance(hooks[key], dict):
                gbs = {}
                for s in ["S","M","D","s","m","d"]:
                    if s in hooks[key]:
                        gbs[s.upper()] = np.asarray(hooks[key][s])
                if gbs:
                    ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                    return gbs, ga

        # 2) last_step_logs 탐색
        last = getattr(self, "last_step_logs", {})
        if isinstance(last, dict):
            for key in ["gates_by_stage","gate_by_stage"]:
                if key in last and isinstance(last[key], dict):
                    gbs = {}
                    for s in ["S","M","D","s","m","d"]:
                        if s in last[key]:
                            gbs[s.upper()] = np.asarray(last[key][s])
                    if gbs:
                        ga = np.concatenate([np.ravel(v) for v in gbs.values() if v is not None], axis=0)
                        return gbs, ga
            # 낱개 배열만 있는 경우
            cand = {}
            for s in ["S","M","D","s","m","d"]:
                if s in last:
                    cand[s.upper()] = np.asarray(last[s])
            if cand:
                ga = np.concatenate([np.ravel(v) for v in cand.values() if v is not None], axis=0)
                return cand, ga

        # 3) 모델 내부 스캔(xhconv_blks 등) — 게이트가 레이어별로 존재할 때
        gbs = {}
        m = getattr(self, "model", None)
        if m is not None:
            layer_arrays: List[np.ndarray] = []
            for attr in ["xhconv_blks","xhc_blks","blocks","modules","layers"]:
                blks = getattr(m, attr, None)
                if blks is None: 
                    continue
                try:
                    it = list(blks)
                except Exception:
                    try: it = list(blks.values())
                    except Exception: it = []
                for b in it:
                    arr = None
                    for k in ["cross_gate","gate","alpha","g","w","gate_log","gate_values"]:
                        if hasattr(b, k):
                            v = getattr(b,k)
                            if hasattr(v, "detach"):
                                try: v = v.detach().cpu().numpy()
                                except Exception: pass
                            arr = np.asarray(v, dtype=float)
                            break
                    if arr is not None:
                        layer_arrays.append(arr)
                # 레이어를 3분할하여 S/M/D로 평균 라우팅
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

        # 실패 시 빈 구조라도 반환
        return {}, np.array([], dtype=float)

    # ------------------------------
    # 모드 1회 실행 후 저장
    # ------------------------------
    def _after_eval_save(self, dataset: str, horizon: int, seed: int, mode: str,
                         direct: Dict[str,Any], hooks: Dict[str,Any],
                         model_tag: str, run_tag: str):
        ctx_base = self._ctx(dataset, horizon, seed, mode, model_tag, run_tag)

        # 1) Forest detail (rmse/mae)
        rmse_real = float(direct.get("rmse") or np.sqrt(float(direct.get("mse_real", np.nan))))
        mae_real  = float(direct.get("mae") or direct.get("mae_real", np.nan))
        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        forest_dir.mkdir(parents=True, exist_ok=True)
        save_w2_forest_detail_row({**ctx_base, "plot_type":"forest_plot"},
                                  rmse_real=rmse_real, mae_real=mae_real,
                                  detail_csv=forest_dir / "forest_plot_detail.csv")

        # 2) 게이트 원시 로그 수집 → 완전 계산 패키징
        gates_by_stage, gates_all = self._extract_gate_raw(hooks)
        tod_bins = infer_tod_bins_by_dataset(dataset)
        # 길이 모름 대비: 최소 tod_bins 길이를 보장
        max_T = max(
            len(gates_by_stage.get("S",[])) if isinstance(gates_by_stage.get("S",[]), (list,np.ndarray)) else 0,
            len(gates_by_stage.get("M",[])) if isinstance(gates_by_stage.get("M",[]), (list,np.ndarray)) else 0,
            len(gates_by_stage.get("D",[])) if isinstance(gates_by_stage.get("D",[]), (list,np.ndarray)) else 0,
            tod_bins
        )
        tod_labels = list(range(max_T))
        gate_metrics = {}
        if gates_by_stage or gates_all.size > 0:
            gate_metrics = package_w2_gate_metrics(
                gates_by_stage=gates_by_stage,
                gates_all=gates_all if gates_all.size>0 else np.zeros((max_T,), dtype=float),
                dataset=dataset,
                tod_labels=tod_labels
            )

        # 3) Gate‑TOD Heatmap
        if any(k in gate_metrics for k in ["gate_tod_mean_s","gate_tod_mean_m","gate_tod_mean_d"]):
            tod_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
            tod_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_tod_heatmap({**ctx_base, "plot_type":"gate_tod_heatmap"},
                                     gate_metrics, out_dir=tod_dir, amp_method="range")

        # 4) Gate Distribution(+VarT)
        if any(k in gate_metrics for k in ["gate_mean","gate_std","gate_entropy","gate_sparsity","gate_kurtosis","w2_gate_variability_time"]):
            dist_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
            dist_dir.mkdir(parents=True, exist_ok=True)
            save_w2_gate_distribution({**ctx_base, "plot_type":"gate_distribution"},
                                      gate_metrics, out_dir=dist_dir)

        # 5) results/results_W2.csv 기록(보고서용 일괄 표)
        results_csv = RESULTS_ROOT / f"results_{EXP_TAG}.csv"
        row = {
            "dataset": dataset, "horizon": horizon, "seed": seed, "mode": mode,
            "model_tag": model_tag, "experiment_type": EXP_TAG,
            "mse_real": float(direct.get("mse_real", np.nan)),
            "rmse": rmse_real, "mae": mae_real,
            "tod_amp_mean": gate_metrics.get("tod_amp_mean", np.nan),
            "gate_mean": gate_metrics.get("gate_mean", np.nan),
            "gate_std": gate_metrics.get("gate_std", np.nan),
            "w2_gate_variability_time": gate_metrics.get("w2_gate_variability_time", np.nan),
        }
        self._append_or_update_results(results_csv, row,
            subset=["dataset","horizon","seed","mode","model_tag","experiment_type"])

    def _append_or_update_results(self, csv_path: Path, row: Dict[str,Any], subset: List[str]):
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
    def run_single(self):
        """
        동적/정적 2모드를 동일 세팅(시드/데이터/호라이즌)으로 수행하고, 
        각 모드 종료 시 그림/CSV 저장. 마지막에 Forest summary 재작성.
        """
        cfg = getattr(self, "cfg", {}) or getattr(self, "config", {}) or {}
        # 안전 추출: dataset/horizon/seed/model_tag
        dataset = str(self._get(cfg, ["dataset","dataset_tag","data_name"], "ETTh1"))
        horizon = int(self._get(cfg, ["horizon","pred_len"], 96))
        seed    = int(self._get(cfg, ["seed"], 42))
        model_tag = str(getattr(self, "model_tag", getattr(self, "tag", getattr(self.model, "model_tag", "HyperConv"))))
        run_tag   = f"{EXP_TAG}_{dataset}_{horizon}_{seed}"

        for mode in MODES:
            direct, hooks = self._run_once_and_collect(mode=mode)
            self._after_eval_save(dataset, horizon, seed, mode, direct, hooks, model_tag, f"{run_tag}_{mode}")

        # Forest summary 재작성
        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        tod_dir    = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
        dist_dir   = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
        rebuild_w2_forest_summary(
            detail_csv = forest_dir / "forest_plot_detail.csv",
            summary_csv= forest_dir / "forest_plot_summary.csv",
            gate_tod_summary_csv = tod_dir  / "gate_tod_heatmap_summary.csv",
            gate_dist_summary_csv= dist_dir / "gate_distribution_summary.csv",
            per_mode="dynamic", last_mode="static",
            group_keys=("experiment_type","dataset","horizon")
        )
        return True
