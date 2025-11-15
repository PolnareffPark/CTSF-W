# ============================================================
# experiments/w3_experiment.py
#   - W3: 데이터 구조 원인 검증 (교란 주입 vs 클린 평가)
#   - 학습 단계에서만 교란을 적용하고, 평가 단계는 항상 클린
#   - 결과 스키마는 W1/W2와 동일 (results_W3.csv + summary/detail)
# ============================================================

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_experiment import BaseExperiment
from data.dataset import build_test_tod_vector
from models.ctsf_model import HybridTS
from utils.direct_evidence import evaluate_with_direct_evidence
from utils.experiment_metrics.all_metrics import compute_all_experiment_metrics
from utils.experiment_plotting_metrics.w3_plotting_metrics import (
    compute_and_save_w3_gate_distribution,
    compute_and_save_w3_gate_tod_heatmap,
    rebuild_w3_forest_summary,
    rebuild_w3_perturb_delta_bar,
    save_w3_forest_detail_row,
)

RESULTS_ROOT = Path("results")
EXP_TAG = "W3"
MODES = ("none", "tod_shift", "smooth")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def _infer_tod_bins(dataset: str) -> int:
    """데이터셋 이름을 기반으로 하루 TOD bin 수 추정."""
    ds = (dataset or "").lower()
    if "ettm" in ds:
        return 96  # 15-min
    if "etth" in ds:
        return 24  # 1-hour
    if "weather" in ds:
        return 144  # 10-min
    return 24  # 안전 기본값


def _default_perturb_cfg(dataset: str) -> Dict[str, Any]:
    """데이터셋별 기본 교란 강도 설정."""
    ds = (dataset or "").lower()
    if "ettm2" in ds:
        return {
            "tod_shift": {"max_shift": 16, "seg_perm_k": 4, "per_channel_jitter": 2, "sdiff_beta": 0.15},
            "smooth":    {"kernel": 7,  "alpha": 0.5},  # ETTm2는 smooth 가 보조 역할
        }
    if "etth2" in ds:
        return {
            "tod_shift": {"max_shift": 4, "seg_perm_k": 3, "per_channel_jitter": 1}, # ETTh2 는 tod_shift 가 보조 역할할
            "smooth":    {"kernel": 7,  "alpha": 0.6},  # 피크 완화 강화
        }
    # 기타 데이터셋 (ETTm1, ETTh1, weather 등)
    tod_bins = _infer_tod_bins(dataset)
    return {
        "tod_shift": {"max_shift": max(1, tod_bins // 48), "seg_perm_k": 0, "per_channel_jitter": 0, "sdiff_beta": 0.0},
        "smooth":    {"kernel": 5, "alpha": 0.3},
    }


def _detect_layout(x: torch.Tensor) -> Tuple[int, int]:
    """
    입력 텐서의 시간축/채널축을 추정.
    허용 형태: (B, L, C) 혹은 (B, C, L)
    반환: (time_axis, channel_axis)
    
    일반적으로 시계열 길이(L)가 채널 수(C)보다 크다고 가정.
    L == C인 경우 (B, L, C)로 해석합니다.
    """
    if x.dim() != 3:
        raise ValueError("W3 wrapper expects 3D input (B, L, C) or (B, C, L).")
    _, a1, a2 = x.shape
    if a1 >= a2:
        return 1, 2  # (B, L, C)
    return 2, 1  # (B, C, L)


# def _moving_avg_1d(x: torch.Tensor, k: int, t_axis: int) -> torch.Tensor:
#     """시간축 기준 이동평균을 적용. (현재 미사용)"""
#     if k <= 1:
#         return x
#     if k % 2 == 0:
#         k += 1  # 홀수 강제
#
#     if t_axis == 1:  # (B, L, C)
#         x_nc_t = x.permute(0, 2, 1)  # (B, C, L)
#     else:
#         x_nc_t = x  # (B, C, L)
#
#     b, c, L = x_nc_t.shape
#     weight = torch.ones(1, 1, k, device=x.device, dtype=x.dtype) / float(k)
#     x_flat = x_nc_t.reshape(b * c, 1, L)
#     pad = k // 2
#     x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
#     y = F.conv1d(x_pad, weight).reshape(b, c, L)
#
#     if t_axis == 1:
#         return y.permute(0, 2, 1)
#     return y


def _append_or_update_results(csv_path: Path | str, row: Dict[str, Any], subset: List[str]) -> None:
    """results_W3.csv에 행을 추가하거나 동일 키(subset)로 갱신."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])

    if path.exists():
        old_df = pd.read_csv(path)
        for col in new_df.columns:
            if col not in old_df.columns:
                old_df[col] = np.nan
        for col in old_df.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged.drop_duplicates(subset=subset, keep="last", inplace=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        merged.to_csv(tmp_path, index=False)
        tmp_path.replace(path)
    else:
        new_df.to_csv(path, index=False)


def _merge_perturb_cfg(defaults: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(defaults)
    if overrides:
        for key, value in overrides.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


def _extract_user_perturb_cfg(cfg: Dict[str, Any], dataset: str) -> Optional[Dict[str, Any]]:
    user_cfg = cfg.get("perturb_cfg")
    if not isinstance(user_cfg, dict):
        return None
    ds_lower = (dataset or "").lower()
    for key, value in user_cfg.items():
        if isinstance(key, str) and key.lower() == ds_lower and isinstance(value, dict):
            return value
    keys_lower = {k.lower() for k in user_cfg.keys() if isinstance(k, str)}
    if "tod_shift" in keys_lower or "smooth" in keys_lower:
        return user_cfg
    return None


# ---------------------------------------------------------------------------
# Perturbation Wrapper
# ---------------------------------------------------------------------------
class _W3PerturbWrapper(nn.Module):
    """
    학습 단계에서만 입력 x에 교란을 가하고, 평가 단계에서는 원본 입력을 그대로 사용.
    """

    def __init__(self, base: nn.Module, mode: str, dataset: str, perturb_cfg: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.add_module("base", base)
        self.mode = str(mode or "none")
        self.dataset = str(dataset or "")
        
        # 기본값 가져오기
        defaults = _default_perturb_cfg(dataset)
        
        # 사용자 제공 값이 있으면 복사 후 기본값으로 채우기, 없으면 기본값 사용
        if isinstance(perturb_cfg, dict):
            self.cfg = perturb_cfg.copy()
            for key, value in defaults.items():
                self.cfg.setdefault(key, value)
        else:
            self.cfg = defaults

        self.tod_bins = _infer_tod_bins(self.dataset)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        if (not self.training) or self.mode == "none":
            return self.base(x, *args, **kwargs)

        if self.mode == "tod_shift":
            x = self._apply_tod_shift(x)
        elif self.mode == "smooth":
            x = self._apply_smooth(x)

        return self.base(x, *args, **kwargs)

    def __getattr__(self, name: str):
        """
        래퍼에 없는 속성은 항상 내부 base 모듈로 위임.
        단, 'base' 자체를 요청하면 base 모듈을 직접 반환.
        """
        modules = object.__getattribute__(self, "_modules")
        base = modules.get("base")
        if base is None:
            raise AttributeError(
                f"{self.__class__.__name__} has no registered 'base' submodule"
            )
        # 'base' 속성 자체를 요청하면 base 모듈을 반환
        if name == "base":
            return base
        # 그 외 속성은 base에 위임
        return getattr(base, name)

    def unwrap(self) -> nn.Module:
        """내부 원본 모델을 반환."""
        return self.base  # __getattr__을 통해 안전하게 접근

    @torch.no_grad()
    def _apply_tod_shift(self, x: torch.Tensor) -> torch.Tensor:
        # ---- 강화된 정합 붕괴형 tod_shift ----
        t_axis, c_axis = _detect_layout(x)               # (B,L,C) 또는 (B,C,L)
        L = x.shape[t_axis]
        if L < 8:                                        # 너무 짧으면 skip
            return x

        cfg = self.cfg.get("tod_shift", {}) or {}
        max_shift = int(cfg.get("max_shift", 8))         # ETTm2: 8→12~24 권장
        seg_k = int(cfg.get("seg_perm_k", 4))            # 세그먼트 수(4 권장)
        ch_jitter = int(cfg.get("per_channel_jitter", 2))# 채널별 ±jitter(0~2)
        mix_beta = float(cfg.get("sdiff_beta", 0.0))     # 0.0~0.25 권장

        # 1) 기본 원형 roll(전구간)
        if max_shift > 0:
            shift = torch.randint(-max_shift, max_shift + 1, (1,), device=x.device).item()
            if shift != 0:
                x = torch.roll(x, shifts=shift, dims=t_axis)

        # 2) 구간 퍼뮤테이션(세그먼트 순열)
        if seg_k > 1 and L >= seg_k:
            seg_len = L // seg_k
            remainder = L % seg_k
            
            # 세그먼트 경계 계산
            segments = []
            start = 0
            for i in range(seg_k):
                end = start + seg_len + (1 if i < remainder else 0)
                segments.append((start, end))
                start = end
            
            # 무작위 순열
            order = torch.randperm(seg_k, device=x.device).tolist()
            
            # 세그먼트 재배열
            if t_axis == 1:  # (B, L, C)
                chunks = [x[:, segments[i][0]:segments[i][1], :] for i in order]
                x = torch.cat(chunks, dim=1)
            else:  # (B, C, L)
                chunks = [x[:, :, segments[i][0]:segments[i][1]] for i in order]
                x = torch.cat(chunks, dim=2)

        # 3) 채널별 미세 시프트(동조성 약화)
        if ch_jitter > 0:
            # 각 채널 별로 독립 시프트
            if t_axis == 1:   # (B,L,C) → (B,C,L)
                x_nc_t = x.permute(0, 2, 1)
            else:
                x_nc_t = x  # (B,C,L)
            B, C, LL = x_nc_t.shape
            if C > 1:  # 채널이 1개 이상일 때만 적용
                j = torch.randint(-ch_jitter, ch_jitter + 1, (C,), device=x.device)
                for c in range(C):
                    s = int(j[c].item())
                    if s != 0:
                        x_nc_t[:, c, :] = torch.roll(x_nc_t[:, c, :], shifts=s, dims=-1)
            x = x_nc_t.permute(0, 2, 1) if t_axis == 1 else x_nc_t

        # 4) 약한 seasonal differencing 혼합(선택)
        if mix_beta > 1e-8 and L > self.tod_bins:
            # mix_beta 클램핑 (0.0 ~ 0.3 권장)
            mix_beta = min(mix_beta, 0.3)
            P = self.tod_bins  # ETTm=96 / ETTh=24
            x = (1.0 - mix_beta) * x + mix_beta * (x - torch.roll(x, shifts=P, dims=t_axis))
        return x


    @torch.no_grad()
    def _apply_smooth(self, x: torch.Tensor) -> torch.Tensor:
        # ---- 피크/급변 완화 강화형 smooth ----
        t_axis, _ = _detect_layout(x)
        cfg = self.cfg.get("smooth", {}) or {}
        k = int(cfg.get("kernel", 7))                    # 7~9 권장
        alpha = float(cfg.get("alpha", 0.6))             # 0.5~0.7 권장
        if k <= 1 or alpha <= 0:
            return x
        if k % 2 == 0: k += 1

        # (A) 삼각 커널로 부드럽게
        half = k // 2
        # float32로 커널 생성 후 정규화
        tri = torch.arange(1, half + 2, device=x.device, dtype=torch.float32)
        weight = torch.cat([tri, tri.flip(0)[1:]]).view(1, 1, -1)  # [1,2,3,2,1] 형태
        weight = weight / weight.sum()  # 정규화

        # conv1d 적용을 위해 (B,C,L) 형태로 맞춤
        if t_axis == 1:  # (B,L,C) → (B,C,L)
            x_nc_t = x.permute(0, 2, 1)
        else:
            x_nc_t = x
        B, C, L = x_nc_t.shape
        x_flat = x_nc_t.reshape(B*C, 1, L)
        pad = k // 2
        x_pad = F.pad(x_flat, (pad, pad), mode="reflect")
        y = F.conv1d(x_pad, weight).reshape(B, C, L)

        # (B) 블렌딩
        y = (1.0 - alpha) * x_nc_t + alpha * y
        return y.permute(0, 2, 1) if t_axis == 1 else y

def _build_mask(df, r, subset_keys):
    m = np.ones(len(df), dtype=bool)
    for k in subset_keys:
        if k in df.columns:
            m &= (df[k].astype(str).fillna("") == str(r.get(k, "")))
        else:
            # 열이 없으면 새로 추가될 것이므로 매칭 불가 -> false
            m &= False
    return m
# ---------------------------------------------------------------------------
# W3 Experiment
# ---------------------------------------------------------------------------
class W3Experiment(BaseExperiment):
    """
    BaseExperiment의 학습/평가 루틴을 그대로 사용하되,
    학습 단계에서만 교란을 주고 평가 단계는 항상 클린으로 수행한다.
    """

    def __init__(self, cfg: Dict[str, Any]):
        cfg_local = copy.deepcopy(cfg or {})
        cfg_local.setdefault("experiment_type", EXP_TAG)
        cfg_local.setdefault("mode", "none")
        super().__init__(cfg_local)

        dataset = str(self.cfg.get("dataset") or "")
        mode = str(self.cfg.get("mode") or "none")
        defaults = _default_perturb_cfg(dataset)
        overrides = _extract_user_perturb_cfg(self.cfg, dataset)
        perturb_cfg = _merge_perturb_cfg(defaults, overrides)
        self.model = _W3PerturbWrapper(self.model, mode=mode, dataset=dataset, perturb_cfg=perturb_cfg)

        self._last_hooks_data: Dict[str, Any] = {}
        self._last_direct: Dict[str, Any] = {}
        self._last_tod_vec: Optional[np.ndarray] = None

    def _create_model(self):
        """BaseExperiment 요구사항: 모델 인스턴스 생성."""
        return HybridTS(self.cfg, self.n_vars)

    def _ctx(self, dataset: str, horizon: int, seed: int, mode: str,
             model_tag: str, run_tag: str, plot_type: str = "") -> Dict[str, Any]:
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

    def _compute_train_acf_snr_for_w3(self):
        """
        훈련 세트 기준 ACF@day / SNR_time (baseline vs perturbed) 계산.
        - train_loader에서 윈도우를 수집하여 1D 연속 시퀀스로 연결(채널0 우선).
        - baseline 시퀀스에 동일 길이의 perturbed 시퀀스를 생성(roll/smooth 근사).
        - ACF@day = corr(x[:-d], x[d:]),  d = 24h에 해당하는 lag(step).
        - SNR_time = Var(x) / Var(diff(x)).
        반환: dict (w3_acf_day_base, w3_acf_day_perturb, w3_intervention_effect_acf_day,
                    w3_snr_time_base, w3_snr_time_perturb, w3_intervention_effect_snr_time)
        """
        # 0) 데이터셋 주기(분) 결정
        def _freq_minutes(dataset_tag: str) -> int:
            ds = (dataset_tag or "").lower()
            if "ettm" in ds:   # ETTm1/2 (15min)
                return 15
            if "etth" in ds:   # ETTh1/2 (hourly)
                return 60
            if "weather" in ds:
                return 10
            return 60  # 안전 기본값   

        dataset_tag = str(getattr(self, "dataset_tag", self.cfg.get("dataset", "")))
        fmin = _freq_minutes(dataset_tag)
        day_pts = max(1, (24 * 60) // fmin)

        # 1) 훈련 윈도우 수집 → 연속 시퀀스로 연결(정규화 해제 시도)
        x_base = []
        max_windows = int(self.cfg.get("w3_train_stats_max_windows", 512))
        mu = getattr(self, "mu", None)
        std = getattr(self, "std", None)

        try:
            collected = 0
            for xb, _yb in self.train_loader:
                # xb: (B, C, L)
                if not isinstance(xb, torch.Tensor):
                    # 텐서가 아닐 경우 numpy로 가정
                    _xb = np.asarray(xb)
                else:
                    _xb = xb.detach().cpu().numpy()

                # 역정규화(가급적)
                if mu is not None and std is not None:
                    try:
                        _mu = np.asarray(mu.detach().cpu().numpy())
                        _std = np.asarray(std.detach().cpu().numpy())
                        # (C,1) 또는 (1,C) → (C,1)로 맞춤
                        if _mu.ndim == 1: _mu = _mu.reshape(-1, 1)
                        if _std.ndim == 1: _std = _std.reshape(-1, 1)
                        _xb = _xb * _std + _mu
                    except Exception:
                        pass

                # 배치 내 모든 샘플의 채널0을 시간축으로 연결
                # _xb: (B, C, L) → 채널0 선택 후 (B, L) → (B*L,)
                ch0 = _xb[:, 0, :] if _xb.ndim == 3 else _xb[..., 0]
                x_base.append(ch0.reshape(-1))
                collected += 1
                if collected >= max_windows:
                    break
        except Exception:
            pass

        if not x_base:
            return {}

        x_base = np.concatenate(x_base, axis=0)  # (T,)
        if x_base.size < (2 * day_pts + 4):
            # 길이가 너무 짧으면 지표 무효
            return {}

        # 2) perturbed 시퀀스 생성(간단 근사; train 교란 강도 반영)
        def _smooth_1d(arr: np.ndarray, k: int = 5, alpha: float = 0.5) -> np.ndarray:
            k = max(1, int(k))
            pad = k // 2
            paded = np.pad(arr, (pad, pad), mode="reflect")
            ker = np.ones(k, dtype=np.float64) / float(k)
            mv = np.convolve(paded, ker, mode="valid")
            # blend
            return (1.0 - alpha) * arr + alpha * mv

        def _apply_perturb_1d(arr: np.ndarray, pert: str, kw: dict) -> np.ndarray:
            if not pert or pert == "none":
                return arr.copy()
            if pert == "tod_shift":
                # 15분 데이터(ETTm*)는 ±8, 60분(ETTh*)는 ±2 정도를 기본값으로
                default_shift = 8 if fmin <= 15 else 2
                k = int(kw.get("shift_points", default_shift))
                k = max(-k, min(k, k))
                if k == 0:
                    return arr.copy()
                # 랜덤 부호 부여
                _k = random.randint(-abs(k), abs(k))
                return np.roll(arr, _k)
            if pert == "smooth":
                k = int(kw.get("k", 5 if fmin >= 60 else 7))
                a = float(kw.get("alpha", 0.5))
                return _smooth_1d(arr, k=k, alpha=a)
            # 미정의 모드는 원본 반환
            return arr.copy()

        pert_type = str(self.cfg.get("perturbation", "none"))
        pert_kw = dict(self.cfg.get("perturbation_kwargs", {}))
        x_pert = _apply_perturb_1d(x_base, pert_type, pert_kw)

        # 3) ACF@day / SNR_time 계산
        def _acf_day(x: np.ndarray, d: int) -> float:
            if x.size <= d:
                return np.nan
            xm = x - np.mean(x)
            num = float(np.dot(xm[:-d], xm[d:]))
            den = float(np.dot(xm, xm)) + 1e-12
            return num / den

        def _snr(x: np.ndarray) -> float:
            if x.size < 3:
                return np.nan
            var_sig = float(np.var(x))
            var_noise = float(np.var(np.diff(x)))
            return var_sig / (var_noise + 1e-12)

        acf_b = _acf_day(x_base, day_pts)
        acf_p = _acf_day(x_pert, day_pts)
        snr_b = _snr(x_base)
        snr_p = _snr(x_pert)

        out = {
            "w3_acf_day_base": float(acf_b) if np.isfinite(acf_b) else np.nan,
            "w3_acf_day_perturb": float(acf_p) if np.isfinite(acf_p) else np.nan,
            "w3_intervention_effect_acf_day": (
                float(abs(acf_p - acf_b)) if np.isfinite(acf_b) and np.isfinite(acf_p) else np.nan
            ),
            "w3_snr_time_base": float(snr_b) if np.isfinite(snr_b) else np.nan,
            "w3_snr_time_perturb": float(snr_p) if np.isfinite(snr_p) else np.nan,
            "w3_intervention_effect_snr_time": (
                float(snr_p - snr_b) if np.isfinite(snr_b) and np.isfinite(snr_p) else np.nan
            ),
        }
        return out

    def _append_or_update_results(self, csv_path, row_or_dict, subset_keys):
        """
        results_W3.csv 방어적 병합(선택): csv_logger가 신규 칼럼을 드롭할 수 있는 환경 대비.
        - subset_keys로 동일 키를 찾으면 해당 행만 업데이트, 없으면 append.
        - 기존/신규 열의 유니온을 보장.
        """
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        # row_or_dict 보정: 기본 식별 키 보장
        r = dict(row_or_dict)
        r.setdefault("experiment_type", "W3")
        r.setdefault("dataset", str(getattr(self, "dataset_tag", self.cfg.get("dataset", ""))))
        r.setdefault("horizon", int(self.cfg.get("horizon", self.cfg.get("pred_len", 96))))
        r.setdefault("seed", int(self.cfg.get("seed", 0)))
        r.setdefault("mode", str(self.cfg.get("mode", self.cfg.get("perturbation", "none"))))
        r.setdefault("model_tag", str(self.cfg.get("model_tag", getattr(self, "model_tag", "CTSF"))))

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # 열 유니온
            for k in r.keys():
                if k not in df.columns:
                    df[k] = pd.NA
            # 키 일치 행 찾기
            mask = _build_mask(df, r, subset_keys)
            if mask.any():
                idx = df[mask].index[0]
                for k, v in r.items():
                    df.at[idx, k] = v
            else:
                df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
        else:
            df = pd.DataFrame([r])

        # dtype 간단 최적화(선택)
        if "horizon" in df.columns:
            df["horizon"] = df["horizon"].astype("int64", errors="ignore")
        if "seed" in df.columns:
            df["seed"] = df["seed"].astype("int64", errors="ignore")

        tmp = csv_path.with_suffix(".tmp.csv")
        df.to_csv(tmp, index=False)
        tmp.replace(csv_path)

    def evaluate_test(self):
        """
        W3 평가 루프 (패치 버전)
        - 기존 평가 로직을 유지하면서 다음을 추가한다:
        (A) 훈련 세트 기준 ACF@day / SNR_time (baseline vs. perturbed) 산출 및 저장
        (B) window_errors 상세(분할 CSV) + 요약 CSV 저장, 그리고 Parquet 통합(메모리 안전)
        (C) conv-GRU 정렬(align)·위상(phase) 지표 baseline/perturbed/delta 저장
        (D) (필요시) results_W3.csv에 신규 칼럼 안전 병합(열 유니온, 부분 갱신)
        """
        import numpy as _np
        from pathlib import Path as _Path

        # --- 0) 기존 로직: 베이스라인/교란 평가 실행 -------------------------------
        # NOTE: 아래 2줄은 기존 코드에서 사용하는 평가 호출을 그대로 둔다.
        #       evaluate_with_direct_evidence(...) 호출과 compute_all_experiment_metrics(...) 병합,
        #       그리고 baseline_direct / current_direct 생성은 기존 구현을 따른다.
        baseline_direct, current_direct = self._evaluate_test_with_baseline_and_perturbed()

        # --- 1) H1/H2: 훈련 세트 기준 ACF@day / SNR 주입 --------------------------
        acf_snr = self._compute_train_acf_snr_for_w3()
        # 키 네이밍: w3_acf_day_base / w3_acf_day_perturb / w3_intervention_effect_acf_day
        #           w3_snr_time_base / w3_snr_time_perturb / w3_intervention_effect_snr_time
        if acf_snr:
            current_direct.update(acf_snr)

        # --- 2) H6/H5: 정렬/위상(Align/Phase) 원시값 병행 저장 -------------------
        # (cg_pearson_mean: conv/gru 미분 상관, cg_bestlag: 최적 랙)
        try:
            _align_b = baseline_direct.get("cg_pearson_mean", _np.nan)
            _align_p = current_direct.get("cg_pearson_mean", _np.nan)
            current_direct["convgru_align_base"] = float(_align_b) if _align_b is not None else _np.nan
            current_direct["convgru_align_pert"] = float(_align_p) if _align_p is not None else _np.nan
            current_direct["convgru_align_delta"] = (
                float(_align_p) - float(_align_b)
                if _np.isfinite(_align_b) and _np.isfinite(_align_p) else _np.nan
            )
        except Exception:
            pass

        try:
            _phase_b = baseline_direct.get("cg_bestlag", _np.nan)
            _phase_p = current_direct.get("cg_bestlag", _np.nan)
            current_direct["convgru_phase_base"] = float(_phase_b) if _phase_b is not None else _np.nan
            current_direct["convgru_phase_pert"] = float(_phase_p) if _phase_p is not None else _np.nan
            current_direct["convgru_phase_delta"] = (
                float(_phase_p) - float(_phase_b)
                if _np.isfinite(_phase_b) and _np.isfinite(_phase_p) else _np.nan
            )
        except Exception:
            pass

        # --- 3) window_errors 상세/요약 저장 + 라벨(MSE) 교정 ---------------------
        base_win = baseline_direct.pop("win_errors", None)
        pert_win = current_direct.pop("win_errors", None)
        if base_win is not None and pert_win is not None:
            try:
                # 저장 컨텍스트
                ctx = {
                    "dataset": str(getattr(self, "dataset_tag", self.cfg.get("dataset", ""))),
                    "horizon": int(self.cfg.get("horizon", self.cfg.get("pred_len", 96))),
                    "seed": int(self.cfg.get("seed", 0)),
                    "model_tag": str(self.cfg.get("model_tag", getattr(self, "model_tag", "CTSF"))),
                    "experiment_type": "W3",
                    "mode": str(self.cfg.get("mode", self.cfg.get("perturbation", "none"))),
                    "run_tag": str(getattr(self, "run_tag", f"{self.cfg.get('dataset','')}-h{self.cfg.get('horizon',96)}-s{self.cfg.get('seed',0)}-W3-{self.cfg.get('perturbation','none')}"))
                }
                # 안전 임포트 (패키지/상대 경로 모두 지원)
                try:
                    from utils.experiment_plotting_metrics import w3_plotting_metrics as _w3pm
                except Exception:
                    import utils.experiment_plotting_metrics.w3_plotting_metrics as _w3pm  # 로컬 실행시

                # per-(H,S) 분할 CSV + 요약 CSV + Parquet 통합
                _w3pm.save_w3_window_errors_detail(
                    ctx=ctx,
                    base_errors=_np.asarray(base_win),
                    pert_errors=_np.asarray(pert_win),
                    perturbation=str(self.cfg.get("perturbation", "none")),
                    out_root=str(self.cfg.get("out_root", "results")),
                )
            except Exception as _e:
                print(f"[W3] window_errors 저장 중 경고: {_e}")

        # (선택) 요약 컬럼에서 win_rmse_* 라벨을 win_mse_*로 교정
        for _old, _new in [("win_rmse_mean", "win_mse_mean"),
                        ("win_rmse_std", "win_mse_std")]:
            if _old in current_direct and _new not in current_direct:
                current_direct[_new] = current_direct.pop(_old)

        # --- 4) (옵션) results_W3.csv에 신규 칼럼 병합(열 유니온/키 중복 업데이트) --
        try:
            # csv_logger가 신규 칼럼을 드롭하는 환경을 대비한 방어적 병합
            results_csv = _Path(self.cfg.get("out_root", "results")) / "results_W3.csv"
            self._append_or_update_results(results_csv, current_direct,
                                        subset_keys=("dataset", "horizon", "seed", "mode", "model_tag", "experiment_type"))
        except Exception as _e:
            # csv_logger가 이미 모든 칼럼을 저장한다면 무시
            print(f"[W3] results_W3.csv 보강(선택) 중 경고: {_e}")

        return current_direct

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
        ctx = self._ctx(dataset, horizon, seed, mode, model_tag, run_tag)

        # 디렉터리 준비
        forest_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "forest_plot"
        heatmap_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_tod_heatmap"
        dist_dir   = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "gate_distribution"
        forest_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        dist_dir.mkdir(parents=True, exist_ok=True)

        # 스칼라 성능치 정리 (rmse_real/mae_real)
        mse_real = direct.get("mse_real")
        rmse_direct = direct.get("rmse")
        try:
            rmse_real = float(rmse_direct) if rmse_direct is not None else float(np.sqrt(float(mse_real)))
        except Exception:
            rmse_real = float("nan")

        mae_direct = direct.get("mae", direct.get("mae_real"))
        try:
            mae_real = float(mae_direct) if mae_direct is not None else float("nan")
        except Exception:
            mae_real = float("nan")

        # forest detail (각 run의 실측 RMSE/MAE 기록)
        save_w3_forest_detail_row(
            {**ctx, "plot_type": "forest_plot"},
            rmse_real=rmse_real,
            mae_real=mae_real,
            detail_csv=forest_dir / "forest_plot_detail.csv",
        )

        # Gate‑TOD Heatmap / Gate 분포
        tod_vec = self._last_tod_vec  # 재사용

        if tod_vec is not None:
            compute_and_save_w3_gate_tod_heatmap(
                {**ctx, "plot_type": "gate_tod_heatmap"},
                hooks_data=hooks,
                tod_vec=tod_vec,
                dataset=dataset,
                out_dir=heatmap_dir,
                amp_method="range",
            )

        compute_and_save_w3_gate_distribution(
            {**ctx, "plot_type": "gate_distribution"},
            hooks_data=hooks,
            out_dir=dist_dir,
        )

        # results_W3.csv에 넣을 기본 행
        results_row: Dict[str, Any] = {
            "dataset": dataset,
            "horizon": horizon,
            "seed": seed,
            "mode": mode,
            "model_tag": model_tag,
            "experiment_type": EXP_TAG,
            "mse_real": float(mse_real) if mse_real is not None else np.nan,
            "rmse": rmse_real,
            "mae":  mae_real,
        }

        # --- win_errors: summary만 results에 기록, 원본 분포는 별도 detail로 저장 ---
        we = direct.get("win_errors", None)
        if isinstance(we, (list, tuple, np.ndarray)) and len(we) > 0:
            w = np.asarray(we, float)
            w = w[np.isfinite(w)]
            if w.size > 0:
                results_row["win_rmse_mean"] = float(np.nanmean(w))
                results_row["win_rmse_std"]  = float(np.nanstd(w))
                results_row["win_rmse_p95"]  = float(np.nanpercentile(w, 95))
                results_row["win_rmse_max"]  = float(np.nanmax(w))
                # run 별 원본 분포 저장 (append 모드)
                win_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "window_errors"
                win_dir.mkdir(parents=True, exist_ok=True)
                win_csv = win_dir / "window_errors_detail.csv"
                
                dfw = pd.DataFrame({
                    "dataset": [dataset]*w.size,
                    "horizon": [horizon]*w.size,
                    "seed": [seed]*w.size,
                    "mode": [mode]*w.size,
                    "model_tag": [model_tag]*w.size,
                    "idx": np.arange(w.size),
                    "rmse": w
                })
                
                # append 또는 create
                if win_csv.exists():
                    old = pd.read_csv(win_csv)
                    merged = pd.concat([old, dfw], ignore_index=True)
                    # 같은 설정의 이전 run 제거
                    subset = ["dataset", "horizon", "seed", "mode", "model_tag", "idx"]
                    merged.drop_duplicates(subset=subset, keep="last", inplace=True)
                    merged.to_csv(win_csv, index=False)
                else:
                    dfw.to_csv(win_csv, index=False)
            # CSV에는 raw vector를 넣지 않음
            direct.pop("win_errors", None)

        # direct 나머지 스칼라/문자열 합류
        for key, value in direct.items():
            if key in results_row:
                continue
            try:
                results_row[key] = float(value)
            except Exception:
                results_row[key] = value

        _append_or_update_results(
            RESULTS_ROOT / f"results_{EXP_TAG}.csv",
            results_row,
            subset=["dataset", "horizon", "seed", "mode", "model_tag", "experiment_type"],
        )

        # 요약/Δ 리빌드 (데이터셋 폴더에 자신의 Δ만 쓰게 하려면 filter_dataset 인자 사용)
        perturb_dir = RESULTS_ROOT / f"results_{EXP_TAG}" / dataset / "perturb_delta_bar"
        perturb_dir.mkdir(parents=True, exist_ok=True)

        rebuild_w3_forest_summary(
            detail_csv=forest_dir / "forest_plot_detail.csv",
            summary_csv=forest_dir / "forest_plot_summary.csv",
            gate_tod_summary_csv=heatmap_dir / "gate_tod_heatmap_summary.csv",
            group_keys=("experiment_type", "dataset", "horizon"),
        )

        # w3_plotting_metrics.rebuild_w3_perturb_delta_bar가 filter_dataset 인자를 받도록
        rebuild_w3_perturb_delta_bar(
            results_csv=RESULTS_ROOT / f"results_{EXP_TAG}.csv",
            detail_csv=perturb_dir / "perturb_delta_bar_detail.csv",
            summary_csv=perturb_dir / "perturb_delta_bar_summary.csv",
            filter_dataset=dataset,
        )

    def run(self):
        dataset = str(self.cfg.get("dataset") or getattr(self, "dataset_tag", ""))
        horizon = int(self.cfg.get("horizon", 96))
        seed = int(self.cfg.get("seed", 42))
        mode = str(self.cfg.get("mode", "none"))
        model_tag = getattr(self.model, "model_tag", "HyperConv")
        run_tag = f"{EXP_TAG}_{dataset}_{horizon}_{seed}_{mode}"

        try:
            self.train()
            direct = self.evaluate_test()
            hooks = self._last_hooks_data
            self._after_eval_save(dataset, horizon, seed, mode, direct, hooks, model_tag, run_tag)
        finally:
            self.cleanup()

