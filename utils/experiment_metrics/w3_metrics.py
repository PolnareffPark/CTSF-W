# ============================================================
# utils/experiment_metrics/w3_metrics.py
#   - W3: 데이터 구조 원인 확인 (baseline vs perturbed) 보조 계산기
#     * 기본적으로 그림 집계(rebuild_*)에서 Δ를 계산하며,
#       본 모듈은 필요시(둘 다 주어질 때) 동일 정의로 Δ를 산출.
# ============================================================

from __future__ import annotations
from typing import Dict, Optional
import numpy as np

def compute_w3_metrics(
    baseline_metrics: Optional[Dict] = None,
    perturb_metrics: Optional[Dict] = None,
    win_errors_base: Optional[np.ndarray] = None,
    win_errors_pert: Optional[np.ndarray] = None,
    dz_ci: bool = False
) -> Dict:
    """
    W3는 '쌍 비교'가 본질이라 per-run 호출 시에는 사용할 일이 거의 없음.
    이 함수는 (집계기가 쌍을 확보했을 때) Δ 계산을 수행하도록 설계할 수 있으나,
    현재 파이프라인에서는 plotting 단계(rebuild_*)에서 Δ를 전담하므로
    여기서는 빈 dict를 반환해도 무방합니다.
    """
    return {}