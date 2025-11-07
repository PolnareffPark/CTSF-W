# W5 실험 수정 사항 상세 문서

## 📋 개요

이 문서는 W5 실험(게이트 고정 시험)에 대한 평가 피드백을 반영한 수정 사항을 상세히 기록합니다.

**수정 날짜**: 2025-11-07  
**관련 실험**: W5 (게이트 고정 효과 검증)

---

## 🎯 W5 실험 목적

W5 실험은 **동적 게이트의 효과를 검증**하기 위한 실험입니다:
- 학습된 모델의 게이트를 평균값으로 고정했을 때 성능이 얼마나 저하되는가?
- 게이트 고정 시 시간대(TOD) 민감도와 이벤트 탐지 능력은 어떻게 변하는가?
- 동적 게이트가 이벤트 발생 시점에 반응하여 변동하는 정도는 얼마나 되는가?

---

## ❌ 기존 코드의 문제점

### 1. 평가 분리 문제

**문제 상황**:
```python
# 기존 코드 (w5_experiment.py)
def evaluate_test(self):
    if self.cfg.get("gate_fixed", False):
        # 고정 모드만 평가
        fixed_model = GateFixedModel(self.model)
        return evaluate_with_direct_evidence(fixed_model, ...)
    else:
        # 동적 모드만 평가
        return super().evaluate_test()
```

**문제점**:
- `gate_fixed=False`로 실행 → 동적 게이트만 평가
- `gate_fixed=True`로 실행 → 고정 게이트만 평가
- 한 번의 실행에서 두 결과를 모두 얻을 수 없음
- W5 비교 지표(`w5_performance_degradation_ratio` 등)가 계산되지 않음

### 2. 지표 계산 불가

**문제 상황**:
```python
# w5_metrics.py
def compute_w5_metrics(model, fixed_model_metrics=None, dynamic_model_metrics=None):
    if dynamic_model_metrics is None or fixed_model_metrics is None:
        # 항상 이 분기에 걸림!
        return {
            "w5_performance_degradation_ratio": np.nan,
            "w5_sensitivity_gain_loss": np.nan,
            ...
        }
```

**결과**:
- CSV 파일에 모든 W5 지표가 `NaN`으로 기록됨
- 동적 vs 고정 비교가 불가능
- 실험 목적 달성 불가

### 3. 실행 복잡성

- 두 번의 별도 실행 필요 (gate_fixed=False, True)
- 결과를 수동으로 비교해야 함
- 자동화 및 재현성 저하

---

## ✅ 수정 내용

### 1. W5Experiment.evaluate_test() 완전 재구성

**수정된 코드**:

```python
def evaluate_test(self):
    """동적 게이트와 고정 게이트를 모두 평가하여 비교"""
    from utils.direct_evidence import evaluate_with_direct_evidence
    from data.dataset import build_test_tod_vector
    from utils.experiment_metrics.w5_metrics import compute_w5_metrics
    
    tod_vec = build_test_tod_vector(self.cfg)
    
    # 1. 동적 게이트 모드 평가 (원래 모델 그대로)
    self.model.eval()
    dynamic_results = evaluate_with_direct_evidence(
        self.model, self.test_loader, self.mu, self.std,
        tod_vec=tod_vec, device=self.device
    )
    
    # 2. 게이트 고정 모드 평가
    fixed_model = GateFixedModel(self.model)
    fixed_model.eval()
    fixed_results = evaluate_with_direct_evidence(
        fixed_model, self.test_loader, self.mu, self.std,
        tod_vec=tod_vec, device=self.device
    )
    
    # 3. W5 특화 비교 지표 계산
    w5_metrics = compute_w5_metrics(
        self.model,
        fixed_model_metrics=fixed_results,
        dynamic_model_metrics=dynamic_results
    )
    
    # 4. 결과 병합
    final_results = {**dynamic_results}
    final_results.update(w5_metrics)
    
    # 고정 모델의 주요 지표를 별도 키로 추가
    final_results['rmse_fixed'] = fixed_results.get('rmse', np.nan)
    final_results['mae_fixed'] = fixed_results.get('mae', np.nan)
    final_results['gc_kernel_tod_dcor_fixed'] = fixed_results.get('gc_kernel_tod_dcor', np.nan)
    final_results['cg_event_gain_fixed'] = fixed_results.get('cg_event_gain', np.nan)
    
    return final_results
```

**변경 효과**:
- ✅ 한 번의 실행으로 동적 + 고정 평가 완료
- ✅ W5 비교 지표가 정상적으로 계산됨
- ✅ 동적/고정 개별 성능도 모두 CSV에 기록
- ✅ `gate_fixed` 플래그 불필요

### 2. run_tag 단순화

**이전**:
```python
def _get_run_tag(self):
    gate_fixed = "fixed" if self.cfg.get("gate_fixed", False) else "dynamic"
    return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W5-{gate_fixed}"
```

**수정 후**:
```python
def _get_run_tag(self):
    return f"{self.dataset_tag}-h{self.cfg['horizon']}-s{self.cfg['seed']}-W5"
```

### 3. W5 지표 docstring 개선

**수정된 docstring**:
```python
def compute_w5_metrics(...) -> Dict:
    """
    W5 실험 특화 지표 계산: 동적 게이트 vs 고정 게이트 비교
    
    Returns:
        dict with keys:
        - w5_performance_degradation_ratio: 성능 저하율
          (rmse_fixed - rmse_dynamic) / rmse_dynamic
          양수면 고정 시 성능 악화, 음수면 오히려 개선
        - w5_sensitivity_gain_loss: 민감도 이득 손실
          tod_dynamic - tod_fixed
          양수면 동적 게이트가 시간대 패턴을 더 잘 포착
        - w5_event_gain_loss: 이벤트 이득 손실
          event_gain_dynamic - event_gain_fixed
          양수면 동적 게이트가 이벤트를 더 잘 탐지
        - w5_gate_event_alignment_loss: 게이트-이벤트 정렬 손실
          동적 게이트는 이벤트 발생 시 크게 변동하지만,
          고정 게이트는 변화 없음. 그 차이를 정량화
    """
```

---

## 📊 결과 구조 변화

### CSV 출력 컬럼

**동적 모델 성능 (기본 컬럼)**:
- `rmse`, `mae`, `mape`, `mse` 등

**고정 모델 성능 (새로 추가)**:
- `rmse_fixed`: 고정 게이트 모델의 RMSE
- `mae_fixed`: 고정 게이트 모델의 MAE
- `gc_kernel_tod_dcor_fixed`: 고정 게이트 모델의 TOD 민감도
- `cg_event_gain_fixed`: 고정 게이트 모델의 이벤트 게인

**W5 비교 지표 (새로 추가)**:
- `w5_performance_degradation_ratio`: 성능 저하율
- `w5_sensitivity_gain_loss`: TOD 민감도 손실
- `w5_event_gain_loss`: 이벤트 탐지 손실
- `w5_gate_event_alignment_loss`: 게이트-이벤트 정렬 손실

### 예시 결과

```
rmse: 1.234
rmse_fixed: 1.456
w5_performance_degradation_ratio: 0.180  # (1.456-1.234)/1.234 = 18% 성능 저하

gc_kernel_tod_dcor: 0.723
gc_kernel_tod_dcor_fixed: 0.512
w5_sensitivity_gain_loss: 0.211  # 동적이 TOD 패턴을 더 잘 포착

cg_event_gain: 0.634
cg_event_gain_fixed: 0.421
w5_event_gain_loss: 0.213  # 동적이 이벤트를 더 잘 탐지
```

**해석**:
- 게이트를 고정하면 RMSE가 18% 증가 (성능 저하)
- 동적 게이트는 시간대 패턴을 21.1% 더 잘 포착
- 동적 게이트는 이벤트를 21.3% 더 잘 탐지
- **결론**: 동적 게이트가 모델 성능에 중요한 기여를 함

---

## 🧪 테스트

### 테스트 파일

**위치**: `docs/experiment_modifications/test_w5_modifications.py`

### 테스트 항목

#### 1. `test_gate_fixed_model()`
- GateFixedModel이 게이트를 올바르게 평균값으로 고정하는지 검증
- ReLU 적용으로 음수 게이트가 없는지 확인
- forward hook이 모든 블록에 등록되는지 확인
- 동적 vs 고정 출력이 다른지 확인 (게이트 효과 검증)

#### 2. `test_w5_metrics_computation()`
- W5 지표 계산이 정확한지 검증
- 성능 저하율 계산식 확인
- 민감도/이벤트 손실 계산식 확인
- 지표 해석이 올바른지 확인

#### 3. `test_w5_metrics_with_missing_data()`
- 데이터 누락 시 graceful handling 확인
- None 입력 시 NaN 반환 확인
- 일부 지표만 있을 때 계산 가능한 것만 계산하는지 확인

#### 4. `test_w5_evaluate_test_integration()`
- evaluate_test 전체 로직 시뮬레이션
- 동적 평가 → 고정 평가 → 지표 계산 → 병합 순서 확인
- 최종 결과에 모든 필수 키가 포함되는지 확인

### 테스트 실행

```bash
# 환경 설정 후
cd /home/himchan/proj/CTSF/CTSF-W
python docs/experiment_modifications/test_w5_modifications.py
```

**예상 출력**:
```
╔══════════════════════════════════════════════════════════════════════════════╗
║                    W5 실험 수정 사항 테스트                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

================================================================================
테스트 1: GateFixedModel 게이트 고정 검증
================================================================================
...
  ✓ 테스트 1 통과

================================================================================
테스트 2: W5 지표 계산 확인
================================================================================
...
  ✓ 테스트 2 통과

================================================================================
모든 테스트 완료!
================================================================================
```

---

## 🚀 실험 실행 가이드

### 실행 방법

**이전 (문제 있는 방식)**:
```bash
# 동적 평가
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96 --gate_fixed false

# 고정 평가
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96 --gate_fixed true

# 결과를 수동으로 비교해야 함
```

**현재 (개선된 방식)**:
```bash
# 한 번의 실행으로 모든 비교 완료
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96

# CSV에 동적/고정/비교 지표가 모두 기록됨
```

### 결과 확인

```bash
# CSV 파일 확인
cat results/results_W5/W5_results.csv

# 주요 컬럼:
# - rmse: 동적 게이트 RMSE
# - rmse_fixed: 고정 게이트 RMSE
# - w5_performance_degradation_ratio: 성능 저하율
# - w5_sensitivity_gain_loss: 민감도 손실
# - w5_event_gain_loss: 이벤트 손실
```

---

## 📝 주요 변경 파일 요약

| 파일 | 변경 내용 | 중요도 |
|------|-----------|--------|
| `experiments/w5_experiment.py` | evaluate_test() 완전 재구성, run_tag 단순화 | ★★★★★ |
| `utils/experiment_metrics/w5_metrics.py` | docstring 개선, 지표 해석 추가 | ★★★☆☆ |
| `docs/experiment_modifications/test_w5_modifications.py` | 테스트 스크립트 신규 작성 | ★★★★☆ |
| `docs/experiment_modifications/W5_MODIFICATIONS_SUMMARY.md` | 상세 문서 신규 작성 | ★★★☆☆ |
| `CHANGES_SUMMARY.md` | W5 수정 내역 추가 | ★★☆☆☆ |

---

## 🔍 코드 품질 개선 사항

### 1. 명확한 실험 흐름

```
학습 (동적 게이트)
    ↓
평가 단계:
    ├─ 동적 게이트 평가 → dynamic_results
    ├─ 고정 게이트 평가 → fixed_results
    ├─ W5 지표 계산 → w5_metrics
    └─ 결과 병합 → final_results
```

### 2. 자동화 개선

- 더 이상 두 번의 실행 불필요
- CSV에 모든 정보가 자동으로 기록
- 분석 스크립트가 쉽게 데이터를 활용 가능

### 3. 유지보수성 향상

- 코드 의도가 명확함
- 테스트 코드로 회귀 방지
- 문서화로 이해도 향상

---

## ⚠️ 주의사항 및 향후 개선 방향

### 1. 게이트 고정 방식

**현재 방식**:
```python
# alpha 파라미터의 ReLU 평균 사용
self.gate_means[i] = torch.relu(blk.alpha).detach().clone()
```

**개선 가능성**:
- 학습 중 실제 게이트 출력의 EMA(Exponential Moving Average) 수집
- 더 정확한 평균값 사용
- 현재 방식도 충분히 의미 있는 비교 가능

### 2. 게이트-이벤트 정렬 지표

**현재 방식**:
```python
# 게이트 변동성과 이벤트 게인의 곱
alignment = event_gain * gate_variability
```

**개선 가능성**:
- 이벤트 발생 시점의 게이트 변화율 직접 계산
- 시계열 상관 분석
- 현재 방식도 합리적인 근사

### 3. 통계적 유의성

- 여러 시드로 실행하여 평균/표준편차 계산 권장
- 성능 차이가 통계적으로 유의한지 검증

---

## 📚 참고 자료

### 관련 코드

- `experiments/w5_experiment.py`: W5 실험 클래스
- `utils/experiment_metrics/w5_metrics.py`: W5 지표 계산
- `experiments/base_experiment.py`: 베이스 실험 클래스
- `utils/direct_evidence.py`: 직접 증거 평가

### 관련 문서

- `CHANGES_SUMMARY.md`: 전체 변경 내역
- `docs/experiment_modifications/test_w5_modifications.py`: 테스트 코드
- `hp2_config.yaml`: 실험 설정

---

## 📧 문의

수정 사항에 대한 문의나 추가 개선 제안이 있으면 알려주세요.

---

**작성일**: 2025-11-07  
**작성자**: AI Assistant  
**버전**: 1.0

