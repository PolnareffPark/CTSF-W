# W5 실험 수정 사항 상세 문서

## 📋 개요

이 문서는 W5 실험(게이트 고정 시험)에 대한 평가 피드백을 반영한 수정 사항을 상세히 기록합니다.

**수정 날짜**: 2025-11-07 (1차), 2025-11-08 (2차)  
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

## ✅ 수정 내용 (1차 - 2025-11-07)

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

## ✅ 수정 내용 (3차 - 2025-11-08)

### 7. GateFixedModel 원본 모델 보호 개선

**문제 상황**:
피드백에서 지적한 바와 같이, 기존 GateFixedModel이 forward 훅에서 `module.alpha.data = mean_val`로 원본 모델의 파라미터를 직접 덮어쓰는 문제가 있었습니다:
- 훅이 등록되면 이후 모든 forward에서 계속 α를 고정
- 고정 평가 후에도 훅이 제거되지 않아 원본 모델이 영구적으로 변형
- 실험 종료 후 해당 모델을 다시 사용하면 동적 게이트로 동작하지 않음

**해결 방안**:
Context Manager 패턴으로 GateFixedModel을 재구성하여 안전한 리소스 관리 구현

**수정된 코드**:

```python
class GateFixedModel:
    """
    게이트를 평균값으로 고정하는 래퍼 (Context Manager)
    
    사용 방법:
        with GateFixedModel(model) as fixed_model:
            # 고정 게이트로 평가
            results = evaluate(fixed_model, ...)
        # with 블록을 벗어나면 자동으로 원본 모델 복원
    """
    def __init__(self, model):
        self.model = model
        self.gate_means = {}
        self.original_alphas = {}  # 원본 alpha 값 백업
        self.hooks = []
        self._active = False
        self._compute_gate_means()
    
    def _compute_gate_means(self):
        """게이트 평균값 계산 및 원본 백업"""
        for i, blk in enumerate(self.model.xhconv_blks):
            self.gate_means[i] = torch.relu(blk.alpha).detach().clone()
            # 원본 alpha 값 백업!
            self.original_alphas[i] = blk.alpha.data.clone()
    
    def __enter__(self):
        """Context manager 진입: 훅 등록"""
        self._register_hooks()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료: 훅 제거 및 원본 복원"""
        self._remove_hooks()
        self._restore_alphas()  # 원본 복원!
        return False
    
    def _remove_hooks(self):
        """등록된 훅 제거"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self._active = False
    
    def _restore_alphas(self):
        """원본 alpha 값 복원"""
        for i, blk in enumerate(self.model.xhconv_blks):
            if i in self.original_alphas:
                blk.alpha.data = self.original_alphas[i].clone()
```

**사용 방법 변경**:

```python
# 이전 (문제 있음)
fixed_model = GateFixedModel(self.model)
fixed_model.eval()
fixed_results = evaluate_with_direct_evidence(fixed_model, ...)
# 원본 모델이 변형됨!

# 수정 후 (안전)
with GateFixedModel(self.model) as fixed_model:
    fixed_model.eval()
    fixed_results = evaluate_with_direct_evidence(fixed_model, ...)
# with 블록을 벗어나면 원본 모델 자동 복원!
```

**변경 효과**:
- ✅ 원본 모델의 alpha가 절대 변형되지 않음
- ✅ 훅이 자동으로 제거됨 (메모리 누수 방지)
- ✅ 예외 발생 시에도 안전하게 복원됨 (finally와 동일한 효과)
- ✅ 코드 의도가 명확해짐 (context manager = 임시 상태)
- ✅ Python의 best practice 준수

### 8. 테스트 코드 강화

테스트에 원본 모델 보호 검증 추가:

```python
def test_gate_fixed_model():
    # 원본 alpha 저장
    original_alphas = [blk.alpha.data.clone() for blk in model.xhconv_blks]
    
    with fixed_model_wrapper as fixed_model:
        # 고정 모델 평가
        out_fixed = fixed_model(x)
        # 훅이 등록되어 있음
        assert len(fixed_model.hooks) > 0
    
    # Context 종료 후 검증
    assert len(fixed_model_wrapper.hooks) == 0  # 훅 제거 확인
    
    # 원본 alpha 복원 확인
    for i, blk in enumerate(model.xhconv_blks):
        assert torch.allclose(blk.alpha.data, original_alphas[i])
    
    # 원본 모델이 정상 작동하는지 확인
    out_dynamic_after = model(x)
    assert torch.allclose(out_dynamic_before, out_dynamic_after)
```

---

## ✅ 수정 내용 (2차 - 2025-11-08)

### 4. 게이트 출력 수집 활성화

**배경**:
피드백에서 지적한 바와 같이, `w5_gate_event_alignment_loss` 지표가 게이트 변동성을 활용하지 못하고 fallback으로 이벤트 게인 차이만 사용하는 문제가 있었습니다.

**수정된 코드**:

```python
# 1. 동적 게이트 모드 평가
# 게이트 출력 수집을 활성화하여 게이트 변동성 지표 계산
self.model.eval()
dynamic_results = evaluate_with_direct_evidence(
    self.model, self.test_loader, self.mu, self.std,
    tod_vec=tod_vec, device=self.device,
    collect_gate_outputs=True  # 추가!
)

# 2. 게이트 고정 모드 평가
# 게이트 출력 수집을 활성화하여 고정 시 변동성이 0에 가까운지 확인
fixed_model = GateFixedModel(self.model)
fixed_model.eval()
fixed_results = evaluate_with_direct_evidence(
    fixed_model, self.test_loader, self.mu, self.std,
    tod_vec=tod_vec, device=self.device,
    collect_gate_outputs=True  # 추가!
)
```

**변경 효과**:
- ✅ `w2_gate_variability_time`, `w2_gate_variability_sample` 등 게이트 변동성 지표가 계산됨
- ✅ `w5_gate_event_alignment_loss`가 정확한 계산식 사용
  - 이전: `event_dynamic - event_fixed` (fallback)
  - 이후: `(event_dynamic * gate_var_dynamic) - (event_fixed * gate_var_fixed)`
- ✅ 고정 게이트의 변동성이 실제로 0에 가까운지 검증 가능

### 5. run_suite.py 중복 실행 문제 해결

**문제 상황**:
피드백에서 지적한 바와 같이, W5 실험이 `modes=["dynamic", "fixed"]`로 두 번 실행되어 불필요한 중복이 발생했습니다. W5Experiment.evaluate_test()가 이미 한 번의 실행으로 동적/고정을 모두 평가하므로 두 번 실행할 필요가 없습니다.

**수정된 코드**:

```python
# run_suite.py
elif experiment_type == "W5":
    # W5는 한 번 실행으로 동적/고정 비교를 모두 수행함
    modes = ["dynamic"]
```

**변경 효과**:
- ✅ 한 번의 실행으로 동적/고정 비교 완료
- ✅ 실행 시간 절반으로 단축
- ✅ CSV에 중복 행 생성 방지
- ✅ 사용자 혼동 감소

### 6. CSV 컬럼 확장

**문제 상황**:
1차 수정에서 `rmse_fixed`, `mae_fixed` 등을 결과에 추가했지만, CSV 컬럼 정의에 없어서 무시되는 문제가 있었습니다.

**수정된 코드**:

```python
# utils/csv_logger.py
"W5": [
    "w5_performance_degradation_ratio", "w5_sensitivity_gain_loss",
    "w5_event_gain_loss", "w5_gate_event_alignment_loss",
    # 고정 모델 개별 지표 (분석 용이성)
    "rmse_fixed", "mae_fixed", "gc_kernel_tod_dcor_fixed", "cg_event_gain_fixed",
    # W2 게이트 변동성 지표 (동적 모델)
    "w2_gate_variability_time", "w2_gate_variability_sample", "w2_gate_entropy",
    "w2_gate_tod_alignment", "w2_gate_gru_state_alignment",
    "w2_event_conditional_response",
    "w2_channel_selectivity_kurtosis", "w2_channel_selectivity_sparsity",
    # 보고용 그림 지표
    "gate_var_t", "gate_var_b", "gate_entropy",
    "gate_q10", "gate_q50", "gate_q90", "gate_hist10",
],
```

**변경 효과**:
- ✅ 고정 모델의 개별 성능 지표가 CSV에 기록됨
- ✅ 게이트 변동성 지표가 CSV에 기록됨
- ✅ 동적 vs 고정 비교 분석이 용이해짐

---

## 📊 결과 구조 변화

### CSV 출력 컬럼

**동적 모델 성능 (기본 컬럼)**:
- `rmse`, `mae`, `mape`, `mse` 등

**고정 모델 성능 (1차 수정에서 추가, 2차에서 CSV 컬럼 정의)**:
- `rmse_fixed`: 고정 게이트 모델의 RMSE
- `mae_fixed`: 고정 게이트 모델의 MAE
- `gc_kernel_tod_dcor_fixed`: 고정 게이트 모델의 TOD 민감도
- `cg_event_gain_fixed`: 고정 게이트 모델의 이벤트 게인

**W5 비교 지표 (1차 수정에서 추가)**:
- `w5_performance_degradation_ratio`: 성능 저하율
- `w5_sensitivity_gain_loss`: TOD 민감도 손실
- `w5_event_gain_loss`: 이벤트 탐지 손실
- `w5_gate_event_alignment_loss`: 게이트-이벤트 정렬 손실

**게이트 변동성 지표 (2차 수정에서 추가)**:
- `w2_gate_variability_time`: 시간 차원 게이트 변동성
- `w2_gate_variability_sample`: 샘플 차원 게이트 변동성
- `w2_gate_entropy`: 게이트 엔트로피
- `w2_gate_tod_alignment`: 게이트-TOD 정렬
- `w2_gate_gru_state_alignment`: 게이트-GRU 상태 정렬
- `w2_event_conditional_response`: 이벤트 조건부 반응
- `w2_channel_selectivity_kurtosis`: 채널 선택도 첨도
- `w2_channel_selectivity_sparsity`: 채널 선택도 희소성

### 예시 결과 (2차 수정 후)

```
# 기본 성능 지표
rmse: 1.234
rmse_fixed: 1.456
w5_performance_degradation_ratio: 0.180  # (1.456-1.234)/1.234 = 18% 성능 저하

# TOD 민감도
gc_kernel_tod_dcor: 0.723
gc_kernel_tod_dcor_fixed: 0.512
w5_sensitivity_gain_loss: 0.211  # 동적이 TOD 패턴을 더 잘 포착

# 이벤트 탐지
cg_event_gain: 0.634
cg_event_gain_fixed: 0.421
w5_event_gain_loss: 0.213  # 동적이 이벤트를 더 잘 탐지

# 게이트 변동성 (2차 수정에서 추가)
w2_gate_variability_time: 0.245  # 동적 게이트 시간 변동성
w2_gate_entropy: 1.823  # 동적 게이트 엔트로피

# 게이트-이벤트 정렬
w5_gate_event_alignment_loss: 0.155  # 정확한 계산식 사용
# = (0.634 * 0.245) - (0.421 * 0.001)  # 고정 게이트 변동성 ≈ 0
```

**해석**:
- 게이트를 고정하면 RMSE가 18% 증가 (성능 저하)
- 동적 게이트는 시간대 패턴을 21.1% 더 잘 포착
- 동적 게이트는 이벤트를 21.3% 더 잘 탐지
- 동적 게이트는 시간에 따라 변동하며 (variability=0.245), 이벤트에 반응함
- 고정 게이트는 변동성이 거의 0에 가까워 이벤트에 반응하지 못함
- **결론**: 동적 게이트가 모델 성능에 중요한 기여를 하며, 특히 이벤트 반응에 핵심적

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

**이전 (문제 있는 방식 - 1차 수정 전)**:
```bash
# 동적 평가
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96 --gate_fixed false

# 고정 평가
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96 --gate_fixed true

# 결과를 수동으로 비교해야 함
```

**1차 수정 후 (동적/고정을 한 번에 평가하지만 중복 실행)**:
```bash
# 한 번의 실행으로 동적/고정 비교 완료
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96

# 문제: modes=["dynamic", "fixed"]로 두 번 실행되어 중복 발생
# CSV에 동일한 결과가 두 행으로 기록됨
```

**2차 수정 후 (최종 - 권장 방식)**:
```bash
# 한 번의 실행으로 모든 비교 완료 (중복 없음)
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96

# 또는 명시적으로 mode 지정
python run_suite.py --experiments W5 --datasets ETTh2 --horizons 96 --modes dynamic

# CSV에 한 행만 기록되며, 동적/고정/비교/변동성 지표 모두 포함
```

### 결과 확인

```bash
# CSV 파일 확인
cat results/results_W5.csv

# 주요 컬럼 (2차 수정 후):
# - rmse: 동적 게이트 RMSE
# - rmse_fixed: 고정 게이트 RMSE
# - w5_performance_degradation_ratio: 성능 저하율
# - w5_sensitivity_gain_loss: 민감도 손실
# - w5_event_gain_loss: 이벤트 손실
# - w5_gate_event_alignment_loss: 게이트-이벤트 정렬 손실
# - w2_gate_variability_time: 동적 게이트 시간 변동성
# - w2_gate_entropy: 동적 게이트 엔트로피
```

---

## 📝 주요 변경 파일 요약

| 파일 | 변경 내용 | 중요도 | 수정 차수 |
|------|-----------|--------|----------|
| `experiments/w5_experiment.py` | GateFixedModel context manager 개선, 원본 보호 | ★★★★★ | 1차, 2차, 3차 |
| `run_suite.py` | W5 modes를 ["dynamic"]만 사용하도록 수정 | ★★★★☆ | 2차 |
| `utils/csv_logger.py` | W5 CSV 컬럼 확장 (고정 지표, 게이트 변동성) | ★★★★☆ | 2차 |
| `utils/experiment_metrics/w5_metrics.py` | docstring 개선, 지표 해석 추가 | ★★★☆☆ | 1차 |
| `docs/experiment_modifications/test_w5_modifications.py` | 테스트 스크립트 및 원본 복원 검증 | ★★★★☆ | 1차, 3차 |
| `docs/experiment_modifications/W5_MODIFICATIONS_SUMMARY.md` | 상세 문서 (1-3차 수정 내역) | ★★★☆☆ | 1차, 2차, 3차 |
| `CHANGES_SUMMARY.md` | W5 수정 내역 (1-3차) | ★★☆☆☆ | 1차, 2차, 3차 |

---

## 🔍 코드 품질 개선 사항

### 1. 명확한 실험 흐름

```
학습 (동적 게이트)
    ↓
평가 단계:
    ├─ 동적 게이트 평가 → dynamic_results
    ├─ 고정 게이트 평가 (Context Manager) → fixed_results
    │   └─ with GateFixedModel(model):
    │       ├─ 훅 등록
    │       ├─ 평가 수행
    │       └─ 자동 복원 (훅 제거 + alpha 복원)
    ├─ W5 지표 계산 → w5_metrics
    └─ 결과 병합 → final_results
```

### 2. 안전성 향상 (3차 수정)

**원본 모델 보호**:
- Context Manager로 임시 상태 관리
- 예외 발생 시에도 안전하게 복원
- 훅 자동 제거로 메모리 누수 방지
- Python best practice 준수

**Before (문제)**:
```python
fixed_model = GateFixedModel(model)
results = evaluate(fixed_model, ...)
# 원본 모델이 변형됨!
```

**After (안전)**:
```python
with GateFixedModel(model) as fixed_model:
    results = evaluate(fixed_model, ...)
# 자동으로 원본 복원!
```

### 3. 자동화 개선

- 더 이상 두 번의 실행 불필요 (2차 수정)
- CSV에 모든 정보가 자동으로 기록
- 분석 스크립트가 쉽게 데이터를 활용 가능

### 4. 유지보수성 향상

- 코드 의도가 명확함 (context manager = 임시 상태)
- 테스트 코드로 회귀 방지 (원본 복원 검증 포함)
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

**작성일**: 2025-11-07 (1차), 2025-11-08 (2차, 3차)  
**작성자**: AI Assistant  
**버전**: 1.3 (최종)

