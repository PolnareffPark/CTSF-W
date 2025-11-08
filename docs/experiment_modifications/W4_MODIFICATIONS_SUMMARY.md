# W4 실험 수정 사항 요약

## 최신 업데이트 (2025-11-08)

### 🎯 층 분할 로직 개선 (피드백 반영)

#### 문제점
기존 로직에서 `depth // 3`를 사용하여 **일부 층이 어떤 모드(shallow/mid/deep)에도 속하지 않는 문제** 발견:
- 예: depth=8일 때 shallow=[0,1], mid=[2,3], deep=[6,7] → **4,5번 층이 누락됨**

#### 해결책
**mid 모드를 전체 중간 영역으로 확장**하여 모든 층이 정확히 한 범주에 속하도록 개선:

```python
# 개선 전
elif cross_layers == "mid":
    n = max(1, depth // 3)
    start = depth // 3
    active_layers = list(range(start, start + n))

# 개선 후
elif cross_layers == "mid":
    shallow_end = max(1, depth // 3)
    deep_start = depth - max(1, depth // 3)
    active_layers = list(range(shallow_end, deep_start))
```

#### 검증 결과
다양한 depth에서 **모든 층이 정확히 한 범주에 속함**을 확인:
- depth=6: shallow=[0,1], mid=[2,3], deep=[4,5] ✓
- depth=7: shallow=[0,1], mid=[2,3,4], deep=[5,6] ✓
- **depth=8: shallow=[0,1], mid=[2,3,4,5], deep=[6,7] ✓ (4,5 누락 문제 해결!)**
- depth=9: shallow=[0,1,2], mid=[3,4,5], deep=[6,7,8] ✓
- depth=10: shallow=[0,1,2], mid=[3,4,5,6], deep=[7,8,9] ✓

---

## 초기 수정 (2025-11-07)

### 수정 배경
W4 실험 코드에 대한 평가에서 다음의 문제점이 지적되었습니다:

1. **하드코딩된 depth 값**: `base_experiment.py`에서 `cnn_depth=7` 기본값을 사용하여 모델과 불일치 가능
2. **로직 중복**: W4Experiment와 BaseExperiment 양쪽에서 active_layers를 계산
3. **중간층 표현 미수집**: `w4_layerwise_representation_similarity` 지표가 NaN이 될 수 있음

## 수정 내용

### 1. `experiments/w4_experiment.py` 수정

#### 변경 사항
- **모델로부터 depth 가져오기**: `self.cfg["cnn_depth"]` 대신 `len(model.xhconv_blks)` 사용
- **active_layers 속성 추가**: `self.active_layers`를 인스턴스 속성으로 저장하여 BaseExperiment에서 재사용

#### 수정 코드
```python
def _create_model(self):
    model = HybridTS(self.cfg, self.n_vars)
    
    # 모델로부터 실제 depth를 가져옴 (하드코딩 방지)
    depth = len(model.xhconv_blks)
    cross_layers = self.cfg.get("cross_layers", "all")
    
    # ... active_layers 계산 로직 ...
    
    # active_layers를 인스턴스 속성으로 저장 (BaseExperiment에서 재사용)
    self.active_layers = active_layers
    model.set_cross_layers(active_layers)
    return model
```

#### 효과
- 설정 파일에 `cnn_depth`가 없거나 모델과 불일치해도 자동으로 올바른 depth 사용
- BaseExperiment에서 중복 계산 없이 active_layers 재사용 가능

---

### 2. `experiments/base_experiment.py` 수정

#### 변경 사항
1. **active_layers 속성 초기화**: `self.active_layers = None` 추가
2. **중간층 표현 수집 훅 추가**: W4 실험 시 각 활성 층의 CNN/GRU 표현을 자동 수집
3. **중복 로직 제거**: W4의 active_layers를 중복 계산하지 않고 `self.active_layers` 사용

#### 주요 추가 코드

##### __init__ 메서드
```python
def __init__(self, cfg):
    # ... 기존 코드 ...
    self.active_layers = None  # W4 실험에서 사용
```

##### evaluate_test 메서드 - 훅 등록
```python
# W4 실험이면 중간층 표현 수집을 위한 훅 등록
hooks_data = {}
hooks = []
if self.experiment_type == "W4" and self.active_layers:
    # 층별로 표현을 저장할 딕셔너리 초기화
    cnn_repr_by_layer = {i: [] for i in self.active_layers}
    gru_repr_by_layer = {i: [] for i in self.active_layers}
    
    def make_hook(layer_idx, repr_dict):
        def hook_fn(module, input, output):
            # output: zc or zr (T, B, d)
            if isinstance(output, tuple):
                repr_tensor = output[0]
            else:
                repr_tensor = output
            
            # (T, B, d) -> (B, d) 평균
            repr_mean = repr_tensor.mean(dim=0).detach().cpu().numpy()
            repr_dict[layer_idx].append(repr_mean)
        return hook_fn
    
    # 각 활성 층에 훅 등록
    for i in self.active_layers:
        if i < len(self.model.conv_blks):
            h1 = self.model.conv_blks[i].register_forward_hook(
                make_hook(i, cnn_repr_by_layer)
            )
            hooks.append(h1)
        if i < len(self.model.gru_blks):
            h2 = self.model.gru_blks[i].register_forward_hook(
                make_hook(i, gru_repr_by_layer)
            )
            hooks.append(h2)
```

##### evaluate_test 메서드 - 훅 정리 및 데이터 저장
```python
finally:
    # 훅 제거
    for h in hooks:
        h.remove()
    
    # W4: 수집된 표현을 hooks_data에 저장
    if self.experiment_type == "W4" and self.active_layers:
        import numpy as np
        cnn_repr_list = []
        gru_repr_list = []
        
        # 활성 층 순서대로 정리
        for i in self.active_layers:
            if cnn_repr_by_layer[i]:
                cnn_repr_list.append(np.concatenate(cnn_repr_by_layer[i], axis=0))
            else:
                cnn_repr_list.append(None)
            
            if gru_repr_by_layer[i]:
                gru_repr_list.append(np.concatenate(gru_repr_by_layer[i], axis=0))
            else:
                gru_repr_list.append(None)
        
        hooks_data["cnn_representations"] = cnn_repr_list
        hooks_data["gru_representations"] = gru_repr_list
```

##### evaluate_test 메서드 - 중복 로직 제거
```python
# W4의 경우 self.active_layers 사용 (중복 계산 제거)
active_layers = self.active_layers if self.experiment_type == "W4" else []

exp_specific = compute_all_experiment_metrics(
    experiment_type=self.experiment_type,
    model=self.model,
    hooks_data=hooks_data if hooks_data else None,
    # ... 기타 파라미터 ...
    active_layers=active_layers,
)
```

#### 효과
- W4 실험 시 중간층 표현이 자동으로 수집됨
- `w4_layerwise_representation_similarity` 지표가 정상적으로 계산됨
- 코드 중복 제거로 유지보수성 향상

---

### 3. `utils/experiment_metrics/w4_metrics.py` 수정

#### 변경 사항
- **표현 유사도 계산 로직 강화**: None 체크, 예외 처리 추가
- **더 명확한 주석**: 데이터 구조 설명 추가

#### 수정 코드
```python
# 층별 표현 유사도 (CNN↔GRU)
if hooks_data is not None:
    cnn_repr = hooks_data.get("cnn_representations")
    gru_repr = hooks_data.get("gru_representations")
    
    if cnn_repr is not None and gru_repr is not None:
        # cnn_repr와 gru_repr는 활성 층 순서대로 정렬된 리스트
        # cnn_repr[idx] = 활성 층 active_layers[idx]의 표현
        similarities = []
        
        for idx in range(len(active_layers)):
            if idx < len(cnn_repr) and idx < len(gru_repr):
                cnn_i = cnn_repr[idx]
                gru_i = gru_repr[idx]
                
                # None 체크
                if cnn_i is None or gru_i is None:
                    continue
                
                # numpy 배열로 변환
                if not isinstance(cnn_i, np.ndarray):
                    cnn_i = np.array(cnn_i)
                if not isinstance(gru_i, np.ndarray):
                    gru_i = np.array(gru_i)
                
                # shape 확인 및 distance correlation 계산
                if cnn_i.shape[0] == gru_i.shape[0] and cnn_i.shape[0] > 1:
                    try:
                        sim = _dcor_u(cnn_i, gru_i)
                        if np.isfinite(sim):
                            similarities.append(sim)
                    except Exception:
                        # distance correlation 계산 실패 시 스킵
                        pass
        
        if len(similarities) > 0:
            metrics["w4_layerwise_representation_similarity"] = float(np.mean(similarities))
        else:
            metrics["w4_layerwise_representation_similarity"] = np.nan
```

#### 효과
- 예외 상황에서도 안정적으로 동작
- distance correlation 계산 실패 시 graceful degradation

---

## 사용 방식 변경

### 변경 사항 없음
기존과 동일한 방식으로 사용 가능합니다.

### 개선 사항
1. **더 안전한 depth 처리**: 설정 파일에 `cnn_depth`를 명시하지 않아도 자동으로 모델로부터 가져옴
2. **자동 표현 수집**: W4 실험 실행 시 중간층 표현이 자동으로 수집되어 지표 계산에 사용됨
3. **향상된 안정성**: 예외 상황에서도 안정적으로 동작

### 실행 예시
```bash
# 기존과 동일하게 실행 가능
python main.py --experiment W4 --dataset ETTh2 --horizon 192 --seed 42 --mode shallow

# 또는 전체 실험 실행
python run_all_experiments.py
```

---

## 테스트

### 테스트 스크립트
수정 사항을 검증하기 위한 테스트 스크립트가 제공됩니다: `test_w4_modifications.py`

### 실행 방법
```bash
python test_w4_modifications.py
```

### 테스트 항목
1. **active_layers 속성 확인**: 각 모드(all, shallow, mid, deep)에서 올바른 층이 활성화되는지 확인
2. **중간층 표현 수집 구조 확인**: 훅을 통해 수집된 데이터의 형식이 올바른지 확인
3. **W4 지표 계산 확인**: 모든 필수 지표가 올바르게 계산되는지 확인

---

## 영향 범위

### 수정된 파일
1. `experiments/w4_experiment.py`
2. `experiments/base_experiment.py`
3. `utils/experiment_metrics/w4_metrics.py`

### 영향받는 실험
- **W4 실험만 영향받음**
- W1, W2, W3, W5 실험은 영향받지 않음

### 호환성
- **기존 설정 파일과 완전 호환**
- **기존 실행 스크립트와 완전 호환**
- **기존 결과 CSV 형식과 완전 호환**

---

## 검증 체크리스트

- [x] 린터 오류 없음
- [x] 기존 설정 파일과 호환
- [x] 실행 스크립트 (`run_all_experiments.py`, `main.py`) 호환
- [x] W4 특화 지표 계산 로직 개선
- [x] 중간층 표현 자동 수집 구현
- [x] 테스트 스크립트 작성
- [x] 문서화 완료

---

## 추가 참고 사항

### 중간층 표현 데이터 구조
```python
hooks_data = {
    "cnn_representations": [
        np.ndarray,  # Layer 0: shape (total_samples, d_embed)
        np.ndarray,  # Layer 1: shape (total_samples, d_embed)
        ...
    ],
    "gru_representations": [
        np.ndarray,  # Layer 0: shape (total_samples, d_embed)
        np.ndarray,  # Layer 1: shape (total_samples, d_embed)
        ...
    ]
}
```

### W4 지표
- `w4_layerwise_gate_usage`: 활성화된 층들의 게이트 사용률 평균
- `w4_layer_contribution_score`: 정규화된 기여 점수 (NCS)
- `w4_layerwise_representation_similarity`: 활성층의 CNN↔GRU 표현 유사도 (Distance Correlation)

---

## 문의 사항
수정 사항에 대한 문의나 문제가 발생하면 다음을 확인하세요:
1. 린터 오류 확인: `python -m pylint experiments/w4_experiment.py`
2. 테스트 실행: `python test_w4_modifications.py`
3. 로그 확인: 실험 실행 시 상세 로그 출력

