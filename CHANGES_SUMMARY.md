# CTSF-W Ablation Study 코드 수정 사항 요약

## ✅ 완료된 수정 사항

### 1. 데이터 분할 방식
- ✅ 기존 `CTSF-V1.py`와 동일한 분할 방식 유지
- ✅ `tr_end + L + H - 1` 인덱싱 방식 동일
- ✅ train/val/test 분할 로직 동일

### 2. 하이퍼파라미터
- ✅ Loss: `ps_loss` (Patch-Stat + MSE) - 기존과 동일
- ✅ Optimizer: `AdamW` (lr=1e-4, weight_decay=1e-3) - 기존과 동일
- ✅ Scheduler: `OneCycleLR` (pct_start=0.15, div_factor=25.0, final_div_factor=1e4) - 기존과 동일
- ✅ EarlyStop: patience=20 - 기존과 동일
- ✅ Gradient clipping: 0.5 - 기존과 동일

### 3. 실험 파일 구현
- ✅ **W1**: `per_layer` vs `last_layer` 모드 구현 완료
- ✅ **W2**: `dynamic` vs `static` 모드 구현 완료
- ✅ **W3**: 데이터 교란 (`tod_shift`, `smooth`) 구현 완료
- ✅ **W4**: `shallow`/`mid`/`deep` 교차 층 제어 구현 완료
- ✅ **W5**: 게이트 고정 래퍼 구현 완료

### 4. CSV 저장
- ✅ 실험별로 분리 저장: `results_W1.csv`, `results_W2.csv`, ...
- ✅ `read_results_rows` 함수가 실험별 파일 읽기 지원

### 5. 전체 실험 실행
- ✅ `run_all_experiments.py` 추가: W1~W5 순차 실행
- ✅ 각 실험 독립 실행 (하나 실패해도 나머지 계속)
- ✅ 전체/개별 실험 시간 측정 및 출력

### 6. 로그 출력 개선
- ✅ `verbose` 모드: 상세 로그 vs 간단 진행 상황
- ✅ 전체 실험 시 간단한 진행 상황만 출력
- ✅ 개별 실험 시간 측정 및 저장

### 7. 리소스 관리
- ✅ `gc.collect()` 사용 (cleanup_resources에서)
- ✅ GPU 메모리 정리 자동화
- ✅ 체크포인트 자동 삭제

### 8. Metrics 함수 사용
- ✅ `_spearman_corr`, `_time_deriv_norm`, `_event_gain_and_hit` 등 모두 `direct_evidence.py`에서 사용 중
- ✅ 모든 함수가 실제로 호출됨

### 9. 코드 정리
- ✅ 사용하지 않는 `_try_int` 함수 제거
- ✅ 중복 docstring 수정

## 📝 추가 지표 구현 (향후 확장)

사용자가 요청한 추가 지표들은 다음과 같이 구조화되어 있습니다:

### 공통 지표 (A)
- MASE, sMAPE
- 잔차 ACF, Ljung-Box
- 레짐별 성능 (TOD bin RMSE 등)
- 효율 지표 (FLOPs, 파라미터 수 등)

### 실험별 특화 지표 (B)
- **W1**: CKA/CCA 유사도, 층별 상향 개선도, 그래디언트 정렬
- **W2**: 게이트 변동성, 게이트-문맥 정렬, 이벤트 조건 반응
- **W3**: 개입 효과, 순위 보존률, 라그 분포 변화
- **W4**: 층별 기여 점수, 게이트 사용률
- **W5**: 성능 저하율, 민감도 손실

이러한 지표들은 `utils/direct_evidence.py`와 `utils/metrics.py`에 확장 가능한 구조로 되어 있으며, 필요시 추가 구현 가능합니다.

## ⚠️ 중요 사항

1. **모델 구조 절대 변경 금지**: `models/ctsf_model.py`는 수정하지 않음
2. **하이퍼파라미터 일치**: 기존 `CTSF-V1.py`와 동일한 설정 사용
3. **데이터 분할 일치**: 기존과 동일한 분할 방식 유지
4. **실험별 최소 변경**: 각 실험은 필요한 최소 변경만 수행

## 🚀 사용법

### 개별 실험 실행
```bash
python run_suite.py --experiment W1 --seeds 42 2 3 --horizons 96 192
```

### 전체 실험 실행
```bash
python run_all_experiments.py --seeds 42 2 3 --horizons 96 192
```

### 단일 실험 실행
```bash
python main.py --experiment W1 --dataset ETTh2 --horizon 192 --seed 42 --mode per_layer
```
