# CTSF-W Ablation Study

CTSF 모델의 Ablation Study를 위한 모듈화된 코드베이스입니다.

## ⚠️ 중요 경고

**모델 구조는 절대 변경하지 마세요!** 
- `models/ctsf_model.py`의 기본 모델 구조는 변경 금지
- 실험별 변형은 `models/experiment_variants.py`에서만 추가
- 모델 구조 변경 시 기존 실험 결과가 무효화될 수 있습니다

## 구조

```
.
├── CHANGES_SUMMARY.md
├── config/
│   └── config.py
├── CTSF-W-Overview.md
├── data/
│   ├── __init__.py
│   └── dataset.py
├── datasets/
├── docs/
│   ├── exp-plan/
│   ├── experiment_modifications/
│   ├── original-code/
│   └── Papers/
├── environment-check.ipynb
├── experiments/
│   ├── base_experiment.py
│   ├── w1_experiment.py
│   ├── w2_experiment.py
│   ├── w3_experiment.py
│   ├── w4_experiment.py
│   └── w5_experiment.py
├── hp2_config.yaml
├── main.py
├── models/
│   ├── ctsf_model.py
│   └── experiment_variants.py
├── results/
├── results_test/
├── run_all_experiments.py
├── run_suite.py
├── test_single_experiments.py
├── test_w3_feedback_fixes.py
└── utils/
    ├── csv_logger.py
    ├── direct_evidence.py
    ├── error_logger.py
    ├── experiment_metrics/
    │   ├── all_metrics.py
    │   ├── w1_metrics.py
    │   ├── w2_metrics.py
    │   ├── w3_metrics.py
    │   ├── w4_metrics.py
    │   └── w5_metrics.py
    ├── hooks.py
    ├── metrics.py
    ├── plot_results.py
    ├── plotting_metrics.py
    └── training.py
```

## 실험 개요

### W1: 층별 교차 vs. 최종 결합 (Per-layer Cross vs. Last-layer Fusion)
- **목적**: 매 층마다 교차 연결을 수행하는 것과 최종 층에서만 결합하는 것의 효과 비교
- **모드**: 
  - `per_layer`: 모든 층에서 교차 연결 수행 (원안)
  - `last_layer`: 최종 층에서만 결합 (대조군)
- **평가 지표**: CKA/CCA 유사도, 층별 상향 개선도, 그래디언트 정렬

### W2: 동적 교차 vs. 정적 교차 (Dynamic Cross vs. Static Cross)
- **목적**: 동적 hypernetwork가 생성하는 게이트의 효과 검증
- **모드**: 
  - `dynamic`: 동적 게이트 사용 (원안)
  - `static`: 정적 게이트 사용 (대조군)
- **평가 지표**: 게이트 변동성, 게이트-TOD 정렬, 이벤트 조건 반응, 채널 선택도

### W3: 데이터 구조 원인 확인 (Data Structure Cause Verification)
- **목적**: 데이터 특성(TOD, 피크 패턴 등)별 성능 원인 검증
- **모드**: 
  - `none`: 교란 없음 (baseline)
  - `tod_shift`: 시간대(TOD) 패턴 교란 (ETTm2 데이터셋용)
  - `smooth`: 급격한 변화 패턴 평활화 (ETTh2 데이터셋용)
- **평가 지표**: 개입 효과 (ΔRMSE, ΔTOD, Δpeak, Cohen's d), 순위 보존률, 라그 분포 변화

### W4: 교차 층 기여도 분석 (Cross-layer Contribution Analysis)
- **목적**: 어느 층의 교차 연결이 가장 중요한지 분석
- **모드**: 
  - `all`: 모든 층에서 교차 연결
  - `shallow`: 얕은 층만 교차 연결
  - `mid`: 중간 층만 교차 연결
  - `deep`: 깊은 층만 교차 연결
- **평가 지표**: 층별 기여 점수, 층별 게이트 사용률, 층별 표현 유사도

### W5: 게이트 고정 시험 (Gate Fixing Test)
- **목적**: 동적 적응성의 순효과 확인 (동적→정적 중간 형태)
- **모드**: 
  - `dynamic`: 동적 게이트 사용 (원안)
  - `fixed`: 게이트를 평균값으로 고정 (대조군)
- **평가 지표**: 성능 저하율, 민감도/이벤트 이득 손실, 게이트-이벤트 정렬 손실

## 사용법

### 단일 실험 실행

```bash
python main.py \
    --experiment W1 \
    --dataset ETTh2 \
    --horizon 192 \
    --seed 42 \
    --mode per_layer
```

### 실험 스위트 실행 (특정 실험의 여러 설정)

```bash
# W1 실험: 지정된 시드, 호라이즌, 데이터셋에 대해 실행
python run_suite.py \
    --experiment W1 \
    --seeds 42 2 3 \
    --horizons 96 192 \
    --datasets ETTh2 ETTm2 \
    --resume next
```

### 모든 시드, 데이터셋, 호라이즌에 대해 특정 실험 실행

**참고**: `--seeds`, `--horizons`, `--datasets` 옵션을 생략하면 기본값이 사용됩니다.

#### W1 실험 전체 실행

```bash
# 옵션 생략 시 기본값 사용 (모든 시드/데이터셋/호라이즌)
python run_suite.py --experiment W1 --resume next

# 또는 명시적으로 지정
python run_suite.py \
    --experiment W1 \
    --seeds 42 2 3 5 7 11 13 17 19 23 \
    --horizons 96 192 336 720 \
    --datasets ETTm1 ETTm2 ETTh1 ETTh2 weather \
    --resume next
```

#### W2 실험 전체 실행

```bash
# 옵션 생략 시 기본값 사용
python run_suite.py --experiment W2 --resume next

# 또는 명시적으로 지정
python run_suite.py \
    --experiment W2 \
    --seeds 42 2 3 5 7 11 13 17 19 23 \
    --horizons 96 192 336 720 \
    --datasets ETTm1 ETTm2 ETTh1 ETTh2 weather \
    --resume next
```

**기본값**:
- `--seeds`: `42 2 3 5 7 11 13 17 19 23` (10개 시드)
- `--horizons`: `96 192 336 720` (4개 호라이즌)
- `--datasets`: `ETTm1 ETTm2 ETTh1 ETTh2 weather` (5개 데이터셋)

### 전체 실험 (W1~W5) 순차 실행

```bash
# 옵션 생략 시 기본값 사용 (모든 시드/데이터셋/호라이즌)
python run_all_experiments.py --resume next

# 또는 명시적으로 지정
python run_all_experiments.py \
    --seeds 42 2 3 5 7 11 13 17 19 23 \
    --horizons 96 192 336 720 \
    --datasets ETTm1 ETTm2 ETTh1 ETTh2 weather \
    --resume next
```

**참고**: `run_all_experiments.py`는 W1~W5를 순차적으로 실행하며, 각 실험의 기본 모드들을 자동으로 설정합니다.

### 빠른 테스트 (단일 실험 검증)

각 실험의 모든 모드를 빠르게 테스트하고 싶을 때 사용합니다:

```bash
# 모든 실험(W1~W5)의 모든 모드 테스트
python test_single_experiments.py
```

**테스트 설정**:
- 데이터셋: `ETTh1` (가장 빠른 데이터셋)
- 호라이즌: `96` (가장 빠른 호라이즌)
- 시드: `42`
- Epochs: `5` (빠른 검증용, 원래는 100)
- Early Stop Patience: `3` (빠른 검증용, 원래는 20)

**테스트 범위**:
- W1: `per_layer`, `last_layer`
- W2: `dynamic`, `static`
- W3: `none`, `tod_shift`, `smooth`
- W4: `all`, `shallow`, `mid`, `deep`
- W5: `dynamic`, `fixed`

**결과 저장**: 테스트 결과는 `results_test/` 폴더에 저장됩니다.

### Resume 모드

- `next`: 마지막 완료 실험 이후부터 계속 실행
- `fill_missing`: 누락된 실험만 실행 (완료된 실험은 스킵)
- `all`: 모든 실험 재실행 (완료된 실험도 다시 실행)

### 추가 옵션

#### run_suite.py 옵션

```bash
# 상세 로그 출력
python run_suite.py --experiment W1 --verbose

# 완료된 실험도 재실행
python run_suite.py --experiment W1 --overwrite

# 계획만 출력 (실제 실행 안 함)
python run_suite.py --experiment W1 --dry_run

# 실험당 최대 실행 개수 제한
python run_suite.py --experiment W1 --max_jobs 10
```

#### run_all_experiments.py 옵션

```bash
# 상세 로그 출력
python run_all_experiments.py --verbose

# 완료된 실험도 재실행
python run_all_experiments.py --overwrite

# 계획만 출력 (실제 실행 안 함)
python run_all_experiments.py --dry_run

# 실험당 최대 실행 개수 제한
python run_all_experiments.py --max_jobs_per_exp 10
```

## 환경 설정

### 필수 라이브러리

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.3.0
tqdm
scipy (W3 실험용)
pyyaml (설정 파일용)
```

### 재현성 설정

코드는 자동으로 다음 설정을 적용합니다:
- 시드 고정 (torch, numpy, random)
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`
- `torch.set_float32_matmul_precision('highest')`

## 결과 저장

### CSV 결과 파일

- **실험별 분리 저장**: `results/results_W1.csv`, `results_W2.csv`, ..., `results_W5.csv`
- 각 실험 완료 시 자동으로 추가/업데이트됩니다
- 중복 제거: 동일한 (dataset, horizon, seed, mode) 조합은 자동으로 업데이트

### 보고용 그림 데이터

- **저장 경로**: `results/results_W{실험번호}/{데이터셋}/{그림타입}/{그림타입}_summary.csv`
- **예시**: 
  - `results/results_W1/ETTm2/forest_plot/forest_plot_summary.csv`
  - `results/results_W2/ETTh2/gate_tod_heatmap/gate_tod_heatmap_summary.csv`
- **그림 타입**:
  - W1: `forest_plot`, `cka_heatmap`, `grad_align_bar`
  - W2: `forest_plot`, `gate_tod_heatmap`, `gate_distribution`
  - W3: `effect_size_bar`, `rank_preservation`, `lag_distribution`
  - W4: `rmse_line`, `gate_usage_bar`, `cka_heatmap`
  - W5: `degradation_bar`, `gate_distribution`
- **참고**: 그림 그리기 코드는 포함되지 않습니다. 그림 데이터만 CSV로 저장되며, 별도 스크립트로 그림을 그려야 합니다.

### 체크포인트 관리

- Best 모델은 학습 중 자동 저장됩니다
- 테스트 완료 후 자동으로 삭제되어 저장 공간을 절약합니다

### 오류 로깅

- 실패한 실험의 상세 정보는 `results/errors_W*.json`에 저장됩니다
- 실험 타입, 데이터셋, 시드, 호라이즌, 오류 메시지 등이 포함됩니다

## 편의 기능

- **tqdm**: 진행 상황 표시
- **안전한 상관 계산**: NaN 처리
- **Resume 기능**: 중단 후 이어서 실행
- **자동 리소스 정리**: GPU 메모리 정리 (`torch.cuda.empty_cache()`, `gc.collect()`)
- **체크포인트 관리**: Best 모델 저장/삭제
- **시간 측정**: 실험별 소요 시간 추적
- **오류 처리**: 실험 실패 시에도 다른 실험 계속 진행

## 평가 지표

### 공통 지표
- 기본 성능: RMSE, MAE, MSE
- Conv→GRU 경로: Pearson/Spearman 상관, 거리상관, 이벤트 게인/적중률, 최대 동행도/최적 지연
- GRU→Conv 경로: 커널-TOD 상관, 특징-TOD 상관, 커널-특징 정렬

### 실험별 특화 지표

#### W1
- `w1_cka_similarity_cnn_gru`: CKA 유사도
- `w1_cca_similarity_cnn_gru`: CCA 유사도
- `w1_layerwise_upward_improvement`: 층별 상향 개선도
- `w1_inter_path_gradient_align`: 경로 간 그래디언트 정렬
- 보고용: `cka_s/m/d`, `grad_align_s/m/d`

#### W2
- `w2_gate_variability_time`: 시간별 게이트 변동성
- `w2_gate_variability_sample`: 샘플별 게이트 변동성
- `w2_gate_entropy`: 게이트 엔트로피
- `w2_gate_tod_alignment`: 게이트-TOD 정렬
- `w2_gate_gru_state_alignment`: 게이트-GRU 상태 정렬
- `w2_event_conditional_response`: 이벤트 조건 반응
- `w2_channel_selectivity_kurtosis`: 채널 선택도 (첨도)
- `w2_channel_selectivity_sparsity`: 채널 선택도 (희소성)
- 보고용: `gate_tod_mean_s/m/d`, `gate_var_t/b`, `gate_entropy`, `gate_channel_kurt/sparsity`, `gate_q10/50/90`, `gate_hist10`

#### W3
- `w3_intervention_effect_rmse`: 개입 효과 (ΔRMSE)
- `w3_intervention_effect_tod`: 개입 효과 (ΔTOD)
- `w3_intervention_effect_peak`: 개입 효과 (Δpeak)
- `w3_intervention_cohens_d`: Cohen's d (효과 크기)
- `w3_rank_preservation_rate`: 순위 보존률
- `w3_lag_distribution_change`: 라그 분포 변화
- 보고용: `bestlag_neg_ratio`, `bestlag_var`, `bestlag_hist21`

#### W4
- `w4_layer_contribution_score`: 층별 기여 점수
- `w4_layerwise_gate_usage`: 층별 게이트 사용률
- `w4_layerwise_representation_similarity`: 층별 표현 유사도
- 보고용: `cka_s/m/d`

#### W5
- `w5_performance_degradation_ratio`: 성능 저하율
- `w5_sensitivity_gain_loss`: 민감도 이득 손실
- `w5_event_gain_loss`: 이벤트 이득 손실
- `w5_gate_event_alignment_loss`: 게이트-이벤트 정렬 손실
- 보고용: `gate_var_t/b`, `gate_entropy`, `gate_q10/50/90`, `gate_hist10`

## 주의사항

1. **모델 구조 변경 금지**: `models/ctsf_model.py`는 절대 수정하지 마세요
2. **실험별 최소 변경**: 실험별로 필요한 최소 변경만 허용
3. **리소스 관리**: 대량 실험 시 GPU 메모리 모니터링 필요
4. **결과 백업**: 중요한 결과는 별도 백업 권장
5. **하이퍼파라미터 일치**: `hp2_config.yaml`의 설정을 준수해야 실험 결과의 일관성 유지
6. **데이터 분할 일치**: 기존 `CTSF-V1.py`와 동일한 데이터 분할 방식 사용

## 변경 내역

코드 변경 내역은 `CHANGES_SUMMARY.md`에 기록됩니다.
- 코드 수정, 추가, 삭제 등의 동작 내역
- 구조나 실험 수행 방법 변경 사항은 `README.md`에 반영
