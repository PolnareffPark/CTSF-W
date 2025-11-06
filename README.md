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
├── models/              # 모델 정의
│   ├── ctsf_model.py   # 기본 CTSF 모델 (절대 변경 금지)
│   └── experiment_variants.py  # 실험별 변형
├── data/               # 데이터 로딩
│   └── dataset.py
├── utils/              # 유틸리티
│   ├── metrics.py      # 평가 지표
│   ├── hooks.py        # Forward/Backward hooks
│   ├── direct_evidence.py  # 직접 근거 지표
│   ├── training.py      # 학습/평가 함수
│   └── csv_logger.py   # CSV 저장
├── experiments/        # 실험 실행 코드
│   ├── base_experiment.py
│   ├── w1_experiment.py
│   ├── w2_experiment.py
│   ├── w3_experiment.py
│   ├── w4_experiment.py
│   └── w5_experiment.py
├── config/             # 설정 관리
│   └── config.py
├── main.py            # 단일 실험 실행
├── run_suite.py       # 실험 스위트 실행
└── hp2_config.yaml    # HP2 설정 파일
```

## 실험 개요

### W1: 층별 교차 vs. 최종 결합
- **목적**: 매 층 교차의 효과 검증
- **모드**: `per_layer` (원안), `last_layer` (대조군)

### W2: 동적 교차 vs. 정적 교차
- **목적**: 동적 hypernetwork의 효과 검증
- **모드**: `dynamic` (원안), `static` (대조군)

### W3: 데이터 구조 원인 확인
- **목적**: 데이터 특성별 성능 원인 검증
- **모드**: `none`, `tod_shift`, `smooth`

### W4: 교차 층 기여도 분석
- **목적**: 어느 층의 교차가 중요한지 분석
- **모드**: `all`, `shallow`, `mid`, `deep`

### W5: 게이트 고정 시험
- **목적**: 동적 적응성의 순효과 확인
- **모드**: `dynamic`, `fixed`

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

### 실험 스위트 실행

```bash
python run_suite.py \
    --experiment W1 \
    --seeds 42 2 3 \
    --horizons 96 192 \
    --datasets ETTh2 ETTm2 \
    --resume next
```

### Resume 모드

- `next`: 마지막 완료 실험 이후부터 계속
- `fill_missing`: 누락된 실험만 실행
- `all`: 모든 실험 재실행

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

- 결과는 `results/results.csv`에 저장됩니다
- 각 실험 완료 시 자동으로 추가/업데이트됩니다
- 체크포인트는 테스트 완료 후 자동 삭제됩니다

## 편의 기능

- **tqdm**: 진행 상황 표시
- **안전한 상관 계산**: NaN 처리
- **Resume 기능**: 중단 후 이어서 실행
- **자동 리소스 정리**: GPU 메모리 정리
- **체크포인트 관리**: Best 모델 저장/삭제

## 주의사항

1. **모델 구조 변경 금지**: `models/ctsf_model.py`는 절대 수정하지 마세요
2. **실험별 최소 변경**: 실험별로 필요한 최소 변경만 허용
3. **리소스 관리**: 대량 실험 시 GPU 메모리 모니터링 필요
4. **결과 백업**: 중요한 결과는 별도 백업 권장