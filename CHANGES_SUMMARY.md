## 📝 최근 변경 내역

### 2025-01-XX: 보고용 그림 지표 및 결과 저장 구조 추가

#### 1. 하이퍼파라미터 처리 개선
- **파일**: `config/config.py`, `models/ctsf_model.py`
- **변경**: `hp2_config.yaml`의 `alpha_init`, `revin` 하위 구조를 올바르게 처리하도록 수정
- **상세**: 
  - `alpha_init_diag`, `alpha_init_offdiag` 추출 및 모델에 전달
  - `revin_affine` 추출 및 `RevIN` 모듈에 적용
  - `conv_kernel`, `hyperconv_k` 하이퍼파라미터 사용

#### 2. direct_evidence.py 수정
- **파일**: `utils/direct_evidence.py`
- **변경**: `cg_on` 조건을 명확히 하고, both 환경에서도 정상 작동하도록 주석 추가
- **상세**:
  - Conv→GRU 지표 계산 시 `cg_on` 조건 추가 (76번 줄)
  - Conv→GRU 요약 지표도 `cg_on`이 True일 때만 계산 (124-139번 줄)
  - both 환경에서는 `cg_on=True`이므로 정상 작동함을 명시

#### 3. 실험별 특화 지표 구현
- **파일**: `utils/experiment_metrics/` 폴더 생성
- **변경**: W1~W5 각 실험별 특화 지표를 별도 파일로 분리
- **상세**:
  - `w1_metrics.py`: CKA/CCA 유사도, 층별 상향 개선도, 그래디언트 정렬
  - `w2_metrics.py`: 게이트 변동성, 게이트-TOD 정렬, 이벤트 조건 반응, 채널 선택도
  - `w3_metrics.py`: 개입 효과, 순위 보존률, 라그 분포 변화 (구현 완료)
  - `w4_metrics.py`: 층별 기여 점수, 층별 게이트 사용률, 층별 표현 유사도
  - `w5_metrics.py`: 성능 저하율, 민감도/이벤트 이득 손실, 게이트-이벤트 정렬 손실 (구현 완료)
  - `all_metrics.py`: 통합 함수 제공

#### 4. 보고용 그림 지표 모듈 추가
- **파일**: `utils/plotting_metrics.py` (신규)
- **변경**: 보고용 그림 생성을 위한 추가 지표 계산 모듈 생성
- **상세**:
  - `compute_layerwise_cka()`: 층별 CKA 유사도 (얕/중/깊)
  - `compute_gradient_alignment()`: 층별 그래디언트 정렬 (얕/중/깊)
  - `compute_gate_tod_heatmap()`: 시간대별 게이트 히트맵 데이터 (24-bin)
  - `compute_gate_distribution()`: 게이트 분포 통계
  - `compute_bestlag_distribution()`: 라그 분포 요약 통계
  - `compute_all_plotting_metrics()`: 통합 함수
- **참고**: 그림 그리기 코드는 포함되지 않음. 그림 데이터만 계산하여 CSV로 저장

#### 5. 결과 저장 구조 개선
- **파일**: `utils/plot_results.py` (신규), `experiments/base_experiment.py`
- **변경**: `results/results_W1/dataset/plot_type/plot_summary.csv` 형식으로 그림 데이터 저장
- **상세**:
  - `save_plot_data()`: 그림 데이터를 실험별/데이터셋별/그림타입별로 저장
  - `load_plot_data()`: 저장된 그림 데이터 로드
  - `base_experiment.py`의 `save_results()`에서 자동으로 그림 데이터 저장
- **폴더 구조**:
  ```
  results/
    results_W1/
      ETTm2/
        forest_plot/
          forest_plot_summary.csv
        cka_heatmap/
          cka_heatmap_summary.csv
    results_W2/
      ...
  ```

#### 6. CSV 컬럼 확장
- **파일**: `utils/csv_logger.py`
- **변경**: 보고용 그림 지표 컬럼 추가
- **상세**:
  - W1: `cka_s/m/d`, `grad_align_s/m/d`
  - W2: `gate_tod_mean_s/m/d`, `gate_var_t/b`, `gate_entropy`, `gate_channel_kurt/sparsity`, `gate_q10/50/90`, `gate_hist10`
  - W3: `bestlag_neg_ratio`, `bestlag_var`, `bestlag_hist21`
  - W4: `cka_s/m/d`
  - W5: `gate_var_t/b`, `gate_entropy`, `gate_q10/50/90`, `gate_hist10`

#### 7. 코드 정리
- **파일**: `utils/plotting_metrics.py`
- **변경**: 사용하지 않는 import 제거 및 주석 추가
- **상세**: `_cca_similarity`는 향후 사용 가능하므로 `# noqa: F401` 주석 추가
