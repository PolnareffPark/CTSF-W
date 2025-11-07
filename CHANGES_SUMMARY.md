## 📝 최근 변경 내역

### 2025-11-07: W5 실험 코드 개선 (평가 피드백 반영)

#### 1. W5Experiment.evaluate_test() 완전 재구성
- **파일**: `experiments/w5_experiment.py`
- **문제**: 기존 코드는 `gate_fixed` 플래그에 따라 동적 또는 고정 중 하나만 평가하여 W5 비교 지표가 계산되지 않음
- **해결**: 한 번의 실행에서 동적과 고정 게이트를 모두 평가하도록 수정
- **상세**:
  - **동적 게이트 평가**: 원래 학습된 모델로 테스트 성능 평가
  - **고정 게이트 평가**: `GateFixedModel` 래퍼를 적용하여 게이트를 평균값으로 고정한 후 평가
  - **W5 지표 계산**: `compute_w5_metrics()`에 두 결과를 전달하여 비교 지표 계산
  - **결과 병합**: 동적 모델 지표 + W5 비교 지표 + 고정 모델 개별 지표(rmse_fixed 등)
- **효과**: 
  - `w5_performance_degradation_ratio` 등 비교 지표가 정상적으로 계산됨
  - CSV에 동적/고정 성능이 모두 기록되어 분석 용이
  - `gate_fixed` 플래그 불필요 (run_tag에서 제거)

#### 2. W5 지표 계산 로직 명확화
- **파일**: `utils/experiment_metrics/w5_metrics.py`
- **변경**: docstring 개선 및 지표 해석 추가
- **상세**:
  - **성능 저하율**: `(rmse_fixed - rmse_dynamic) / rmse_dynamic`
    - 양수면 고정 시 성능 악화, 음수면 오히려 개선
  - **민감도 손실**: `tod_dynamic - tod_fixed`
    - 양수면 동적 게이트가 시간대 패턴을 더 잘 포착
  - **이벤트 손실**: `event_gain_dynamic - event_gain_fixed`
    - 양수면 동적 게이트가 이벤트를 더 잘 탐지
  - **정렬 손실**: 게이트 변동성과 이벤트 게인의 곱 차이
    - 동적 게이트는 이벤트 발생 시 크게 변동하지만, 고정 게이트는 변화 없음

#### 3. 고정 모델 개별 지표 추가
- **파일**: `experiments/w5_experiment.py`
- **변경**: 고정 모델의 주요 성능 지표를 별도 키로 저장
- **상세**:
  - `rmse_fixed`: 고정 게이트 모델의 RMSE
  - `mae_fixed`: 고정 게이트 모델의 MAE
  - `gc_kernel_tod_dcor_fixed`: 고정 게이트 모델의 TOD 민감도
  - `cg_event_gain_fixed`: 고정 게이트 모델의 이벤트 게인
- **효과**: 절대값 비교 및 분석이 용이해짐

#### 4. 테스트 및 검증
- **신규 파일**: `docs/experiment_modifications/test_w5_modifications.py`
- **테스트 항목**:
  - `test_gate_fixed_model()`: GateFixedModel이 게이트를 올바르게 고정하는지 검증
  - `test_w5_metrics_computation()`: W5 지표 계산 정확성 검증
  - `test_w5_metrics_with_missing_data()`: 누락된 데이터 처리 검증
  - `test_w5_evaluate_test_integration()`: evaluate_test 통합 로직 검증
- **실행**: `python docs/experiment_modifications/test_w5_modifications.py`

#### 5. 실험 실행 방식 변경
- **이전**: `gate_fixed=False`와 `gate_fixed=True`로 두 번 실행 필요
- **현재**: 한 번의 실행으로 동적과 고정 비교 완료
- **run_tag**: `W5-dynamic` / `W5-fixed` → `W5`로 단순화
- **CSV 출력**: 
  - 동적 모델 지표 (rmse, mae, ...)
  - 고정 모델 지표 (rmse_fixed, mae_fixed, ...)
  - W5 비교 지표 (w5_performance_degradation_ratio, ...)

---

### 2025-11-07: W4 실험 코드 개선 (평가 피드백 반영)

#### 1. W4Experiment 개선
- **파일**: `experiments/w4_experiment.py`
- **변경**: 모델로부터 실제 depth를 가져와 하드코딩 방지
- **상세**:
  - `depth = len(model.xhconv_blks)`로 모델 구조와 자동 동기화
  - `self.active_layers` 속성 추가로 BaseExperiment와 정보 공유
  - 설정 파일에 `cnn_depth` 누락/불일치 시에도 안전하게 동작

#### 2. BaseExperiment W4 지원 강화
- **파일**: `experiments/base_experiment.py`
- **변경**: W4 실험 시 중간층 표현 자동 수집 및 중복 로직 제거
- **상세**:
  - `evaluate_test()`에 중간층 표현 수집 훅(hook) 추가
  - 각 활성 층의 CNN/GRU 출력을 자동으로 수집하여 hooks_data에 저장
  - W4 active_layers 중복 계산 제거 (W4Experiment의 속성 재사용)
  - `w4_layerwise_representation_similarity` 지표가 정상적으로 계산됨

#### 3. W4 지표 계산 로직 강화
- **파일**: `utils/experiment_metrics/w4_metrics.py`
- **변경**: 표현 유사도 계산 시 예외 처리 및 안정성 개선
- **상세**:
  - None 체크 추가로 누락된 데이터 graceful handling
  - Distance correlation 계산 실패 시 예외 처리
  - 데이터 구조에 대한 명확한 주석 추가

#### 4. 테스트 및 문서화
- **신규 파일**: 
  - `test_w4_modifications.py`: W4 수정 사항 검증 테스트
  - `W4_MODIFICATIONS_SUMMARY.md`: 상세 수정 내역 및 가이드
- **효과**: 
  - 코드 품질 향상 및 유지보수성 개선
  - 실험 재현성 보장

---

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
