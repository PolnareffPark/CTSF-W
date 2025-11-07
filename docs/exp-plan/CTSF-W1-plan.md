# **W1 실험, 층별 교차 vs. 끝단 융합(후기 결합)** 〔**필수**·재학습〕

### 목적

- **왜 “층마다” 교차해야 하는가?** 끝단에서 한 번만 합치면 왜 떨어지는가를 **직접 검증**합니다. 이는 “층별 교차 보정”이라는 **설계 철학의 핵심 정당성**입니다.
    

### 근거(직접 인용)

- Cross‑stitch(층별 결합): “**At each layer we learn a linear combination of the activation maps** from both the tasks.”(각 층에서 선형결합을 학습) ([CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf?utm_source=chatgpt.com "Cross-Stitch Networks for Multi-Task Learning"))  
    → 층마다 신호를 섞어 **공유표현↔전용표현의 균형**을 **깊이에서** 맞출 때 성능이 좋아짐(동 논문에서 “**… improved performance** over baseline” 요지). ([CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf?utm_source=chatgpt.com "Cross-Stitch Networks for Multi-Task Learning"))
    
- 병렬 경로 교차(입력 의존): “**inserting feature‑dependent cross‑connections** between parallel sets of feature maps … **coefficients are computed from the input features**.”(입력 의존 교차연결) ([arXiv](https://arxiv.org/abs/2006.13904?utm_source=chatgpt.com "Feature-Dependent Cross-Connections in Multi-Path ..."))  
    → **연속 층들 사이**에서 병렬 경로를 **특성 의존적으로** 연결할 때 **정확도 향상** 보고. ([arXiv](https://arxiv.org/abs/2006.13904?utm_source=chatgpt.com "Feature-Dependent Cross-Connections in Multi-Path ..."))
    

### 모델 조작(무엇을 바꾸나)

- **원안(Per‑layer Cross)**: 현재 CTSF(각 블록의 **CrossHyperConvBlock** 활성, 양방향).
    
- **대조(Last‑layer Fusion)**: 모든 블록의 cross를 **OFF**, 마지막 블록에서만 `concat → 1×1 conv(또는 FiLM)`로 결합(파라미터/FLOPs **±3% 이내 매칭**, B1에서 다시 엄밀 통제).
    
- **다른 모든 것 동일(HP2)**.
    

### 평가·통계

- **주지표**: RMSE(실스케일), MSE_std, MAE.
    
- **직접 근거**: TOD 민감도 2종(**문맥→필터/출력**), 피크 반응·시차·방향성 3종(**모양→업데이트/망각**).
    
- **판정**: Per‑layer Cross **>** Last‑layer Fusion (전 지표, 특히 **ETTm2/ETTh2**에서 격차).
    

### 기대 관찰/해석

- 끝단 융합은 **깊이에 따른 보정·상호학습이 불가**, **층별 교차**만이 **수용영역 확대·문맥·모양의 단계별 정렬**을 가능케 함 → 성능 격차.
    

### 보고 형식

- 표: dataset×horizon별 **ΔRMSE(Per‑layer − Last‑layer)**, **TOD 민감도 차이**, **피크·시차 차이**(평균±CI, p).
    
- 그림: **레이더/포리스트 플롯**(데이터셋별 효과크기).
    
## **W1 실험 필수 보완 항목: 파라미터/연산량 통제(공정 비교)** 〔**필수 보완**〕

### 목적

- 성능 이득이 **교차 설계** 때문인지, **용량/연산 증가** 때문인지 분리.
    

### 근거(직접 인용)

- 효율 비교의 함정(“공정 비교” 언급): “**A seemingly fair comparison … compute matched**.”(계산량 매칭) 그러나 모델 축소 등 **비공정** 유발 가능 지적 ([arXiv](https://arxiv.org/pdf/2110.12894?utm_source=chatgpt.com "the efficiency misnomer"))  
    → 본 연구는 역으로, **FLOPs/파라미터 ±3% 이내**로 **양측을 맞춘 대조군**을 두어 **설계 자체의 효과**를 분리.
    

### 모델 조작

- A2/A3 대조군을 **FLOPs·파라미터 매칭**으로 재튜닝(채널·1×1폭 조정).
    
- HP2 동일.
    

### 평가·통계/판정

- ΔRMSE와 직접 지표 **차이값**에 대해 **대응 검정**. **CTSF(both) ≥ 매칭 대조군**이면 **설계 효과**로 귀속.
    
---

# W1 실험 – 층별 교차 vs. 최종 결합 (Late Fusion)

[배경 및 필요성]: CTSF의 핵심 철학 중 하나는 “계층마다 교차로 신호를 섞는 것”입니다. 만약 교차 연결을 각 층에서 하지 않고 맨 마지막 한 번만 두 경로 출력을 합친다면 성능이 어떻게 달라질까요? W1 실험은 “왜 매 층 교차해야 하는가?”에 답하기 위한 것으로, 층별 교차 보정이라는 설계 철학의 핵심 정당성을 직접 검증합니다. 이는 Cross-Stitch Networks 등의 선행 연구에서 층별 공유/전용 표현 균형이 성능 향상에 중요하다고 보고한 것과 맥락을 같이 합니다.
[실험 목적]: Per-layer 교차 연결된 원래 모델과, Last-layer에서 한 번만 결합하는 대조 모델을 비교하여, 계층별 교차의 효과를 정량적으로 확인합니다. 만약 층별 교차가 없다면 성능이 떨어지는지를 입증함으로써, CTSF 구조의 우월성을 보이고자 합니다.
[이론적 근거]: 선행 문헌에서 층별 결합의 중요성을 찾을 수 있습니다: - Cross-Stitch Network (멀티태스크 학습)에서는 “At each layer we learn a linear combination of the activation maps from both the tasks.”[18]라고 하여, 매 층에서 두 경로의 활성화를 섞어 공유 표현과 전용 표현 간 균형을 맞추는 것이 성능 향상의 비결임을 보여줍니다. → 실제로 해당 논문에서 제안한 층별 결합 모델은 baseline 대비 성능 향상을 달성하였으며, 이는 깊이 방향으로 계층별 보정이 일어났기 때문으로 해석됩니다[8]. - Feature-Dependent Cross-Connections (다중 경로 CNN) 연구에서는 “inserting feature‑dependent cross‑connections between parallel sets of feature maps … coefficients are computed from the input features.”[19]라고 하여, 연속된 계층 사이에 특성 의존적 교차연결을 삽입하면 정확도 향상을 보였다고 보고합니다. → 병렬 경로를 입력 상황에 따라 동적으로 연결해 줌으로써, 단순 병렬보다 일관된 성능 개선이 나타난 사례입니다 (동 논문 ICPR 2020).
[모델 조작]: - 원안 (Per-layer Cross): 기본 CTSF 모델 그대로, 모든 블록에 양방향 교차 연결(CrossHyperConv 블록 활성) 사용. - 대조군 (Last-layer Fusion): 각 블록의 교차 연결을 모두 비활성화하고, 맨 마지막에만 두 경로 출력을 결합합니다. 결합 방법은 concat → 1×1 conv 또는 FiLM 등으로 구현하되, 모델 파라미터 수와 FLOPs를 원안과 ±3% 이내로 맞춰 용량 차이를 통제합니다. (해당 용량 매칭은 실험 B1에서 다시 엄밀히 확인) - 그 외 설정: 동일한 하이퍼파라미터 세트 (HP2 세팅)로 학습합니다. 시드, 학습 epoch 등도 동일하게 유지하고, 교차 연결 on/off 이외의 구조 변화는 일절 없음을 강조합니다.
[평가 지표 및 통계]: - 주요 성능 지표: 각 설정에 대해 RMSE (실제값 스케일), MSE_std (표준화 스케일), MAE 등을 산출합니다. 특히 RMSE(real)로 성능 비교. - 직접 근거 지표: 교차 연결이 미치는 영향을 세부적으로 보기 위해 TOD(time-of-day) 민감도 지표 2종과 피크 반응 관련 지표 3종을 추가로 측정합니다. 예컨대 GRU→CNN 경로의 필터 변화 vs 시간대 상관성 (TOD 민감도), CNN→GRU 경로의 피크 이벤트 검출 정도, 최대 상관 및 시차 등을 계산하여, 두 설정 간 컨텍스트 정렬 능력 차이를 분석합니다. - 통계 분석: 동일 데이터셋·Horizon별로 대응 표본 t-검정을 수행하여, Per-layer 모델이 Last-layer 모델 대비 우월한지 여부를 유의수준(p-value)으로 검증합니다. 또한 모든 지표에서 우월한 경향이 있는지 포괄 검정합니다.
•	판정 기준: 기대되는 결과는 Per-layer Cross 모델이 Last-layer Fusion 모델보다 모든 주요 지표에서 더 나은 성능을 보이는 것입니다. 특히 ETTm2/ETTh2와 같이 본 모델 이득이 컸던 데이터셋에서는 성능 격차가 유의하게 크게 나타날 것으로 예상합니다.
[기대되는 관찰 및 해석]: - Last-layer 한 번의 융합만으로는 두 경로의 학습이 부분적으로만 상호작용하므로, 깊이에 따른 단계별 보정이 이루어지지 못합니다. 그 결과 Residual 경로로만 두 신호가 최종 합쳐져 표현력이 제한되고, 오차 누적을 효과적으로 완화하지 못해 성능이 떨어질 것입니다. - 반면 Per-layer 교차 모델은 매 층마다 CNN과 GRU가 서로의 상태를 참조하여 오류를 교정하고 표현 공간을 공유하므로, 수용 영역(receptive field)이 확장되고 문맥 vs 모양 단서가 적절히 정렬되어 최종 성능이 향상됩니다. 이를 성능 격차로 확인할 수 있을 것입니다. - 특히, 교차 연결이 없는 모델은 특정 시간대(예: 야간 vs 주간)별로 성능 편차가 커지거나, 피크 응답의 지연이 발생하는 등 한계가 드러날 것으로 예상합니다. 이러한 패턴 차이를 TOD 민감도와 피크 반응 지표에서 확인하면, 층별 교차의 역할을 뒷받침하는 직접적 증거가 될 것입니다.
[결과 보고 형식]: - 표(Table): 각 데이터셋 × Horizon 조합마다, Per-layer 대비 Last-layer 설정의 ΔRMSE (차이), TOD 민감도 차이, 피크 반응 및 시차 차이를 정리합니다 (평균±신뢰구간, 및 대응 t-검정 p-value 표기). - 그림(Figure): 레이더 차트 또는 포리스트 플롯을 그려서, 데이터셋별로 두 설정 간 효과 크기를 시각화합니다. 예컨대 ETTm2의 경우 거의 모든 지표에서 Per-layer가 우세함을 한 눈에 보여줄 계획입니다.
