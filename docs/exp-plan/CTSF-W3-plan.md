#W3 실험, **데이터 구조 원인 확인(교란 시험)** 〔**선택(권장)**·**소규모** 재학습〕

> CTSF 012의 **Data‑Level Analysis(상관 중심)**를 **원인 수준**으로 보강하는 목적. **ETTm2/ETTh2의 “왜 잘 맞는가”**를 **작게**라도 원인적 증거로 보여줍니다.

### A4‑TOD(ETTm2; 시간대성 교란)

- **목적**: **GRU→CNN(문맥→필터/출력)** 이득이 **일/주기 신호**에 기인함을 확인.
    
- **근거(직접 인용)**: Informer(ETT 소개) “{ETTh1, ETTh2} for 1‑hour‑level and **ETTm1 for 15‑minute‑level**.”(세분 주기) ([AAAI](https://cdn.aaai.org/ojs/17325/17325-13-20819-1-2-20210518.pdf?utm_source=chatgpt.com "Informer: Beyond Efficient Transformer for Long Sequence ...")) ; Nixtla(설명) “**ETTm2 (freq: ‘15T’)** … at a fifteen minute frequency.” ([Nixtlaverse](https://nixtlaverse.nixtla.io/datasetsforecast/long_horizon.html?utm_source=chatgpt.com "Long-Horizon Datasets - Nixtla"))  
    전력수요 계열의 다중 계절성: “**The power load follows … annual, weekly, and daily seasonality**.” ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0306261922017500?utm_source=chatgpt.com "Short-term electricity load forecasting—A systematic ...")) / EIA: “**daily, weekly, and seasonal patterns**.” ([U.S. Energy Information Administration](https://www.eia.gov/todayinenergy/detail.php?id=4190&utm_source=chatgpt.com "Electricity demand changes in predictable patterns"))
    
- **조작**: 학습 시 **일 단위 원형 시프트(day‑roll)·페이즈 지터**(소량, 예: ±1~2h 상하).
    
- **지표/판정**: 조작 시 **TOD 민감도↓**, **gc_only/both의 RMSE 이득↓**면 **문맥→필터 기여의 원인성** 강화.
    

### A4‑PEAK(ETTh2; 급변성 교란)

- **목적**: **CNN→GRU(모양→업데이트/망각)** 이득이 **국소 급변 단서**에서 기인함을 확인.
    
- **근거(데이터 특성 일반)**: ETTh는 1‑hour granularity(급변이 상대적으로 거칠게 드러남) ([AAAI](https://cdn.aaai.org/ojs/17325/17325-13-20819-1-2-20210518.pdf?utm_source=chatgpt.com "Informer: Beyond Efficient Transformer for Long Sequence ...")) ; 전력 계열은 계절·주간 변동 외에도 **피크/저부** 구조가 뚜렷함(일반 전력부하 문헌 다수).
    
- **조작**: 입력을 **경미 평활**(Savitzky–Golay/median, 파라미터 약)하여 **피크/꺾임 단서 약화**.
    
- **지표/판정**: 조작 시 **cg_event_gain↓**, **cg_bestlag(절댓값)↑(느려짐)**, **cg_spearman_mean↓**이면 **모양→게이팅 기여의 원인성** 강화.
    

> **비고**: A4는 **논지 강화용**(필수는 아님). CTSF 012의 상관·연결 분석이 이미 충분하다면 생략 가능하나, **ETTm2/ETTh2 “설계 적합성”**을 한 장으로 못 박는 데 매우 유용합니다.

---

W3 실험 – 데이터 구조에 따른 성능 원인 검증 (교란 시험)
[배경 및 필요성]: 본 실험은 앞선 데이터 분석에서 도출된 CTSF 모델의 데이터 적합성 가설을 원인적으로 확인(causal inference)하고자 하는 것입니다. 즉, “왜 ETTm2/ETTh2에서 성능이 특히 좋았는가?”를 데이터 특성 요소별로 직접 교란하여 확인합니다. 이는 CTSF 012 보고의 Data-Level Analysis가 상관관계 위주였다면, 이번에는 원인-결과 관계를 작게나마 실험적으로 보여주고자 함입니다. 실험 규모는 작지만 설계 적합성을 못박는 증거를 마련하는 것이 목표입니다.
W3은 두 부분으로 나뉩니다: - A4-TOD 실험: ETTm2 데이터의 일별 주기성을 교란하여 GRU→CNN 문맥→필터 경로의 이득이 줄어드는지 관찰. - A4-PEAK 실험: ETTh2 데이터의 국소 피크/급변 신호를 완화하여 CNN→GRU 모양→업데이트 경로의 이득이 감소하는지 관찰.
두 실험 모두 작은 데이터 변형을 가해 성능 변화를 보는 것으로, CTSF의 각 교차 경로가 해당 데이터의 특정 구조에 의존해 이득을 냈음을 검증하려 합니다.
A4-TOD 실험 (ETTm2 – 시간대 패턴 교란)
•	목적: ETTm2의 경우 일중 (하루 96포인트) 및 주중 패턴이 뚜렷하여 GRU 경로의 주기적 문맥을 CNN이 활용한 것이 성능 이득의 원천이라는 가설을 세웠습니다[22]. 이를 검증하기 위해 학습 시 데이터의 시간대 정보를 교란한 후 성능 변화를 관찰합니다. GRU→CNN 경로(문맥→필터/출력)의 효과가 줄어드는지 확인하는 것이 목표입니다.
•	근거 인용: Informer 논문에서 ETT 데이터셋을 소개하며 “{ETTh1, ETTh2} for 1‑hour‑level and ETTm1 for 15‑minute‑level.”[23]라고 명시했고, Nixtla에서도 “ETTm2 (freq: '15T') … at a fifteen minute frequency.”[24]라고 설명하듯 ETTm 시리즈는 세분화된 주기를 갖습니다. 또한 전력 부하 데이터는 “The power load follows … annual, weekly, and daily seasonality.”[25]라고 알려져 있듯 하루 주기의 반복 패턴이 예측에 중요합니다. 이러한 배경에서, CTSF의 GRU→CNN 교차경로가 일/주 주기의 문맥 신호를 필터 조정에 활용했을 가능성이 높습니다.
•	조작 방법: 학습 데이터의 시간대 정보를 인위적으로 교란합니다. 예를 들어 1일 주기 (96포인트) 단위로 시계열을 순환 시프트(roll)하여 날짜와 요일 정보를 어긋나게 하거나, 시간 축을 약간 뒤틀어 페이즈 재배열을 수행합니다. 구체적으로: 일부 배치는 1~2시간 정도 앞으로 당겨지거나 늦춰진 시계열로 학습시킵니다. 이렇게 하면 모델이 학습 시 일정한 시간대 패턴을 잡기 어렵게 됩니다. (또는 더 간단히, 입력 시각 피처를 임의 섞기 등 가능)
•	평가 및 판정: 교란을 가한 모델과 일반 모델의 성능(RMSE)을 비교합니다. 만약 교란으로 인해 CTSF의 문맥→필터 이득이 감소한다면, 다음과 같은 현상이 나타날 것입니다:
•	교차 연결 모드별 성능 격차 감소: 예컨대, 기존에는 both (양방향 교차) > gc_only (GRU→Conv만) > cg_only > none 순으로 성능이 좋았다면, 교란 후에는 그 차이가 줄어듭니다.
•	TOD 민감도 지표 감소: GRU→Conv 경로의 kernel-TOD distance correlation 등이 교란 전보다 유의하게 내려갈 것입니다.
•	전체 RMSE 상승: 특히 gc_only나 both 설정에서 성능 저하폭이 크게 나타나 none 설정과 차이가 좁혀지면, 해당 경로 이득이 줄었다고 해석할 수 있습니다.
•	이러한 변화(↑오차, ↓민감도, ↓이득)를 관찰하면, “ETTm2에서의 성능 이득은 일중 주기 신호를 활용했기 때문”이라는 원인적 주장을 강화할 수 있습니다.
A4-PEAK 실험 (ETTh2 – 급변 패턴 교란)
•	목적: ETTh2 데이터는 1시간 간격이라 급격한 변화(peaks & valleys)가 상대적으로 도드라지게 나타납니다. CNN 경로가 이러한 국소 급변 형태를 감지하여 GRU 상태 업데이트를 도운 것이 성능 이득의 원인이라는 가설을 검증합니다. 즉, CNN→GRU 경로(모양→업데이트)의 기여 원인을 확인합니다.
•	근거: ETTh(ETTh1/2)는 1-hour granularity라 15-min 데이터보다 급변이 덜 매끄럽게 나타납니다[26]. 일반적으로 전력 수요는 계절/주기 외에도 하루 중 피크와 저점을 형성하는 뚜렷한 패턴이 있습니다 (전력 부하 문헌 다수). 이러한 피크 패턴을 CNN이 감지하여 GRU의 망각 게이트 등을 조정함으로써 성능 향상을 이루었을 수 있습니다.
•	조작 방법: 입력 시계열을 약간 평활화하여 급변 단서를 약화시킵니다. 예컨대 Savitzky–Golay 필터나 이동 중앙값 필터를 적용해, 원 데이터의 피크尖 / 꼬임 급변 부분을 완만하게 만듭니다. 중요한 것은 신호의 주요 추세는 보존하되 급작스런 변화 폭을 줄이는 것입니다 (참고: 파라미터는 변화량의 ±10% 정도 감소를 목표로 설정). 이렇게 학습하면 모델은 예측 시 극단 변화 감지 신호가 줄어들게 됩니다.
•	평가 및 판정: 교란 전후 모델의 피크 반응 지표 변화를 확인합니다. 예상되는 결과:
•	cg_event_gain↓: Conv→GRU 경로를 통해 얻는 이벤트 게인 (예: CNN이 감지한 이벤트로 GRU 출력이 개선되는 정도)이 줄어듭니다.
•	cg_bestlag 변화: Conv→GRU 사이의 최대 상관 시차(best lag)의 절대값이 증가(더 느려짐)할 수 있습니다. 이는 원래는 CNN 신호가 발생하고 곧바로 GRU가 반응했는데, 평활화 후에는 반응이 지연될 수 있음을 뜻합니다.
•	cg_spearman_mean↓: Conv 출력과 GRU state 간 순위상관이 낮아집니다, 즉 두 경로의 상호작용 일치도가 떨어집니다.
•	전체 성능: cg_only나 both 설정의 RMSE 이득이 감소하여, baseline과 큰 차이 없거나 일부 반전될 수 있습니다.
•	위 변화들이 관찰된다면, “ETTh2에서 모델 성능 이득은 국소 피크/변화 패턴을 활용한 덕분”이라는 해석을 강하게 뒷받침하게 됩니다.
비고: W3 (A4) 실험은 필수는 아니지만 권장되는 보강 실험입니다. 만약 앞선 W1, W2, W4, W5 결과와 기존 상관분석만으로도 충분히 설득력이 있다면 생략 가능하나, ETTm2/ETTh2에서 왜 설계 적합성이 높은지를 한 장의 그림으로 명확히 보여줄 수 있다는 점에서 유용합니다. 규모가 작고 추가 학습이 필요하므로, 시간적 여유와 자원 상황에 따라 시행을 결정합니다.
