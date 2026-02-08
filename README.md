# 6-Strike-Bomb: Korean Lotto 6/45 Predictor

한국 로또 6/45 번호 예측 시스템입니다. 12개의 AI 모델을 통합한 **Ultimate Ensemble** 시스템 + 조합 최적화 파이프라인으로 예측 번호를 생성합니다.

## Features

### Ultimate Ensemble System (v3.0)
12개의 독립적인 AI 모델을 메타 앙상블로 통합하고, 조합 분석/필터링/조건부 확률 샘플링으로 예측 품질을 극대화합니다.

```
    ┌────────────────────────────────────────────────────────┐
    │                UltimateEnsembleModel                    │
    │    (메타 앙상블 + CombinationScorer + CombinationFilter)│
    └──────────────────────────┬─────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
  ┌──────▼──────┐      ┌──────▼──────┐      ┌───────▼───────┐
  │  Tier 1     │      │  Tier 2     │      │   Tier 3      │
  │ 기존 모델   │      │ 딥러닝/그래프│      │  메타 전략    │
  └──────┬──────┘      └──────┬──────┘      └───────┬───────┘
         │                    │                     │
  ┌──────┴──────┐     ┌──────┴──────┐       ┌──────┴───────┐
  │ StatsModel  │     │ LSTMModel   │       │ PatternModel │
  │ BayesModel  │     │ Transformer │       │ MonteCarloM  │
  │ GNNModel    │     │ DeepSetsM   │       └──────────────┘
  └─────────────┘     │ PageRankM   │
                      │ CommunityM  │
                      │ MarkovModel │
                      └─────────────┘

    ┌────────────────────────────────────────────────────────┐
    │             StackingEnsembleModel                       │
    │    (Ridge 메타 모델 + 시간적 분할 교차검증)              │
    └────────────────────────────────────────────────────────┘
```

### 모델 설명

| 카테고리 | 모델 | 설명 |
|----------|------|------|
| **통계 기반** | Stats | Z-score와 softmax 가중치를 사용한 빈도 분석 |
| | Bayes | Beta-Binomial 켤레 사전분포 기반 베이지안 추론 |
| **그래프 기반** | GNN | Graph Convolutional Network로 동시 출현 패턴 분석 |
| | PageRank | 번호 간 중요도 랭킹 분석 |
| | Community | Louvain 알고리즘을 통한 클러스터 탐지 |
| **딥러닝** | LSTM | 시계열 패턴 학습 (BCEWithLogitsLoss) |
| | Transformer | Self-Attention 기반 패턴 인식 (BCEWithLogitsLoss) |
| | DeepSets | 순서 불변 집합 인코딩 + GRU 시간적 모델 |
| **확률/패턴** | Markov | 마르코프 체인 전이 확률 |
| | Pattern | 주기성, 간격, 홀짝 패턴 분석 |
| | Monte Carlo | 시뮬레이션 기반 최적화 |
| **앙상블** | Ultimate | 11개 모델 통합 메타 앙상블 + 조합 최적화 |
| | Stacking | 7개 모델의 확률 분포를 Ridge 메타 모델로 학습 |

### 핵심 기능
- **데이터 자동 수집**: 동행복권 API에서 최신 당첨 번호 자동 수집
- **수동 데이터 입력**: API 오류 시 수동 입력 지원
- **14개 알고리즘**: Stats, GNN, Bayes, LSTM, Transformer, DeepSets, PageRank, Community, Markov, Pattern, MonteCarlo, Weighted/Ultimate/Stacking Ensemble
- **예측 피드백 루프**: 과거 예측 결과 기반 알고리즘 가중치 동적 조정
- **조합 분석**: AC값, 끝수 분포, 합계, 홀짝, 고저, 연번, 번호대 분포, 종합 점수
- **조합 필터링**: 역대 당첨번호 통계 기반 비현실적 조합 자동 제거
- **조건부 확률 샘플링**: 번호 간 동시출현 상관관계를 반영한 번호 선택
- **휠링 시스템**: 후보 번호에서 수학적 커버리지 보장 (3/4/5-매치)
- **Walk-Forward 검증**: 시간순 분할로 모델 과적합 탐지
- **다중 예측 생성**: 한 번에 여러 세트의 예측 번호 생성
- **확률 분포 시각화**: 각 번호별 예측 확률 표시

## Installation

### Prerequisites
- Python 3.8+
- (선택) CUDA GPU (딥러닝 모델 가속)

### Setup

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt
```

### Dependencies
- `streamlit`: 웹 UI 프레임워크
- `pandas`, `numpy`: 데이터 처리
- `torch`, `torch_geometric`: PyTorch 및 GNN
- `networkx`: 그래프 분석
- `scikit-learn`: 머신러닝 유틸리티
- `scipy`: 과학 계산
- `python-louvain`: 커뮤니티 탐지

## Usage

### 웹 애플리케이션 실행

```bash
streamlit run app.py
```

브라우저에서 자동으로 UI가 열립니다.

### 프로그래밍 방식 사용

```python
from models.ultimate_ensemble_model import UltimateEnsembleModel
from utils.fetcher import load_data

# 데이터 로드
df = load_data()

# Ultimate Ensemble 모델 생성 및 학습
model = UltimateEnsembleModel()
model.train(df)

# 단일 예측 (6개 번호)
prediction = model.predict()
print(f"예측 번호: {prediction}")

# 다중 예측 (5세트)
predictions = model.predict_multiple(5)
for i, pred in enumerate(predictions, 1):
    print(f"세트 {i}: {pred}")

# 상위 10개 유망 번호
top_numbers = model.get_top_numbers(10)
print(f"상위 10개: {top_numbers}")

# 전체 확률 분포
probs = model.get_probability_distribution()
# probs[i-1] = 번호 i의 예측 확률
```

### 스태킹 앙상블 사용

```python
from models.stacking_ensemble_model import StackingEnsembleModel

# Ridge 메타 모델 스태킹 앙상블
model = StackingEnsembleModel(meta_model_type='ridge')
model.train(df)
prediction = model.predict()
multi = model.predict_multiple(5)
```

### 개별 모델 사용

```python
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.deepsets_model import DeepSetsModel

# LSTM 모델 (BCEWithLogitsLoss)
lstm = LSTMModel(hidden_size=64, num_layers=2, epochs=50)
lstm.train(df)
prediction = lstm.predict()

# Transformer 모델 (BCEWithLogitsLoss)
transformer = TransformerModel(d_model=64, nhead=4, epochs=50)
transformer.train(df)
prediction = transformer.predict()

# DeepSets 모델 (순서 불변)
deepsets = DeepSetsModel(epochs=50)
deepsets.train(df)
prediction = deepsets.predict()
```

### 조합 분석

```python
from utils.analysis import LottoAnalyzer, CombinationFilter

analyzer = LottoAnalyzer()
combo = [3, 15, 22, 31, 38, 44]

# 개별 지표 분석
print(f"AC값: {analyzer.ac_value(combo)}")          # 7-10이 이상적
print(f"합계: {analyzer.sum_value(combo)}")          # ~135 근처가 이상적
print(f"홀짝: {analyzer.odd_even_ratio(combo)}")     # 2:4~4:2가 이상적
print(f"종합 점수: {analyzer.comprehensive_score(combo)}")

# 조합 필터링
cf = CombinationFilter(df)
print(f"필터 통과: {cf.filter(combo)}")
```

### 휠링 시스템

```python
from utils.wheeling import WheelingSystem

# 상위 10개 후보 번호에서 5등 보장 휠 생성
numbers = [3, 7, 12, 18, 25, 31, 37, 42, 44, 45]
ws = WheelingSystem(numbers, guarantee_match=3)
wheel = ws.generate_abbreviated_wheel()

report = ws.get_coverage_report(wheel)
print(f"티켓 수: {report['총_티켓_수']}장")
print(f"커버리지: {report['커버리지']}")
```

### Walk-Forward 검증

```python
from utils.validation import WalkForwardValidator
from models.stats_model import StatsModel

validator = WalkForwardValidator(initial_train_size=500, test_size=5, step_size=100)
result = validator.validate(StatsModel, df)

print(f"평균 적중: {result.avg_hits:.3f}")
print(f"과적합 갭: {result.overfit_gap:.3f}")

overfit = WalkForwardValidator.detect_overfit(result)
print(f"과적합 진단: {overfit['severity']}")
```

## Project Structure

```
6-strike-bomb/
├── app.py                      # Streamlit 메인 애플리케이션
├── models/                     # 예측 알고리즘 모델들 (14개 알고리즘)
│   ├── base_model.py          # 추상 기본 클래스 (LottoModel)
│   ├── enums.py               # AlgorithmType enum 정의 (14개)
│   ├── stats_model.py         # 통계 기반 모델 (Z-score, softmax)
│   ├── bayes_model.py         # 베이즈 정리 기반 모델
│   ├── gnn_model.py           # Graph Neural Network 모델
│   ├── lstm_model.py          # LSTM 시계열 모델 (BCEWithLogitsLoss)
│   ├── transformer_model.py   # Transformer 모델 (BCEWithLogitsLoss)
│   ├── deepsets_model.py      # DeepSets+GRU 순서 불변 모델
│   ├── pagerank_model.py      # PageRank 그래프 모델
│   ├── community_model.py     # 커뮤니티 탐지 모델
│   ├── markov_model.py        # 마르코프 체인 모델
│   ├── pattern_model.py       # 패턴 분석 모델
│   ├── montecarlo_model.py    # 몬테카를로 시뮬레이션 모델
│   ├── weighted_ensemble_model.py    # 가중치 앙상블 모델
│   ├── ultimate_ensemble_model.py    # Ultimate 메타 앙상블 모델
│   └── stacking_ensemble_model.py    # 스태킹 앙상블 모델 (Ridge 메타)
├── utils/                     # 유틸리티 함수들
│   ├── fetcher.py            # 데이터 수집 및 로드
│   ├── history_manager.py    # 예측 히스토리 및 가중치 관리
│   ├── analysis.py           # AC값, 끝수, 합계, 조합 필터
│   ├── combination_scorer.py # 조건부 확률 기반 조합 평가
│   ├── validation.py         # Walk-Forward 시간적 검증
│   ├── wheeling.py           # 수학적 커버리지 휠링 시스템
│   └── meta_learner.py       # BMA 및 교차검증 가중치 학습
├── data/                      # 데이터 파일들
│   ├── lotto_history.csv     # 로또 당첨 번호 히스토리
│   └── prediction_history.csv # 예측 히스토리
├── verify_*.py               # 검증 스크립트들
├── requirements.txt          # Python 의존성
├── CLAUDE.md                 # Claude Code 작업 가이드
└── README.md                 # 이 파일
```

## Algorithm Details

### 확률 분포 기반 통합
모든 모델은 45차원 확률 벡터를 반환합니다. Ultimate Ensemble은 이를 가중 평균으로 통합합니다:

```
P_final(n) = Σ(w_i × P_i(n)) / Σ(w_i)
```

여기서:
- `P_i(n)`: 모델 i가 번호 n에 부여한 확률
- `w_i`: 모델 i의 가중치 (과거 성능 기반)

### 동적 가중치 시스템
각 모델의 가중치는 과거 예측 적중률을 기반으로 동적으로 조정됩니다:
- 지수 감쇠 (exponential decay): 최근 예측에 더 높은 가중치
- 추세 보너스: 성능이 향상 중인 모델에 추가 가중치
- 일관성 보너스: 안정적인 성능의 모델에 추가 가중치

### 다양성 보너스
모델 간 상관관계가 낮은 예측을 장려하여 과적합을 방지합니다.

### 조합 분석 및 필터링 (v3.0)
예측 결과에 후처리 파이프라인을 적용하여 품질을 높입니다:
- **AC값 (Arithmetic Complexity)**: 번호 쌍 차이값의 다양성 측정 (7-10이 이상적)
- **합계 범위**: 역대 평균 ~135, 표준편차 ±1.5σ 이내
- **홀짝/고저 비율**: 극단적 편향 제거
- **연번 제한**: 4쌍 이상 연번 제거
- **끝수/번호대 분포**: 과도한 집중 제거

### 조건부 확률 샘플링 (v3.0)
번호 간 동시출현 상관관계 행렬(45×45)을 구축하여, 이미 선택된 번호와 상관관계가 높은 번호를 우선 선택합니다.

### 휠링 시스템 (v3.0)
상위 N개 후보 번호에서 수학적 매치 보장을 위한 최소 티켓 세트를 생성합니다:
- **축약 휠 (Abbreviated Wheel)**: 그리디 알고리즘으로 95%+ 티켓 절감
- **3/4/5-매치 보장**: 후보 중 당첨번호가 있으면 해당 등급 보장

### 스태킹 앙상블 (v3.0)
7개 서브 모델(Stats, Bayes, PageRank, Community, Markov, Pattern, MonteCarlo)의 확률 분포를 메타 특성으로 구축하고, Ridge 회귀 메타 모델로 학습합니다. 시간적 분할 교차검증으로 과적합을 방지합니다.

### Walk-Forward 검증 (v3.0)
시간순 분할로 모델의 실제 예측력을 측정하고, 학습-테스트 적중 차이로 과적합 여부를 진단합니다.

## Testing

```bash
# 전체 보완 사항 검증 (11개 테스트)
python verify_enhancements.py

# 기존 검증 스크립트
python verify_ultimate.py      # Ultimate Ensemble
python verify_script.py        # 기본 모델
python verify_bayes.py         # 베이즈 모델
python verify_feedback.py      # 피드백 시스템
python debug_fetch.py          # 데이터 수집 디버깅
```

## Optimal Number of Predictions (k 값)

### 확률 분석

로또 1등 당첨 확률:
- 단일 조합: **1 / 8,145,060 ≈ 0.0000123%**
- k개 조합: **k / 8,145,060**

| k (구매 횟수) | 당첨 확률 | 기대값 (1등 20억 기준) | 비용 (1,000원/장) |
|--------------|----------|---------------------|-----------------|
| 1 | 0.0000123% | 245원 | 1,000원 |
| 5 | 0.0000614% | 1,228원 | 5,000원 |
| 10 | 0.000123% | 2,455원 | 10,000원 |
| 100 | 0.00123% | 24,554원 | 100,000원 |
| 1,000 | 0.0123% | 245,544원 | 1,000,000원 |

### 권장 k 값

**실용적 권장: k = 5 ~ 10**

근거:
1. **다양성 확보**: Ultimate Ensemble의 다양한 모델들이 서로 다른 관점의 예측을 제공
2. **비용 대비 효율**: 주당 5,000~10,000원은 합리적인 오락 비용
3. **기대값 한계**: k를 아무리 높여도 기대값 < 비용 (모든 k에서 손실)

### 코드 예시

```python
# 최적 k 값으로 예측 생성
model = UltimateEnsembleModel()
model.train(df)

# 5세트 생성 (권장)
predictions = model.predict_multiple(k=5)
for i, pred in enumerate(predictions, 1):
    print(f"조합 {i}: {sorted(pred)}")
```

## Disclaimer

이 소프트웨어는 **교육 및 엔터테인먼트 목적**으로만 사용해야 합니다.

- 로또 번호는 **완전한 무작위**입니다
- 어떤 알고리즘도 당첨 확률을 수학적으로 높일 수 없습니다
- 이 소프트웨어는 **당첨을 보장하지 않습니다**
- 책임감 있는 도박을 실천하세요

## License

MIT License

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

---

*Developed with Claude Code assistance*
