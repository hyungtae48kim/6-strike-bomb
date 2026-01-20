# 6-Strike-Bomb: Korean Lotto 6/45 Predictor

한국 로또 6/45 번호 예측 시스템입니다. 10개의 AI 모델을 통합한 **Ultimate Ensemble** 시스템을 통해 예측 번호를 생성합니다.

## Features

### Ultimate Ensemble System (v2.0)
10개의 독립적인 AI 모델을 메타 앙상블로 통합하여 예측 정확도를 극대화합니다.

```
                    ┌─────────────────────────────────────┐
                    │      UltimateEnsembleModel          │
                    │   (최종 메타 앙상블 - 확률 통합)     │
                    └──────────────────┬──────────────────┘
                                       │
           ┌───────────────────────────┼───────────────────────────┐
           │                           │                           │
    ┌──────▼──────┐            ┌───────▼───────┐           ┌───────▼───────┐
    │  Tier 1     │            │   Tier 2      │           │   Tier 3      │
    │ 기존 모델   │            │  딥러닝/그래프 │           │  메타 전략    │
    └──────┬──────┘            └───────┬───────┘           └───────┬───────┘
           │                           │                           │
    ┌──────┴──────┐            ┌───────┴───────┐           ┌───────┴───────┐
    │ StatsModel  │            │  LSTMModel    │           │ PatternModel  │
    │ BayesModel  │            │ TransformerM  │           │ MonteCarloM   │
    │ GNNModel    │            │ PageRankModel │           └───────────────┘
    └─────────────┘            │ CommunityM    │
                               │ MarkovModel   │
                               └───────────────┘
```

### 모델 설명

| 카테고리 | 모델 | 설명 |
|----------|------|------|
| **통계 기반** | Stats | Z-score와 softmax 가중치를 사용한 빈도 분석 |
| | Bayes | 조건부 확률 기반 베이지안 추론 |
| **그래프 기반** | GNN | Graph Convolutional Network로 동시 출현 패턴 분석 |
| | PageRank | 번호 간 중요도 랭킹 분석 |
| | Community | Louvain 알고리즘을 통한 클러스터 탐지 |
| **딥러닝** | LSTM | 시계열 패턴 학습 |
| | Transformer | Self-Attention 기반 패턴 인식 |
| **확률/패턴** | Markov | 마르코프 체인 전이 확률 |
| | Pattern | 주기성, 간격, 홀짝 패턴 분석 |
| | Monte Carlo | 시뮬레이션 기반 최적화 |

### 핵심 기능
- **데이터 자동 수집**: 동행복권 API에서 최신 당첨 번호 자동 수집
- **수동 데이터 입력**: API 오류 시 수동 입력 지원
- **예측 피드백 루프**: 과거 예측 결과 기반 알고리즘 가중치 동적 조정
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

### 개별 모델 사용

```python
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from models.pagerank_model import PageRankModel

# LSTM 모델
lstm = LSTMModel(hidden_size=64, num_layers=2)
lstm.train(df)
prediction = lstm.predict()

# Transformer 모델
transformer = TransformerModel(d_model=64, nhead=4)
transformer.train(df)
prediction = transformer.predict()

# PageRank 모델
pagerank = PageRankModel(damping_factor=0.85)
pagerank.train(df)
prediction = pagerank.predict()
```

## Project Structure

```
6-strike-bomb/
├── app.py                      # Streamlit 메인 애플리케이션
├── models/                     # 예측 알고리즘 모델들
│   ├── base_model.py          # 추상 기본 클래스 (LottoModel)
│   ├── enums.py               # AlgorithmType enum 정의
│   ├── stats_model.py         # 통계 기반 모델
│   ├── bayes_model.py         # 베이즈 정리 기반 모델
│   ├── gnn_model.py           # Graph Neural Network 모델
│   ├── lstm_model.py          # LSTM 시계열 모델
│   ├── transformer_model.py   # Transformer 모델
│   ├── pagerank_model.py      # PageRank 그래프 모델
│   ├── community_model.py     # 커뮤니티 탐지 모델
│   ├── markov_model.py        # 마르코프 체인 모델
│   ├── pattern_model.py       # 패턴 분석 모델
│   ├── montecarlo_model.py    # 몬테카를로 시뮬레이션 모델
│   ├── weighted_ensemble_model.py  # 가중치 앙상블 모델
│   └── ultimate_ensemble_model.py  # Ultimate 메타 앙상블 모델
├── utils/                     # 유틸리티 함수들
│   ├── fetcher.py            # 데이터 수집 및 로드
│   └── history_manager.py    # 예측 히스토리 및 가중치 관리
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

## Testing

```bash
# 모든 모델 검증
python verify_ultimate.py

# 개별 검증 스크립트
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
