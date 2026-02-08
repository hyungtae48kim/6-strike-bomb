# 6-Strike-Bomb: Claude Code 작업 가이드

## 프로젝트 개요

한국 로또 6/45 번호 예측 시스템입니다. 12개의 AI 모델을 통합한 메타 앙상블 시스템으로 예측 번호를 생성합니다.

### 핵심 기능
- 로또 당첨 번호 자동 수집 (동행복권 API)
- 14개 알고리즘 지원 (Stats, GNN, Bayes, LSTM, Transformer, DeepSets, PageRank, Community, Markov, Pattern, MonteCarlo, Weighted/Ultimate/Stacking Ensemble)
- 예측 피드백 루프 (과거 예측 결과 기반 알고리즘 가중치 동적 조정)
- 조합 분석 (AC값, 끝수, 합계, 홀짝, 번호대 분포)
- 조합 필터링 (역대 당첨번호 통계 기반 비현실적 조합 제거)
- 조건부 확률 기반 조합 평가 (번호 간 동시출현 상관관계)
- Walk-Forward 시간적 검증 (과적합 탐지)
- 휠링 시스템 (수학적 커버리지 보장)
- 스태킹 앙상블 (Ridge 메타 모델)
- Streamlit 기반 한국어 UI
- 수동 데이터 입력 기능

## 프로젝트 구조

```
6-strike-bomb/
├── app.py                      # Streamlit 메인 애플리케이션
├── models/                     # 예측 알고리즘 모델들
│   ├── base_model.py          # 추상 기본 클래스 (LottoModel)
│   ├── enums.py               # AlgorithmType enum 정의 (14개)
│   ├── stats_model.py         # 통계 기반 모델 (Z-score, softmax)
│   ├── gnn_model.py           # Graph Neural Network 모델
│   ├── bayes_model.py         # 베이즈 정리 기반 모델
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
│   ├── fetcher.py            # 데이터 수집 및 로드 함수
│   ├── history_manager.py    # 예측 히스토리 및 가중치 관리
│   ├── analysis.py           # AC값, 끝수, 합계, 조합 필터
│   ├── combination_scorer.py # 조건부 확률 기반 조합 평가
│   ├── validation.py         # Walk-Forward 시간적 검증
│   ├── wheeling.py           # 수학적 커버리지 휠링 시스템
│   └── meta_learner.py       # BMA 및 교차검증 가중치 학습
├── data/                      # 데이터 파일들
│   ├── lotto_history.csv     # 로또 당첨 번호 히스토리
│   └── prediction_history.csv # 예측 히스토리 및 적중률
├── verify_*.py               # 각종 검증 스크립트들
└── requirements.txt          # Python 의존성
```

## 핵심 컴포넌트 설명

### 1. 모델 아키텍처 (models/)

모든 예측 모델은 [base_model.py](models/base_model.py)의 `LottoModel` 추상 클래스를 상속합니다:

```python
class LottoModel(ABC):
    def train(self, df: pd.DataFrame):  # 데이터로 모델 학습
    def predict(self) -> List[int]:     # 6개 번호 예측 (1-45)
```

#### 알고리즘 유형 ([enums.py](models/enums.py)) - 총 14개

**Tier 1 - 기존 모델:**
- **Stats Based**: Z-score와 softmax 가중치를 사용한 빈도 분석
- **GNN**: 번호 간 동시 출현 패턴 그래프 분석
- **Bayes Theorem**: 조건부 확률 기반 예측
- **Weighted Ensemble**: 과거 적중률 기반 가중치 앙상블

**Tier 2 - 딥러닝:**
- **LSTM**: 시계열 딥러닝 (BCEWithLogitsLoss)
- **Transformer**: Self-Attention 기반 패턴 인식 (BCEWithLogitsLoss)
- **DeepSets**: 순서 불변 집합 인코딩 + GRU 시간적 모델

**Tier 3 - 그래프:**
- **PageRank**: 동시출현 그래프에서 PageRank 중심성 분석
- **Community**: Louvain 알고리즘으로 번호 클러스터 탐지

**Tier 4 - 확률/패턴:**
- **Markov Chain**: 번호 전이 확률 기반 마르코프 체인
- **Pattern Analysis**: 주기성, 간격, 합계 패턴 종합 분석
- **Monte Carlo**: 몬테카를로 시뮬레이션으로 최적 조합 탐색

**앙상블:**
- **Ultimate Ensemble**: 11개 모델 통합 메타 앙상블 + 조합 필터/스코어러
- **Stacking Ensemble**: 7개 모델의 확률 분포를 Ridge 메타 모델로 학습

### 2. 예측 피드백 시스템 ([history_manager.py](utils/history_manager.py))

핵심 기능:
- `save_prediction()`: 예측 결과 저장 ([history_manager.py:23-41](utils/history_manager.py#L23-L41))
- `update_hit_counts()`: 실제 당첨 번호와 비교하여 적중 개수 업데이트 ([history_manager.py:43-86](utils/history_manager.py#L43-L86))
- `get_weights()`: 각 알고리즘의 평균 적중률로 가중치 계산 ([history_manager.py:88-123](utils/history_manager.py#L88-L123))

### 3. 데이터 수집 ([fetcher.py](utils/fetcher.py))

- 동행복권 API에서 자동 수집
- 로컬 CSV 파일로 저장/로드
- 수동 데이터 입력 지원 (API 오류 시)

### 4. 조합 분석 및 필터링 ([analysis.py](utils/analysis.py))

- `LottoAnalyzer`: AC값, 끝수 분포, 합계, 홀짝, 고저, 연번, 번호대 분포, 종합 점수
- `CombinationFilter`: 역대 당첨번호 통계 기준으로 비현실적 조합 제거 (AC값, 합계, 연번, 홀짝, 끝수, 번호대)

### 5. 조건부 확률 기반 조합 평가 ([combination_scorer.py](utils/combination_scorer.py))

- `CombinationScorer`: 번호 간 동시출현 상관관계 행렬(45x45) 구축
- 조건부 확률 기반 번호 선택 (이미 선택된 번호와 상관관계 높은 번호 우선)
- `adjusted_sampling()`: 조건부 확률로 가중된 샘플링

### 6. Walk-Forward 시간적 검증 ([validation.py](utils/validation.py))

- `WalkForwardValidator`: 시간순 분할로 모델의 실제 예측력 측정
- 과적합 탐지 (`detect_overfit()`): 학습-테스트 적중 차이로 과적합 진단

### 7. 휠링 시스템 ([wheeling.py](utils/wheeling.py))

- `WheelingSystem`: 후보 번호에서 수학적 커버리지를 보장하는 조합 세트 생성
- 축약 휠 (Abbreviated Wheel): 그리디 알고리즘으로 최소 티켓 수 산출
- 3/4/5-매치 보장 수준 선택 가능

### 8. 메타 학습기 ([meta_learner.py](utils/meta_learner.py))

- `MetaLearner`: Bayesian Model Averaging (BMA)으로 모델 확률 분포 통합
- softmax 가중치, 캐시 저장/로드 지원

## Claude Code로 작업하는 방법

### 새로운 예측 알고리즘 추가

1. [models/](models/)에 새 파일 생성 (예: `lstm_model.py`)
2. `LottoModel` 상속 및 `train()`, `predict()` 구현
3. [models/enums.py](models/enums.py)에 새 알고리즘 타입 추가
4. [app.py](app.py)의 모델 선택 로직에 추가 ([app.py:112-120](app.py#L112-L120))

예시:
```python
# models/lstm_model.py
from models.base_model import LottoModel

class LSTMModel(LottoModel):
    def train(self, df: pd.DataFrame):
        # LSTM 학습 로직
        pass

    def predict(self) -> List[int]:
        # 예측 로직
        return [1, 2, 3, 4, 5, 6]
```

### UI/UX 개선

Streamlit UI는 [app.py](app.py)에 구현되어 있습니다:
- 메인 UI: [app.py:12-15](app.py#L12-L15)
- 사이드바 설정: [app.py:24-36](app.py#L24-L36)
- 수동 입력 UI: [app.py:38-79](app.py#L38-L79)
- 예측 생성 UI: [app.py:108-143](app.py#L108-L143)

### 데이터 처리 로직 수정

데이터 관련 작업은 [utils/fetcher.py](utils/fetcher.py) 참조:
- API 호출: `fetch_latest_data()`
- 데이터 로드: `load_data()`
- 수동 입력: `add_manual_data()`

### 가중치 알고리즘 조정

가중치 계산 로직은 [utils/history_manager.py:88-123](utils/history_manager.py#L88-L123)에 있습니다.
현재는 평균 적중 개수를 사용하며, 최소값 0.1로 설정되어 있습니다.

## 일반적인 작업 시나리오

### 시나리오 1: 모델 성능 개선
1. 기존 모델 파일 읽기 (예: [models/stats_model.py](models/stats_model.py))
2. `train()` 또는 `predict()` 메서드 수정
3. [verify_script.py](verify_script.py)로 검증

### 시나리오 2: 새 기능 추가
1. 관련 파일 확인 ([app.py](app.py), [models/](models/), [utils/](utils/))
2. 필요한 함수/클래스 구현
3. UI에 통합 ([app.py](app.py))
4. 테스트 실행

### 시나리오 3: 버그 수정
1. 에러 메시지 분석
2. 관련 파일 읽기 및 수정
3. 로컬 테스트: `streamlit run app.py`

### 시나리오 4: 데이터 문제 해결
1. [data/lotto_history.csv](data/lotto_history.csv) 확인
2. [utils/fetcher.py](utils/fetcher.py)의 데이터 수집 로직 검토
3. 필요시 수동 입력 기능 사용

## 중요한 규칙

### 코드 스타일
- 모든 주석과 docstring은 한국어로 작성
- UI 텍스트는 한글/영어 병기 (예: "예측 번호 생성 (Generate Prediction)")
- 함수명은 영어로, 변수명은 명확하게

### 데이터 무결성
- 로또 번호는 항상 1-45 범위
- 예측은 항상 정렬된 6개 고유 번호
- 보너스 번호는 당첨 번호와 중복 불가

### 모델 인터페이스
- 새 모델은 반드시 `LottoModel` 상속
- `train(df)` 메서드로 학습 데이터 받음
- `predict()` 메서드는 `List[int]` 반환 (길이 6)

## 의존성 관리

[requirements.txt](requirements.txt) 파일에 정의된 주요 라이브러리:
- `streamlit`: 웹 UI
- `pandas`: 데이터 처리
- `torch`, `torch_geometric`: GNN 모델
- `networkx`: 그래프 처리
- `numpy`, `scikit-learn`: 수치 계산 및 ML

새 의존성 추가 시 `requirements.txt`에 추가하고 README 업데이트 필요.

## Git 워크플로우

현재 브랜치: main

최근 주요 커밋:
- v3.0 보완: 조합 분석/필터, 조건부 확률 스코어러, Walk-Forward 검증, 휠링, 메타 학습기, DeepSets, 스태킹 앙상블
- LSTM/Transformer BCEWithLogitsLoss 수정
- Ultimate Ensemble에 CombinationScorer/Filter 통합
- 수동 데이터 입력 기능 추가
- 예측 피드백 루프 및 가중치 시스템 도입

## 개발 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
streamlit run app.py
```

## 테스트 및 검증

프로젝트에는 여러 검증 스크립트가 있습니다:
- [verify_enhancements.py](verify_enhancements.py): **전체 보완 사항 검증 (11개 테스트)**
- [verify_script.py](verify_script.py): 기본 모델 검증
- [verify_bayes.py](verify_bayes.py): 베이즈 모델 검증
- [verify_feedback.py](verify_feedback.py): 피드백 시스템 검증
- [verify_manual_update.py](verify_manual_update.py): 수동 입력 검증
- [debug_fetch.py](debug_fetch.py): 데이터 수집 디버깅

## 문제 해결

### API 데이터 수집 실패
- [debug_fetch.py](debug_fetch.py) 실행하여 원인 파악
- 수동 입력 기능 사용 ([app.py:38-79](app.py#L38-L79))

### 모델 학습 오류
- 데이터 형식 확인 (drwNo, drwtNo1-6 컬럼 필수)
- 최소 데이터 개수 확인 (최소 10회차 이상 권장)

### UI 렌더링 문제
- Streamlit 버전 확인
- 브라우저 캐시 삭제
- `streamlit run app.py` 재실행

## Ultimate Ensemble 시스템 (v3.0)

### 개요
12개의 AI 모델을 통합한 메타 앙상블 시스템입니다. v3.0에서 조합 분석/필터링, 조건부 확률 샘플링, 휠링 시스템, Walk-Forward 검증, 스태킹 앙상블이 추가되었습니다.

### 모델 아키텍처
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
    │    서브 모델: Stats, Bayes, PageRank, Community,        │
    │              Markov, Pattern, MonteCarlo                │
    └────────────────────────────────────────────────────────┘
```

### 모델 파일
| 모델 | 파일 | 알고리즘 |
|------|------|----------|
| Stats | [stats_model.py](models/stats_model.py) | Z-score + softmax 빈도 분석 |
| Bayes | [bayes_model.py](models/bayes_model.py) | Beta-Binomial 베이즈 추론 |
| GNN | [gnn_model.py](models/gnn_model.py) | GCN 동시출현 그래프 |
| LSTM | [lstm_model.py](models/lstm_model.py) | 시계열 딥러닝 (BCEWithLogitsLoss) |
| Transformer | [transformer_model.py](models/transformer_model.py) | Self-Attention (BCEWithLogitsLoss) |
| DeepSets | [deepsets_model.py](models/deepsets_model.py) | 순서 불변 집합 인코딩 + GRU |
| PageRank | [pagerank_model.py](models/pagerank_model.py) | 그래프 중심성 |
| Community | [community_model.py](models/community_model.py) | Louvain 클러스터 탐지 |
| Markov | [markov_model.py](models/markov_model.py) | 전이 확률 |
| Pattern | [pattern_model.py](models/pattern_model.py) | 주기성/간격/패턴 분석 |
| MonteCarlo | [montecarlo_model.py](models/montecarlo_model.py) | 시뮬레이션 |
| Ultimate | [ultimate_ensemble_model.py](models/ultimate_ensemble_model.py) | 메타 앙상블 + 조합 최적화 |
| Stacking | [stacking_ensemble_model.py](models/stacking_ensemble_model.py) | Ridge 메타 모델 스태킹 |

### 유틸리티 파일
| 유틸리티 | 파일 | 기능 |
|----------|------|------|
| 조합 분석 | [analysis.py](utils/analysis.py) | AC값, 끝수, 합계, 홀짝, 번호대, 종합 점수, 조합 필터 |
| 조합 스코어러 | [combination_scorer.py](utils/combination_scorer.py) | 동시출현 상관관계, 조건부 확률 샘플링 |
| 시간적 검증 | [validation.py](utils/validation.py) | Walk-Forward 검증, 과적합 탐지 |
| 휠링 | [wheeling.py](utils/wheeling.py) | 수학적 커버리지 보장 축약 휠 |
| 메타 학습 | [meta_learner.py](utils/meta_learner.py) | BMA, softmax 가중치, 캐시 |

### 핵심 기술
1. **확률 분포 기반 통합**: 모든 모델이 45차원 확률 벡터 반환
2. **동적 가중치**: 과거 성능 기반 자동 조정 (지수 감쇠)
3. **다양성 보장**: 모델 간 상관관계 페널티
4. **조합 필터링**: AC값, 합계, 연번, 홀짝, 끝수, 번호대 기반 비현실적 조합 제거
5. **조건부 확률 샘플링**: 번호 간 동시출현 상관관계를 반영한 번호 선택
6. **휠링 시스템**: 후보 번호에서 수학적 매치 보장 (3/4/5-매치)
7. **Walk-Forward 검증**: 시간순 분할로 과적합 탐지
8. **스태킹 앙상블**: Ridge 메타 모델로 서브 모델 확률 분포 학습

### 사용 방법
```python
from models.ultimate_ensemble_model import UltimateEnsembleModel
from models.stacking_ensemble_model import StackingEnsembleModel

# Ultimate Ensemble
model = UltimateEnsembleModel()
model.train(df)
prediction = model.predict()  # 6개 번호
multi = model.predict_multiple(5)  # 5세트
top_nums = model.get_top_numbers(10)  # 상위 10개

# Stacking Ensemble
stacking = StackingEnsembleModel(meta_model_type='ridge')
stacking.train(df)
prediction = stacking.predict()
```

## 라이선스 및 면책

교육 및 엔터테인먼트 목적으로만 사용해야 합니다. 로또 번호는 무작위이며, 이 소프트웨어는 당첨을 보장하지 않습니다.

---

*이 문서는 Claude Code가 프로젝트를 이해하고 효율적으로 작업할 수 있도록 작성되었습니다.*
*파일 경로와 라인 번호는 VSCode에서 클릭 가능한 링크로 제공됩니다.*
