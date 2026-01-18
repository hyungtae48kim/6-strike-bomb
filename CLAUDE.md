# 6-Strike-Bomb: Claude Code 작업 가이드

## 프로젝트 개요

한국 로또 6/45 번호 예측 시스템입니다. 통계, GNN(Graph Neural Network), 베이즈 정리, 앙상블 등 다양한 알고리즘을 사용하여 로또 번호를 예측합니다.

### 핵심 기능
- 로또 당첨 번호 자동 수집 (동행복권 API)
- 다중 예측 알고리즘 지원 (Stats, GNN, Bayes, Weighted Ensemble)
- 예측 피드백 루프 (과거 예측 결과 기반 알고리즘 가중치 동적 조정)
- Streamlit 기반 한국어 UI
- 수동 데이터 입력 기능

## 프로젝트 구조

```
6-strike-bomb/
├── app.py                      # Streamlit 메인 애플리케이션
├── models/                     # 예측 알고리즘 모델들
│   ├── base_model.py          # 추상 기본 클래스 (LottoModel)
│   ├── stats_model.py         # 통계 기반 모델 (Z-score, softmax)
│   ├── gnn_model.py           # Graph Neural Network 모델
│   ├── bayes_model.py         # 베이즈 정리 기반 모델
│   ├── weighted_ensemble_model.py  # 가중치 앙상블 모델
│   └── enums.py               # AlgorithmType enum 정의
├── utils/                     # 유틸리티 함수들
│   ├── fetcher.py            # 데이터 수집 및 로드 함수
│   └── history_manager.py    # 예측 히스토리 및 가중치 관리
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

#### 알고리즘 유형 ([enums.py](models/enums.py:3-7))
- **Stats Based**: Z-score와 softmax 가중치를 사용한 빈도 분석
- **GNN**: 번호 간 동시 출현 패턴 그래프 분석
- **Bayes Theorem**: 조건부 확률 기반 예측
- **Weighted Ensemble**: 과거 적중률 기반 가중치 앙상블

### 2. 예측 피드백 시스템 ([history_manager.py](utils/history_manager.py))

핵심 기능:
- `save_prediction()`: 예측 결과 저장 ([history_manager.py:23-41](utils/history_manager.py#L23-L41))
- `update_hit_counts()`: 실제 당첨 번호와 비교하여 적중 개수 업데이트 ([history_manager.py:43-86](utils/history_manager.py#L43-L86))
- `get_weights()`: 각 알고리즘의 평균 적중률로 가중치 계산 ([history_manager.py:88-123](utils/history_manager.py#L88-L123))

### 3. 데이터 수집 ([fetcher.py](utils/fetcher.py))

- 동행복권 API에서 자동 수집
- 로컬 CSV 파일로 저장/로드
- 수동 데이터 입력 지원 (API 오류 시)

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
- 수동 데이터 입력 기능 추가
- 예측 피드백 루프 및 가중치 시스템 도입
- BayesModel 추가
- StatsModel Z-score 및 softmax 적용

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

## Ultimate Ensemble 시스템 (v2.0)

### 개요
10개의 AI 모델을 통합한 메타 앙상블 시스템입니다.

### 모델 아키텍처
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

### 신규 모델 파일
| 모델 | 파일 | 알고리즘 |
|------|------|----------|
| LSTM | [lstm_model.py](models/lstm_model.py) | 시계열 딥러닝 |
| Transformer | [transformer_model.py](models/transformer_model.py) | Self-Attention |
| PageRank | [pagerank_model.py](models/pagerank_model.py) | 그래프 중심성 |
| Community | [community_model.py](models/community_model.py) | 클러스터 탐지 |
| Markov | [markov_model.py](models/markov_model.py) | 전이 확률 |
| Pattern | [pattern_model.py](models/pattern_model.py) | 패턴 분석 |
| MonteCarlo | [montecarlo_model.py](models/montecarlo_model.py) | 시뮬레이션 |
| Ultimate | [ultimate_ensemble_model.py](models/ultimate_ensemble_model.py) | 메타 앙상블 |

### 핵심 기술
1. **확률 분포 기반 통합**: 모든 모델이 45차원 확률 벡터 반환
2. **동적 가중치**: 과거 성능 기반 자동 조정 (지수 감쇠)
3. **다양성 보장**: 모델 간 상관관계 페널티

### 사용 방법
```python
from models.ultimate_ensemble_model import UltimateEnsembleModel

model = UltimateEnsembleModel()
model.train(df)
prediction = model.predict()  # 6개 번호
multi = model.predict_multiple(5)  # 5세트
top_nums = model.get_top_numbers(10)  # 상위 10개
```

## 라이선스 및 면책

교육 및 엔터테인먼트 목적으로만 사용해야 합니다. 로또 번호는 무작위이며, 이 소프트웨어는 당첨을 보장하지 않습니다.

---

*이 문서는 Claude Code가 프로젝트를 이해하고 효율적으로 작업할 수 있도록 작성되었습니다.*
*파일 경로와 라인 번호는 VSCode에서 클릭 가능한 링크로 제공됩니다.*
