# Lotto Validator Agent — 설계 문서

- 작성일: 2026-04-24
- 작성자: Claude Code (with hyungtae48kim)
- 상태: 승인됨, 구현 플랜 대기

## 1. 목표

최근 2년(약 104회차) 구간을 Walk-Forward 백테스트로 14개 알고리즘 성능을 정량 평가하고, 알고리즘별 약점을 진단해 적중률 향상 대책을 제시하는 검증 서브에이전트를 구축한다.

## 2. 범위

### 2.1 In Scope
- 14개 모든 알고리즘 평가: Stats, GNN, Bayes, LSTM, Transformer, DeepSets, PageRank, Community, Markov, Pattern, MonteCarlo, Weighted Ensemble, Ultimate Ensemble, Stacking Ensemble
- 하이브리드 재학습 전략 (모델별 차등)
- B 메트릭 (예측 기반) + C 메트릭 (확률분포 기반) 병행
- 결과를 `validation_results/<timestamp>/` 디렉토리에 저장
- Claude Code 서브에이전트가 보고서를 해석하여 적중률 향상 대책 제시

### 2.2 Out of Scope
- 제안된 대책의 실제 구현 (별도 작업으로 처리)
- Streamlit UI 통합 (초기 릴리스 이후 고려)
- 실시간 예측 (본 에이전트는 과거 데이터 평가 전용)

## 3. 주요 결정 사항 (브레인스토밍 결과)

| 결정 항목 | 선택 | 근거 |
|-----------|------|------|
| 평가 범위 | 최근 2년 / 약 104회차 / Walk-Forward | 통계적 의미 확보와 실행 시간 균형 |
| 메트릭 수준 | B (예측 기반) + C (확률분포 기반) 병행 | "왜 못 맞추는가"를 진단해야 대책 수립 가능 |
| Agent 형태 | Python 스크립트 + Claude 서브에이전트 조합 | 연산 로직 분리 + 해석/대책 생성 LLM 활용 |
| 재학습 전략 | 모델별 차등 (per_draw / chunk_10) | 공정성과 실행 시간의 균형 |
| 확률적 모델 | 회차당 K=5회 예측 평균 + 확률분포 직접 평가 | 분산 제어와 대책 근거 확보 |
| 결과 저장 | CSV + JSON + Markdown + PNG 차트 + 타임스탬프 디렉토리 | 재현성, 비교 가능성, 사람·기계 모두 읽기 |
| 대책 생성 방식 | 체크리스트 기반 1차 진단 + LLM 자유 해석 하이브리드 | 일관성 + 통찰력 |

## 4. 아키텍처

```
┌─────────────────────────────────────────────────────────┐
│         .claude/agents/lotto-validator.md               │
│         (서브에이전트 정의 - 체크리스트 + 해석 로직)     │
└────────────────────────┬────────────────────────────────┘
                         │ spawn
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  서브에이전트 실행                       │
│  1. run_validation.py 실행 (기존 결과 없거나 오래되면)  │
│  2. validation_results/<latest>/ 읽기                  │
│  3. 체크리스트 기반 진단 + 자유 해석                    │
│  4. 대책 제안 (영향도 × 비용 매트릭스)                  │
└─────────────────────────────────────────────────────────┘
                         │ calls
                         ▼
┌─────────────────────────────────────────────────────────┐
│              validator/ (새 패키지)                      │
│  ┌──────────────────────────────────────────────────┐  │
│  │  run_validation.py         # 엔트리포인트        │  │
│  │  backtest_engine.py        # Walk-Forward 엔진   │  │
│  │  metrics.py                # B+C 메트릭 계산     │  │
│  │  model_registry.py         # 14모델 + 재학습 전략│  │
│  │  report_generator.py       # Markdown + PNG     │  │
│  │  config.py                 # 시드, 기간, K값      │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────┬──────────────────────────────┘
                           │ uses / extends
                           ▼
┌─────────────────────────────────────────────────────────┐
│  기존 자산                                               │
│  - utils/validation.py::WalkForwardValidator (확장)     │
│  - models/base_model.py::get_probability_distribution() │
│  - data/lotto_history.csv                               │
└─────────────────────────────────────────────────────────┘
```

## 5. 컴포넌트 상세

### 5.1 `validator/config.py`

실험 재현성을 위한 설정 모듈.

```python
EVAL_WINDOW_DRAWS = 104          # 최근 2년 (52주 × 2)
RANDOM_SEED = 42                 # numpy, torch, random 통일
PREDICTIONS_PER_DRAW = 5         # 확률적 모델의 회차당 예측 반복 K
REPORT_TTL_HOURS = 24            # 재사용 가능한 결과의 신선도

# 모델별 재학습 전략
MODEL_STRATEGIES = {
    # 가벼운 모델: 매 회차 재학습
    "StatsModel": "per_draw",
    "BayesModel": "per_draw",
    "MarkovModel": "per_draw",
    "PatternModel": "per_draw",
    "PageRankModel": "per_draw",
    "CommunityModel": "per_draw",
    "MonteCarloModel": "per_draw",
    # 중간 비용: 10회차마다 재학습
    "GNNModel": "chunk_10",
    "WeightedEnsembleModel": "chunk_10",
    # 무거운 모델: 10회차마다 재학습
    "LSTMModel": "chunk_10",
    "TransformerModel": "chunk_10",
    "DeepSetsModel": "chunk_10",
    # 메타 앙상블: 10회차마다 재학습
    "UltimateEnsembleModel": "chunk_10",
    "StackingEnsembleModel": "chunk_10",
}
```

### 5.2 `validator/backtest_engine.py`

Walk-Forward 루프의 핵심. `utils/validation.py::WalkForwardValidator`의 기능 확장.

주요 책임:
- 시드 고정 (numpy/torch/random)
- 평가 대상 회차 범위 산정 (최근 104회차)
- 재학습 전략에 따른 학습-예측 루프
- 회차별 K회 예측 + 확률분포 수집
- 체크포인트 저장 (중단 시 재시작 가능: `validation_results/<partial>/checkpoint.pkl`)

핵심 인터페이스 (초안):
```python
class BacktestEngine:
    def __init__(self, df: pd.DataFrame, config: ValidationConfig): ...
    def run(self, model_class, strategy: str) -> BacktestResult: ...
    def run_all(self) -> Dict[str, BacktestResult]: ...
```

`BacktestResult` 데이터클래스는 회차별 `(predictions: List[List[int]], proba: np.ndarray, actual: List[int])` 를 보관.

### 5.3 `validator/metrics.py`

**B 메트릭 (예측 기반)**
- `mean_hits`: K회 평균 적중 수
- `std_hits`: 표준편차
- `hit_distribution`: 적중 개수 0~6개 빈도
- `high_tier_rate`: 5개 이상 적중 비율
- `baseline_comparison`: 무작위 기댓값(0.80) 대비 상대 개선율

**C 메트릭 (확률분포 기반)**
- `top6_prob_sum`: 당첨 6개 번호의 예측 확률 합 (균등 0.133 대비)
- `top10_hits`: 당첨 6개 중 모델 상위 10위 안에 든 개수
- `mean_rank`: 당첨 6개 번호의 모델 예측 평균 순위
- `log_likelihood`: Σ log p(actual_i)
- `zone_bias`: 예측 집중도 편향 (1-10, 11-20, 21-30, 31-40, 41-45 구간별 선택률 vs 실제 당첨 분포)
- `calibration_curve`: 예측 확률 분위수별 실제 적중률 (10분위)

### 5.4 `validator/model_registry.py`

14개 모델의 메타데이터:
- 클래스 참조
- 기본 생성자 인자
- 재학습 전략 (`MODEL_STRATEGIES`에서 조회)
- `get_probability_distribution()` 오버라이드 여부

미구현 모델은 런타임에 감지하여 C 메트릭을 NaN으로 표시하고 보고서에 경고 명시.

### 5.5 `validator/report_generator.py`

산출물:
- `raw_predictions.csv`: 회차·모델·반복번호·예측·적중 수 원시 데이터
- `metrics_summary.json`: 모델별 B+C 메트릭 집계 (머신 판독용)
- `report.md`: 서브에이전트가 읽을 사람용 보고서 (표와 설명)
- `charts/avg_hits_bar.png`: 모델별 평균 적중 막대그래프
- `charts/hit_distribution.png`: 적중 개수 히스토그램 (모델별 중첩)
- `charts/calibration.png`: 예측 확률 vs 실제 적중률 캘리브레이션 곡선
- `config.json`: 실행 당시 설정 스냅샷 (재현용)

### 5.6 `.claude/agents/lotto-validator.md`

서브에이전트 정의 파일. Claude Code 표준 포맷 (frontmatter + system prompt).

**포함 내용**:
- 허용 도구: Bash, Read, Glob, Grep
- 체크리스트 7개 진단 카테고리 내장:
  1. 랜덤 대비 유의미한 우위인가 (Welch's t-test, 기댓값 0.80)
  2. 과적합 진단 (train/test 갭 > 0.3)
  3. 캘리브레이션 품질 (예측 확률과 실제 적중률 일치도)
  4. 번호대 편향 (특정 구간 과집중)
  5. 다양성 결핍 (회차 간 동일 번호 반복)
  6. 앙상블 기여도 (Ultimate/Stacking이 서브모델보다 실제로 나은가)
  7. 가중치 시스템 점검 (`get_advanced_weights()`가 실제 성능 반영 여부)
- 출력 포맷 명세: Top-5 대책을 "영향도 × 구현 비용" 매트릭스로 정렬

## 6. 데이터 흐름

1. 사용자가 이 채팅창에서 "검증 에이전트 돌려줘" 지시
2. 메인 Claude가 `lotto-validator` 서브에이전트 spawn
3. 서브에이전트:
   - `validation_results/` 최신 결과 확인 → `REPORT_TTL_HOURS` 이내면 재사용, 아니면 `python -m validator.run_validation` 실행
   - `metrics_summary.json` + `report.md` 읽음
   - 체크리스트 7개 항목 순회하며 각각 진단
   - Top-5 대책을 "영향도 × 구현 비용" 매트릭스로 정렬
   - 메인에 Markdown 리포트 반환
4. 메인이 사용자에게 요약 전달

## 7. 에러 처리

| 상황 | 처리 |
|------|------|
| 모델 학습 실패 (특정 회차) | 해당 회차 스킵, 로그 기록, 다음 회차로 진행 |
| `get_probability_distribution()` 미구현 | C 메트릭 NaN, B 메트릭만 계속 |
| 백테스트 중단 (Ctrl-C, OOM) | 체크포인트에서 재시작 가능 |
| 데이터 부족 (104회차 미만) | 사용 가능한 최대 구간으로 자동 축소 + 경고 |
| 서브에이전트의 스크립트 실행 타임아웃 | 기존 결과 재사용하거나, 실패를 이 창에 보고 |

## 8. 테스트 전략

- `tests/test_metrics.py`: B+C 메트릭을 수동 케이스 (알려진 예측·당첨 조합) 로 검증
- `tests/test_backtest_smoke.py`: Stats 모델 5회차 짧은 스모크 테스트 (<30초)
- `tests/test_model_registry.py`: 14개 모델 모두 인스턴스화 가능한지
- `tests/test_config.py`: 시드 고정으로 2회 연속 실행 시 결정적 모델은 완전 동일 결과 확인
- 서브에이전트 동작 검증: 첫 실행 시 수동 검수

## 9. 성공 기준

1. `python -m validator.run_validation` 한 줄 실행으로 30–70분 내 완료
2. 동일 시드로 2번 돌리면 결정적 모델은 완전 동일, 확률적 모델도 재현 가능
3. 서브에이전트가 이 채팅창에 구체적·실행 가능한 대책 최소 3개 제시 (막연한 "더 좋은 모델 사용" 금지)
4. 각 대책에 근거 메트릭 수치 인용 (예: "LSTM 과적합 갭 0.42 → dropout 0.2→0.3으로 상향")

## 10. 향후 확장 (본 설계 밖)

- Streamlit "검증/벤치마크" 탭 추가
- 대책의 자동 적용 (예: 가중치 시스템 자동 튜닝)
- 모델 간 예측 상관관계 분석 (앙상블 다양성 측정)
- 실제 배팅 시뮬레이션 (등수별 상금 기대값)
