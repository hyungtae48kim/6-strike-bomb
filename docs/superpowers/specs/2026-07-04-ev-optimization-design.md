# EV 최적화 엔진 설계 (Expected-Value Optimization)

- 작성일: 2026-07-04
- 상태: 확정 (구현 대기)

## 배경 및 문제 정의

기존 6-strike-bomb는 12개 AI 모델과 앙상블로 로또 6/45 당첨 번호를
"예측"하는 것을 목표로 했다. 그러나 217건의 실제 예측 기록을 초기하분포
(무작위 기준선)와 통계적으로 비교한 결과:

| 검정 | 결과 | 해석 |
|---|---|---|
| 전체 z-검정 | z=+0.99, p=0.324 | 무작위와 구분 불가 |
| Ultimate Ensemble | z=+0.71, p=0.475 | 무작위와 구분 불가 |
| Stacking Ensemble | z=-0.64, p=0.525 | 무작위와 구분 불가(오히려 아래) |
| 카이제곱 적합도 | χ²=2.90, p=0.408 | 적중 분포 전체가 초기하분포와 완전 일치 |

또한 피드백 루프(과거 적중률 기반 동적 가중치)는 시간이 갈수록 성적을
**악화**시켰다(Ultimate 전반부 0.985 → 후반부 0.680, Stacking 0.950 →
0.526). 독립 사건인 로또에서 과거 적중은 순수 잡음이며, 시스템이 이를
신호로 착각해 노이즈에 과적합한 결과다.

**결론: 로또 6/45는 기억 없는 균등 추첨이므로 당첨 확률을 높이는 예측은
수학적으로 불가능하다.** 이는 코드 품질이 아니라 게임의 본질이다.

## 목표 (재설정)

당첨 확률(`1/8,145,060`)은 손대지 않는다. 대신 **당첨됐을 때의 기대
수령액**을 최적화한다. 로또 1등은 당첨자끼리 상금을 나누는
파리뮤추얼 방식이므로, 통계적으로 **비인기 조합**을 고르면 당첨 시
분배 인원이 줄어 실수령액 기댓값이 올라간다. 이것이 로또에서 학술적으로
검증된 유일한 실제 엣지다.

### EV의 수학적 근거

```
EV(1티켓) = P(당첨) × (1등 상금풀 / 나와 함께 나눌 당첨자 수)
```

- `P(당첨)` = 모든 조합 동일(`1/8,145,060`) → 변경 불가
- `1등 상금풀` = 회차별 거의 고정 → 변경 불가
- **유일 변수 = 기대 동반당첨자 수 = 인기도** → 비인기일수록 작아짐

따라서 `EV ∝ 1 / (1 + 기대 동반당첨자수)`이고, **EV 최대화 = 인기도
최소화**이다.

## 비목표 (Non-goals)

- 당첨 확률/적중률 개선 (수학적으로 불가능 — 검증 완료)
- 기존 12개 예측 모델(models/), history_manager, 예측 파이프라인 수정
- 조합별 판매량 직접 확보(한국 동행복권 미공개)

## 아키텍처

### 모듈 구조

```
utils/popularity_model.py   # PopularityModel: 휴리스틱 factor 점수 + 보정 가능한 가중치
utils/ev_optimizer.py       # EVOptimizer: 비인기 조합 생성 + 다양성 + EV 배수 산출
utils/winner_data.py        # 1등 당첨자 수/판매액 수집(동행복권 API) → CSV
data/winner_history.csv     # 회차별 firstPrzwnerCo(1등 당첨자수), 인당상금, 총판매액
verify_ev.py                # 검증: 휴리스틱 점수 vs 실제 1등 당첨자 수 회귀
app.py                      # 'EV 최적 조합' 탭 추가 (기존 파일 확장)
```

기존 예측 파이프라인(models/, utils/history_manager.py 등)은 건드리지
않는다. EV 플로우는 완전히 독립된 계층이다.

동행복권 API(`fetcher.py`가 이미 사용)는 회차별로 `firstPrzwnerCo`
(1등 당첨자 수)와 `totSellamnt`(총판매액), `firstWinamnt`(1등 인당
상금)를 제공하므로 검증 데이터는 동일 API로 수집한다.

## 컴포넌트 상세

### 1. PopularityModel (utils/popularity_model.py)

조합의 "인기도 점수"(`∈ [0,1]`, 높을수록 많은 사람이 고를 조합)를 산출.
각 factor는 0~1 penalty를 반환하고 가중합(가중치는 접근법 C에서 데이터로
보정 가능)한다.

**Factor 목록 (연구로 알려진 인간 번호선택 편향):**

1. **생일 편중(calendar_bias)** — 번호 ≤31 비율(특히 ≤12는 월 편향으로
   추가 가중). 낮은 번호가 많을수록 인기↑
2. **낮은 합계(low_sum)** — 생일 기반 조합은 합계가 낮은 경향
3. **연속수(consecutive)** — 인접 연속 쌍 개수(1-2-3 등)
4. **등차·배수 패턴(arithmetic)** — 등차수열, 5·10·15… 배수 패턴
5. **용지 격자 패턴(grid_pattern)** — 실제 마킹용지 배열상의 직선/대각선/
   모서리 등 기하 패턴
6. **행운수/유명조합(lucky)** — 7 편애, 1-2-3-4-5-6 등 유명 novelty 조합
7. **최근 당첨번호 재사용(recent_reuse)** — 직전 N회차 당첨번호를 다수
   포함 시 인기↑ (사람들이 최근 당첨번호를 재구매)

인터페이스:
```python
class PopularityModel:
    def __init__(self, df: pd.DataFrame, weights: dict = None): ...
    def factor_scores(self, combo: List[int]) -> Dict[str, float]:  # factor별 0~1
    def popularity(self, combo: List[int]) -> float:                # 가중합 0~1
    def reasons(self, combo: List[int]) -> List[str]:               # 사람이 읽는 근거 태그
    def set_weights(self, weights: dict): ...                       # 데이터 보정 결과 주입
```

기존 `CombinationFilter`(역대 통계 기반 "현실적" 조합 필터)는 **의도적으로
사용하지 않는다.** factor 6이 유명 패턴을 이미 처리하며, 현실성 필터는
오히려 인기 조합 쪽으로 결과를 몰 수 있기 때문이다.

### 2. EVOptimizer (utils/ev_optimizer.py)

```python
class EVOptimizer:
    def __init__(self, popularity_model: PopularityModel): ...
    def ev_index(self, combo: List[int]) -> float:
        # 인기도 → 평균 조합 대비 기대 수령액 배수(보정된 인기도→기대당첨자수 매핑)
    def generate(self, n_tickets: int, aggressiveness: float = 1.0) -> List[dict]:
        # 1) 대량 후보 샘플링(고번호 가중)
        # 2) 인기도 최소 조합 선별
        # 3) 그리디 다양성 선택으로 겹침 적은 N세트
        # 반환: [{combo, popularity, ev_index, reasons}, ...]
```

- 후보 생성: `C(45,6)=8.1M` 전수는 비현실적 → 대량 랜덤 샘플링(고번호 쪽
  가중) 후 인기도로 정렬.
- 다양성: 선택된 티켓 간 번호 겹침을 최소화하는 그리디 선택.
- `aggressiveness`: 비인기 강도(1.0=최상위 비인기, 낮추면 다양성/자연스러움
  가미).

### 3. winner_data (utils/winner_data.py)

- 동행복권 API에서 회차별 `firstPrzwnerCo`, `firstWinamnt`, `totSellamnt`
  수집 → `data/winner_history.csv` 저장/증분 업데이트.
- `fetcher.py`의 API 호출 패턴을 재사용.

### 4. 검증 (verify_ev.py)

- 각 회차 당첨조합의 휴리스틱 인기도 점수 계산.
- 회귀: `1등당첨자수 ~ 인기도점수 + log(총판매액)`(판매량 통제).
- 보고: 계수, p값, R². **양(+)의 유의한 계수 = 휴리스틱이 실제 인기도를
  포착함을 반증 가능하게 입증.**
- 같은 데이터로 factor별 가중치를 Ridge 회귀 보정 → `PopularityModel`에
  주입(접근법 C 하이브리드).
- 검증 실패(계수 비유의/음수) 시에도 결과를 정직하게 보고하고, 휴리스틱
  기본 가중치를 유지한다.

### 5. UI 탭 (app.py 확장)

- 입력: 티켓 수 N, 공격성 슬라이더.
- 출력: 조합 표(번호 + 인기도 점수 + EV 지수 + 근거 태그).
- **명시 문구**: "당첨 확률은 무작위와 동일하며, 바뀌는 것은 당첨 시
  분배액(기대 수령액)뿐입니다."
- 검증 결과 요약(회귀 계수/p값/R²) 표시로 근거 투명화.

## 데이터 흐름

```
동행복권 API ──> winner_data ──> data/winner_history.csv
                                        │
data/lotto_history.csv ──> PopularityModel(factor 가중치) <── verify_ev(Ridge 보정)
                                        │
                                  EVOptimizer(생성+EV지수+다양성)
                                        │
                                   app.py EV 탭 (표 + 근거 + 검증요약)
```

## 오류 처리

- API 수집 실패: 기존 캐시 CSV 사용, 없으면 검증 스킵하고 기본 가중치로
  엔진만 동작(UI에 "검증 데이터 없음" 표시).
- 검증 데이터 부족(회차 수 적음): 회귀 생략, 휴리스틱 기본 가중치 유지.
- 잘못된 조합(중복/범위 밖): 생성 단계에서 배제.

## 테스트

- **factor 단위 테스트**: 알려진 조합으로 방향성 검증
  - `1,2,3,4,5,6` → 인기도 매우 높음(연속+유명조합)
  - `2,7,12,19,24,31` → 생일/낮은번호 편중 → 인기 높음
  - `13,29,34,38,41,45` 등 고분산·고번호 → 인기 낮음
- **EV 단조성**: 인기도 낮을수록 EV 지수 높음.
- **다양성 선택**: N세트 간 겹침이 임계 이하.
- **통합 검증**: verify_ev.py를 회귀 검증 스크립트로 사용.

## 정직성 원칙 (프로젝트 전반)

- 모든 산출물에 "당첨 확률은 개선되지 않음"을 명시한다.
- 검증 결과는 유리하든 불리하든 그대로 보고한다.
- EV 지수는 "평균 조합 대비 상대 기대 수령액"으로만 제시하고, 절대 금액
  보장을 하지 않는다.

## 향후 확장(이번 범위 아님)

- 휠링 시스템(utils/wheeling.py)과 결합해 비인기 후보 풀 기반 커버리지 세트.
- factor 추가/재보정을 회차 누적에 따라 주기적으로 수행.
