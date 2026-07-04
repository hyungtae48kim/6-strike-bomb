# EV 최적화 엔진 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 당첨 확률은 그대로 두고, 비인기 조합을 골라 당첨 시 기대 수령액(EV)을 최대화하는 엔진 + 데이터 검증 + Streamlit UI를 추가한다.

**Architecture:** 인간 번호선택 편향을 정량화하는 `PopularityModel`(휴리스틱 factor 가중합)을 만들고, `EVOptimizer`가 비인기 조합을 생성·다양성 선택·EV 지수를 산출한다. 동행복권 API의 1등 당첨자 수(`winner_data`)로 휴리스틱을 회귀 검증하고 factor 가중치를 보정한다(`verify_ev.py`). 결과는 app.py의 새 섹션에서 표시한다. 기존 예측 파이프라인(models/, history_manager)은 건드리지 않는다.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn(Ridge), pytest, Streamlit. 테스트 실행은 `venv/bin/pytest`.

**참고 설계 문서:** `docs/superpowers/specs/2026-07-04-ev-optimization-design.md`

---

## 파일 구조

- `utils/popularity_model.py` (신규) — `PopularityModel`: factor 점수 + 가중 인기도 + 근거 태그
- `utils/ev_optimizer.py` (신규) — `EVOptimizer`: EV 지수 + 조합 생성/다양성, `build_ev_rows` 헬퍼
- `utils/winner_data.py` (신규) — 1등 당첨자 수/판매액 수집·저장
- `data/winner_history.csv` (런타임 생성) — 회차별 검증 데이터
- `verify_ev.py` (신규) — 인기도 vs 1등 당첨자 수 회귀 검증 + 가중치 보정
- `app.py` (수정) — 'EV 최적 조합' 섹션 추가
- `tests/test_popularity_model.py` (신규)
- `tests/test_ev_optimizer.py` (신규)
- `tests/test_winner_data.py` (신규)
- `tests/test_verify_ev.py` (신규)

---

## Task 1: PopularityModel — factor 점수 + 가중 인기도

**Files:**
- Create: `utils/popularity_model.py`
- Test: `tests/test_popularity_model.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_popularity_model.py`:
```python
import pandas as pd
import pytest
from utils.popularity_model import PopularityModel, DEFAULT_WEIGHTS

# 인기 조합: 연속 + 전부 생일범위 + 유명조합
POPULAR = [1, 2, 3, 4, 5, 6]
# 비인기 조합: 고번호 다수 + 비패턴 + 분산
UNPOPULAR = [5, 17, 26, 33, 38, 44]


def test_factor_scores_all_in_range():
    pm = PopularityModel()
    fs = pm.factor_scores(POPULAR)
    assert set(fs.keys()) == set(DEFAULT_WEIGHTS.keys())
    assert all(0.0 <= v <= 1.0 for v in fs.values())


def test_famous_combo_is_highly_popular():
    pm = PopularityModel()
    assert pm.popularity(POPULAR) > 0.7


def test_dispersed_combo_is_unpopular():
    pm = PopularityModel()
    assert pm.popularity(UNPOPULAR) < 0.35


def test_popular_greater_than_unpopular():
    pm = PopularityModel()
    assert pm.popularity(POPULAR) > pm.popularity(UNPOPULAR)


def test_calendar_bias_high_for_low_numbers():
    pm = PopularityModel()
    assert pm.factor_scores([1, 2, 3, 4, 5, 6])["calendar_bias"] == pytest.approx(1.0)
    assert pm.factor_scores([32, 35, 38, 41, 44, 45])["calendar_bias"] == pytest.approx(0.0)


def test_recent_reuse_uses_recent_draws():
    df = pd.DataFrame([{
        "drwNo": 1, "drwtNo1": 10, "drwtNo2": 11, "drwtNo3": 12,
        "drwtNo4": 13, "drwtNo5": 14, "drwtNo6": 15, "bnusNo": 40,
    }])
    pm = PopularityModel(df, recent_window=1)
    assert pm.factor_scores([10, 11, 12, 13, 14, 15])["recent_reuse"] == pytest.approx(1.0)
    assert pm.factor_scores([20, 21, 22, 23, 24, 25])["recent_reuse"] == pytest.approx(0.0)


def test_reasons_returns_tags_for_high_factors():
    pm = PopularityModel()
    reasons = pm.reasons(POPULAR)
    assert "연속 번호" in reasons
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `venv/bin/pytest tests/test_popularity_model.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.popularity_model'`

- [ ] **Step 3: 최소 구현 작성**

`utils/popularity_model.py`:
```python
"""인기도(popularity) 휴리스틱 모델.

각 조합이 '얼마나 많은 사람이 고를 조합인가'를 0~1 점수로 추정한다.
높을수록 인기(당첨 시 분배 인원 많음 = EV 낮음).
알려진 인간 번호선택 편향을 factor로 정량화하고 가중합한다.
가중치는 verify_ev.py의 1등 당첨자 수 회귀로 보정할 수 있다.
"""
from typing import List, Dict
import pandas as pd

# 실제 마킹용지 배열: 7열 그리드 (번호 1~45)
GRID_COLS = 7

# 유명 novelty 조합 (사람들이 재미로 자주 고름)
FAMOUS_COMBOS = [
    (1, 2, 3, 4, 5, 6),
    (40, 41, 42, 43, 44, 45),
]

DEFAULT_WEIGHTS = {
    "calendar_bias": 0.28,
    "low_sum": 0.12,
    "consecutive": 0.12,
    "arithmetic": 0.13,
    "grid_pattern": 0.15,
    "lucky": 0.08,
    "recent_reuse": 0.12,
}

FACTOR_LABELS = {
    "calendar_bias": "생일/낮은 번호 편중",
    "low_sum": "낮은 합계",
    "consecutive": "연속 번호",
    "arithmetic": "등차/배수 패턴",
    "grid_pattern": "용지 직선 패턴",
    "lucky": "행운수/유명 조합",
    "recent_reuse": "최근 당첨번호 재사용",
}


class PopularityModel:
    def __init__(self, df: pd.DataFrame = None, weights: Dict[str, float] = None,
                 recent_window: int = 10):
        self.weights = dict(weights) if weights else dict(DEFAULT_WEIGHTS)
        self.recent_window = recent_window
        self._recent_numbers = self._extract_recent(df) if df is not None else set()

    def _extract_recent(self, df: pd.DataFrame) -> set:
        if df is None or df.empty:
            return set()
        cols = [f"drwtNo{i}" for i in range(1, 7)]
        recent = df.sort_values("drwNo").tail(self.recent_window)
        nums = set()
        for _, row in recent.iterrows():
            for c in cols:
                nums.add(int(row[c]))
        return nums

    # ---- 개별 factor (각 0~1) ----
    @staticmethod
    def _calendar_bias(combo: List[int]) -> float:
        low31 = sum(1 for n in combo if n <= 31) / 6
        low12 = sum(1 for n in combo if n <= 12) / 6
        return 0.6 * low31 + 0.4 * low12

    @staticmethod
    def _low_sum(combo: List[int]) -> float:
        s = sum(combo)  # 21(최소)~255(최대). 낮을수록 생일편향 → 인기
        if s <= 90:
            return 1.0
        if s >= 170:
            return 0.0
        return (170 - s) / (170 - 90)

    @staticmethod
    def _consecutive(combo: List[int]) -> float:
        c = sorted(combo)
        pairs = sum(1 for i in range(5) if c[i + 1] - c[i] == 1)
        return min(pairs, 5) / 5

    @staticmethod
    def _arithmetic(combo: List[int]) -> float:
        c = sorted(combo)
        diffs = [c[i + 1] - c[i] for i in range(5)]
        if len(set(diffs)) == 1:  # 완전 등차수열
            return 1.0
        for k in (2, 3, 4, 5):  # 공배수 패턴
            if all(n % k == 0 for n in c):
                return 0.7
        run = best = 1  # 부분 등차(동일 diff 연속)
        for i in range(1, len(diffs)):
            if diffs[i] == diffs[i - 1]:
                run += 1
                best = max(best, run)
            else:
                run = 1
        return min(max(best - 1, 0) / 4, 1.0)

    @staticmethod
    def _grid_pattern(combo: List[int]) -> float:
        cols: Dict[int, int] = {}
        rows: Dict[int, int] = {}
        for n in combo:
            col = (n - 1) % GRID_COLS
            row = (n - 1) // GRID_COLS
            cols[col] = cols.get(col, 0) + 1
            rows[row] = rows.get(row, 0) + 1
        max_line = max(max(cols.values()), max(rows.values()))
        return min(max(max_line - 1, 0) / 5, 1.0)

    @staticmethod
    def _lucky(combo: List[int]) -> float:
        score = 0.0
        if 7 in combo:
            score += 0.5
        if 3 in combo:
            score += 0.2
        cset = set(combo)
        for fam in FAMOUS_COMBOS:
            if len(cset & set(fam)) >= 5:
                score = 1.0
        return min(score, 1.0)

    def _recent_reuse(self, combo: List[int]) -> float:
        if not self._recent_numbers:
            return 0.0
        overlap = len(set(combo) & self._recent_numbers)
        return min(overlap / 6, 1.0)

    def factor_scores(self, combo: List[int]) -> Dict[str, float]:
        return {
            "calendar_bias": self._calendar_bias(combo),
            "low_sum": self._low_sum(combo),
            "consecutive": self._consecutive(combo),
            "arithmetic": self._arithmetic(combo),
            "grid_pattern": self._grid_pattern(combo),
            "lucky": self._lucky(combo),
            "recent_reuse": self._recent_reuse(combo),
        }

    def popularity(self, combo: List[int]) -> float:
        fs = self.factor_scores(combo)
        total_w = sum(self.weights.values())
        return sum(self.weights[f] * fs[f] for f in fs) / total_w

    def reasons(self, combo: List[int], threshold: float = 0.5) -> List[str]:
        fs = self.factor_scores(combo)
        return [FACTOR_LABELS[f] for f in fs if fs[f] >= threshold]

    def set_weights(self, weights: Dict[str, float]):
        self.weights = dict(weights)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `venv/bin/pytest tests/test_popularity_model.py -v`
Expected: PASS (7 passed)

- [ ] **Step 5: 커밋**

```bash
git add utils/popularity_model.py tests/test_popularity_model.py
git commit -m "feat(ev): PopularityModel 휴리스틱 인기도 점수 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: EVOptimizer — EV 지수 + 조합 생성

**Files:**
- Create: `utils/ev_optimizer.py`
- Test: `tests/test_ev_optimizer.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_ev_optimizer.py`:
```python
from utils.popularity_model import PopularityModel
from utils.ev_optimizer import EVOptimizer, build_ev_rows

POPULAR = [1, 2, 3, 4, 5, 6]
UNPOPULAR = [5, 17, 26, 33, 38, 44]


def test_ev_index_higher_for_unpopular():
    opt = EVOptimizer(PopularityModel(), seed=1)
    assert opt.ev_index(UNPOPULAR) > opt.ev_index(POPULAR)


def test_ev_index_above_one_for_unpopular():
    opt = EVOptimizer(PopularityModel(), seed=1)
    # 평균보다 비인기면 EV 지수 > 1
    assert opt.ev_index(UNPOPULAR) > 1.0


def test_generate_returns_valid_tickets():
    opt = EVOptimizer(PopularityModel(), seed=1)
    rows = opt.generate(n_tickets=5, pool_size=3000)
    assert len(rows) == 5
    for r in rows:
        combo = r["combo"]
        assert len(combo) == 6
        assert len(set(combo)) == 6
        assert combo == sorted(combo)
        assert all(1 <= n <= 45 for n in combo)
        assert 0.0 <= r["popularity"] <= 1.0
        assert r["ev_index"] > 0
        assert isinstance(r["reasons"], list)


def test_generate_prefers_low_popularity():
    opt = EVOptimizer(PopularityModel(), seed=1)
    rows = opt.generate(n_tickets=5, pool_size=3000)
    # 생성된 조합 평균 인기도가 무작위 평균보다 낮아야 함
    avg_pop = sum(r["popularity"] for r in rows) / len(rows)
    assert avg_pop < opt._reference_mean()


def test_generate_diversity():
    opt = EVOptimizer(PopularityModel(), seed=1)
    rows = opt.generate(n_tickets=5, pool_size=3000)
    combos = [r["combo"] for r in rows]
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            assert len(set(combos[i]) & set(combos[j])) <= 3


def test_build_ev_rows_helper():
    import pandas as pd
    df = pd.DataFrame(columns=["drwNo", "drwtNo1", "drwtNo2", "drwtNo3",
                               "drwtNo4", "drwtNo5", "drwtNo6", "bnusNo"])
    rows = build_ev_rows(df, n_tickets=3, pool_size=2000)
    assert len(rows) == 3
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `venv/bin/pytest tests/test_ev_optimizer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.ev_optimizer'`

- [ ] **Step 3: 최소 구현 작성**

`utils/ev_optimizer.py`:
```python
"""EV(기대 수령액) 최적화 엔진.

당첨 확률은 모든 조합이 동일(1/8,145,060)하므로, '비인기 조합'을 골라
당첨 시 분배 인원을 줄여 기대 수령액을 최대화한다.
EV ∝ 1 / (1 + K * 인기도).
"""
from typing import List, Dict
import numpy as np
import pandas as pd
from utils.popularity_model import PopularityModel

# EV 지수 산출용 확산 상수(기대 동반당첨자 민감도). verify_ev로 보정 가능.
DEFAULT_K = 8.0


class EVOptimizer:
    def __init__(self, popularity_model: PopularityModel, k: float = DEFAULT_K,
                 seed: int = 42):
        self.pm = popularity_model
        self.k = k
        self.rng = np.random.default_rng(seed)
        self._p_bar = None  # 평균 인기도(기준선)

    def _random_combo(self) -> List[int]:
        return sorted(int(x) for x in
                      self.rng.choice(range(1, 46), size=6, replace=False))

    def _reference_mean(self, n_sample: int = 2000) -> float:
        if self._p_bar is None:
            pops = [self.pm.popularity(self._random_combo()) for _ in range(n_sample)]
            self._p_bar = float(np.mean(pops))
        return self._p_bar

    def ev_index(self, combo: List[int]) -> float:
        """평균 조합 대비 기대 수령액 배수. >1이면 유리."""
        p = self.pm.popularity(combo)
        p_bar = self._reference_mean()
        return (1 + self.k * p_bar) / (1 + self.k * p)

    def _weighted_combo(self, aggressiveness: float) -> List[int]:
        base = np.ones(45)
        if aggressiveness > 0:  # 고번호(>31)에 가중 → 비인기 쪽 후보 확대
            for n in range(32, 46):
                base[n - 1] += aggressiveness * 1.5
        probs = base / base.sum()
        combo = self.rng.choice(range(1, 46), size=6, replace=False, p=probs)
        return sorted(int(x) for x in combo)

    def generate(self, n_tickets: int = 5, aggressiveness: float = 1.0,
                 pool_size: int = 20000, max_overlap: int = 3) -> List[Dict]:
        """비인기 조합 n_tickets세트 생성(다양성 보장)."""
        candidates = []
        seen = set()
        for _ in range(pool_size):
            combo = tuple(self._weighted_combo(aggressiveness))
            if combo in seen:
                continue
            seen.add(combo)
            candidates.append((self.pm.popularity(list(combo)), combo))
        candidates.sort(key=lambda x: x[0])  # 인기도 오름차순

        selected: List[tuple] = []
        for _, combo in candidates:  # 그리디 다양성 선택
            if all(len(set(combo) & set(s)) <= max_overlap for s in selected):
                selected.append(combo)
            if len(selected) >= n_tickets:
                break
        i = 0  # 부족하면 겹침 제약 완화
        while len(selected) < n_tickets and i < len(candidates):
            combo = candidates[i][1]
            if combo not in selected:
                selected.append(combo)
            i += 1

        results = []
        for combo in selected[:n_tickets]:
            c = list(combo)
            results.append({
                "combo": c,
                "popularity": round(self.pm.popularity(c), 4),
                "ev_index": round(self.ev_index(c), 4),
                "reasons": self.pm.reasons(c),
            })
        return results


def build_ev_rows(df: pd.DataFrame, n_tickets: int = 5,
                  aggressiveness: float = 1.0, pool_size: int = 20000,
                  weights: Dict[str, float] = None, seed: int = 42) -> List[Dict]:
    """app/스크립트용 헬퍼: df로 모델 구성 후 EV 조합 생성."""
    pm = PopularityModel(df, weights=weights)
    opt = EVOptimizer(pm, seed=seed)
    return opt.generate(n_tickets=n_tickets, aggressiveness=aggressiveness,
                        pool_size=pool_size)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `venv/bin/pytest tests/test_ev_optimizer.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: 커밋**

```bash
git add utils/ev_optimizer.py tests/test_ev_optimizer.py
git commit -m "feat(ev): EVOptimizer 비인기 조합 생성 및 EV 지수 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: winner_data — 1등 당첨자 수 수집

**Files:**
- Create: `utils/winner_data.py`
- Test: `tests/test_winner_data.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_winner_data.py`:
```python
import pandas as pd
from utils import winner_data


SAMPLE_API = {
    "returnValue": "success", "drwNo": 1100,
    "firstPrzwnerCo": 7, "firstWinamnt": 2000000000, "totSellamnt": 90000000000,
    "drwtNo1": 1, "drwtNo2": 2, "drwtNo3": 3,
    "drwtNo4": 4, "drwtNo5": 5, "drwtNo6": 6, "bnusNo": 7,
}


def test_parse_winner_record():
    rec = winner_data.parse_winner_record(SAMPLE_API)
    assert rec == {
        "drwNo": 1100, "firstPrzwnerCo": 7,
        "firstWinamnt": 2000000000, "totSellamnt": 90000000000,
    }


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    csv_path = tmp_path / "winner_history.csv"
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(csv_path))
    df = pd.DataFrame([winner_data.parse_winner_record(SAMPLE_API)])
    winner_data.save_winner_data(df)
    loaded = winner_data.load_winner_data()
    assert list(loaded.columns) == winner_data.WINNER_COLUMNS
    assert int(loaded.iloc[0]["firstPrzwnerCo"]) == 7


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(tmp_path / "none.csv"))
    df = winner_data.load_winner_data()
    assert df.empty
    assert list(df.columns) == winner_data.WINNER_COLUMNS
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `venv/bin/pytest tests/test_winner_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.winner_data'`

- [ ] **Step 3: 최소 구현 작성**

`utils/winner_data.py`:
```python
"""1등 당첨자 수/판매액 수집 (EV 휴리스틱 검증용).

동행복권 API 응답에는 firstPrzwnerCo(1등 당첨자 수),
firstWinamnt(1등 인당 상금), totSellamnt(총 판매액)가 포함된다.
각 회차 당첨자 수는 '그 회차 당첨조합의 실제 인기도' 실측치로 쓰인다.
"""
import os
import pandas as pd
import requests
from utils.fetcher import DHLOTTERY_API_URL, ensure_data_dir

WINNER_CSV = os.path.join("data", "winner_history.csv")
WINNER_COLUMNS = ["drwNo", "firstPrzwnerCo", "firstWinamnt", "totSellamnt"]


def parse_winner_record(data: dict) -> dict:
    """API JSON dict → 1등 당첨자 레코드."""
    return {
        "drwNo": int(data["drwNo"]),
        "firstPrzwnerCo": int(data["firstPrzwnerCo"]),
        "firstWinamnt": int(data["firstWinamnt"]),
        "totSellamnt": int(data["totSellamnt"]),
    }


def load_winner_data() -> pd.DataFrame:
    if os.path.exists(WINNER_CSV):
        return pd.read_csv(WINNER_CSV)
    return pd.DataFrame(columns=WINNER_COLUMNS)


def save_winner_data(df: pd.DataFrame):
    ensure_data_dir()
    df.sort_values("drwNo").to_csv(WINNER_CSV, index=False)


def fetch_winner_data(start: int, end: int, session=None) -> pd.DataFrame:
    """[start, end] 회차의 1등 당첨자 데이터를 수집·병합·저장.

    차단/오류 회차는 건너뛴다(fetcher.py와 동일한 방어 로직).
    """
    session = session or requests.Session()
    existing = load_winner_data()
    known = set(int(x) for x in existing["drwNo"].tolist()) if not existing.empty else set()
    records = existing.to_dict("records")
    for draw_no in range(start, end + 1):
        if draw_no in known:
            continue
        try:
            resp = session.get(DHLOTTERY_API_URL.format(draw_no), timeout=5)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        if "application/json" not in resp.headers.get("content-type", "").lower():
            continue  # 차단 시 HTML 반환
        try:
            data = resp.json()
        except ValueError:
            continue
        if data.get("returnValue") != "success":
            continue
        records.append(parse_winner_record(data))
    df = pd.DataFrame(records, columns=WINNER_COLUMNS)
    save_winner_data(df)
    return df
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `venv/bin/pytest tests/test_winner_data.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: 커밋**

```bash
git add utils/winner_data.py tests/test_winner_data.py
git commit -m "feat(ev): 1등 당첨자 수 수집(winner_data) 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: verify_ev — 회귀 검증 + 가중치 보정

**Files:**
- Create: `verify_ev.py`
- Test: `tests/test_verify_ev.py`

- [ ] **Step 1: 실패하는 테스트 작성**

`tests/test_verify_ev.py`:
```python
import numpy as np
import pandas as pd
import verify_ev


def _lotto_df():
    rows = []
    combos = [
        [1, 2, 3, 4, 5, 6],       # 인기 조합
        [5, 17, 26, 33, 38, 44],  # 비인기 조합
        [2, 7, 12, 19, 24, 31],   # 생일 편중
        [13, 22, 29, 34, 41, 45], # 분산
    ]
    for i, c in enumerate(combos, start=1):
        rows.append({
            "drwNo": i, "drwtNo1": c[0], "drwtNo2": c[1], "drwtNo3": c[2],
            "drwtNo4": c[3], "drwtNo5": c[4], "drwtNo6": c[5], "bnusNo": 40,
        })
    return pd.DataFrame(rows)


def _winner_df():
    return pd.DataFrame([
        {"drwNo": 1, "firstPrzwnerCo": 20, "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
        {"drwNo": 2, "firstPrzwnerCo": 3,  "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
        {"drwNo": 3, "firstPrzwnerCo": 18, "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
        {"drwNo": 4, "firstPrzwnerCo": 5,  "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
    ])


def test_build_dataset_shapes():
    X, y, sales = verify_ev.build_dataset(_lotto_df(), _winner_df())
    assert X.shape == (4, len(verify_ev.FACTOR_ORDER))
    assert y.shape == (4,)
    assert sales.shape == (4,)


def test_run_regression_returns_metrics():
    X, y, sales = verify_ev.build_dataset(_lotto_df(), _winner_df())
    result = verify_ev.run_regression(X, y, sales)
    assert "r2" in result
    assert set(result["factor_coefs"].keys()) == set(verify_ev.FACTOR_ORDER)
    assert isinstance(result["r2"], float)


def test_calibrate_weights_normalizes_positive():
    coefs = {f: 1.0 for f in verify_ev.FACTOR_ORDER}
    w = verify_ev.calibrate_weights(coefs)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in w.values())


def test_calibrate_weights_falls_back_on_all_negative():
    from utils.popularity_model import DEFAULT_WEIGHTS
    coefs = {f: -1.0 for f in verify_ev.FACTOR_ORDER}
    w = verify_ev.calibrate_weights(coefs)
    assert w == DEFAULT_WEIGHTS
```

- [ ] **Step 2: 테스트 실패 확인**

Run: `venv/bin/pytest tests/test_verify_ev.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'verify_ev'`

- [ ] **Step 3: 최소 구현 작성**

`verify_ev.py`:
```python
"""EV 휴리스틱 검증.

각 회차 당첨조합의 인기도 factor 점수로 실제 1등 당첨자 수를 회귀한다.
양(+)의 유의한 계수 = 휴리스틱이 실제 인기도를 포착함을 반증 가능하게
입증. 같은 회귀로 factor 가중치를 보정한다(하이브리드 접근).
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from utils.popularity_model import PopularityModel, DEFAULT_WEIGHTS
from utils.fetcher import load_data
from utils.winner_data import load_winner_data

FACTOR_ORDER = list(DEFAULT_WEIGHTS.keys())


def build_dataset(lotto_df: pd.DataFrame, winner_df: pd.DataFrame):
    """각 회차 당첨조합 → factor 점수 X, 1등 당첨자수 y, 판매액 sales."""
    pm = PopularityModel()
    merged = lotto_df.merge(winner_df, on="drwNo", how="inner")
    cols = [f"drwtNo{i}" for i in range(1, 7)]
    X, y, sales = [], [], []
    for _, row in merged.iterrows():
        combo = [int(row[c]) for c in cols]
        fs = pm.factor_scores(combo)
        X.append([fs[f] for f in FACTOR_ORDER])
        y.append(float(row["firstPrzwnerCo"]))
        sales.append(float(row["totSellamnt"]))
    return np.array(X), np.array(y), np.array(sales)


def run_regression(X: np.ndarray, y: np.ndarray, sales: np.ndarray) -> dict:
    """factor 점수로 1등 당첨자 수 회귀. log(판매액) 통제."""
    log_sales = np.log(np.clip(sales, 1, None)).reshape(-1, 1)
    features = np.hstack([X, log_sales])
    model = Ridge(alpha=1.0)
    model.fit(features, y)
    r2 = float(model.score(features, y))
    factor_coefs = {f: float(c) for f, c in
                    zip(FACTOR_ORDER, model.coef_[:len(FACTOR_ORDER)])}
    return {"r2": r2, "factor_coefs": factor_coefs,
            "sales_coef": float(model.coef_[-1])}


def calibrate_weights(factor_coefs: dict) -> dict:
    """회귀 계수(양수=인기 기여)를 정규화해 가중치로 변환.

    모든 계수가 비양수(검증 실패)면 기본 가중치를 유지한다.
    """
    pos = {f: max(c, 0.0) for f, c in factor_coefs.items()}
    total = sum(pos.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    return {f: v / total for f, v in pos.items()}


def main():
    lotto_df = load_data()
    winner_df = load_winner_data()
    if winner_df.empty:
        print("검증 데이터 없음. 먼저 수집하세요:")
        print("  python -c \"from utils.winner_data import fetch_winner_data as f; f(1, 1230)\"")
        return
    X, y, sales = build_dataset(lotto_df, winner_df)
    if len(y) < 30:
        print(f"검증 표본 부족(n={len(y)}). 기본 가중치를 유지합니다.")
        return
    result = run_regression(X, y, sales)
    print(f"검증 표본 수: {len(y)}회차")
    print(f"R² = {result['r2']:.4f}")
    print("factor별 계수(양수=인기도↑ 기여):")
    for f, c in result["factor_coefs"].items():
        print(f"  {f:16s} {c:+.4f}")
    weights = calibrate_weights(result["factor_coefs"])
    print("보정된 가중치:", {k: round(v, 3) for k, v in weights.items()})


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `venv/bin/pytest tests/test_verify_ev.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: 커밋**

```bash
git add verify_ev.py tests/test_verify_ev.py
git commit -m "feat(ev): 인기도 vs 1등 당첨자 수 회귀 검증(verify_ev) 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: app.py에 'EV 최적 조합' 섹션 추가

**Files:**
- Modify: `app.py` (기존 `st.subheader("📋 최근 당첨 번호 ...")` 블록 바로 앞에 삽입)

- [ ] **Step 1: 삽입 위치 확인**

Run: `grep -n 'st.subheader("📋 최근 당첨 번호' app.py`
Expected: 한 줄 매칭(대략 404행). 이 줄 바로 위에 아래 섹션을 삽입한다.

- [ ] **Step 2: import 추가**

`app.py` 상단 import 영역(다른 `from utils...` import들과 같은 위치)에 추가:
```python
from utils.ev_optimizer import build_ev_rows
```

- [ ] **Step 3: EV 섹션 삽입**

`app.py`의 `st.subheader("📋 최근 당첨 번호 (Latest Winning Numbers)")` 줄 **바로 앞**에 다음을 삽입:
```python
# ─────────────────────────────────────────────────────────
# 💰 EV 최적 조합 (Expected-Value Optimized Combinations)
# ─────────────────────────────────────────────────────────
st.subheader("💰 EV 최적 조합 (Expected-Value Optimized)")
st.info(
    "⚠️ **당첨 확률은 무작위와 동일합니다.** 이 기능은 당첨 확률을 높이지 "
    "않습니다. 통계적으로 '비인기 조합'을 골라 **당첨됐을 때 다른 사람과 "
    "나눌 인원을 줄여 기대 수령액(EV)만 높입니다.** 로또에서 수학적으로 "
    "유일하게 조절 가능한 값입니다."
)

col_ev1, col_ev2 = st.columns(2)
with col_ev1:
    ev_n = st.slider("생성할 조합 수 (Number of tickets)", 1, 10, 5)
with col_ev2:
    ev_aggr = st.slider("비인기 강도 (Aggressiveness)", 0.0, 2.0, 1.0, 0.1)

if st.button("💰 EV 최적 조합 생성 (Generate EV-Optimized)"):
    df_ev = load_data()
    if df_ev.empty:
        st.warning("데이터가 없습니다. 먼저 데이터를 업데이트하세요.")
    else:
        with st.spinner("비인기 조합 탐색 중..."):
            ev_rows = build_ev_rows(df_ev, n_tickets=ev_n,
                                    aggressiveness=ev_aggr, pool_size=20000)
        for idx, row in enumerate(ev_rows, start=1):
            nums = " · ".join(f"{n:02d}" for n in row["combo"])
            tags = ", ".join(row["reasons"]) if row["reasons"] else "인기 요인 없음"
            st.markdown(
                f"**{idx}세트:** `{nums}`  \n"
                f"인기도 {row['popularity']:.3f} · "
                f"**EV 지수 {row['ev_index']:.2f}배** (평균 대비) · "
                f"근거: {tags}"
            )
        st.caption(
            "EV 지수 = 평균 조합 대비 기대 수령액 배수. 1보다 크면 당첨 시 "
            "평균보다 많이 받을 것으로 기대됨(확률은 동일)."
        )
```

- [ ] **Step 4: 임포트/문법 검증**

Run: `venv/bin/python -c "import ast; ast.parse(open('app.py').read()); print('syntax ok')"`
Expected: `syntax ok`

Run: `venv/bin/python -c "from utils.ev_optimizer import build_ev_rows; print('import ok')"`
Expected: `import ok`

- [ ] **Step 5: 커밋**

```bash
git add app.py
git commit -m "feat(ev): app에 EV 최적 조합 섹션 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: 전체 테스트 + 수동 검증

**Files:** 없음 (검증만)

- [ ] **Step 1: 전체 테스트 스위트 실행**

Run: `venv/bin/pytest tests/ -v`
Expected: 기존 테스트 + 신규 21개(7+6+3+5) 모두 PASS

- [ ] **Step 2: EV 엔진 스모크 실행**

Run:
```bash
venv/bin/python -c "
from utils.fetcher import load_data
from utils.ev_optimizer import build_ev_rows
rows = build_ev_rows(load_data(), n_tickets=5, pool_size=20000)
for r in rows:
    print(r['combo'], 'pop=%.3f' % r['popularity'], 'ev=%.2f' % r['ev_index'], r['reasons'])
"
```
Expected: 5개 조합 출력, 대부분 고번호 다수 포함·인기도 낮음(<0.35)·EV 지수 >1.

- [ ] **Step 3: (선택) 실데이터 검증 실행**

네트워크가 가능하면 winner 데이터 수집 후 회귀 검증:
```bash
venv/bin/python -c "from utils.winner_data import fetch_winner_data; fetch_winner_data(1000, 1230)"
venv/bin/python verify_ev.py
```
Expected: R²과 factor 계수 출력. 차단 환경이면 "검증 데이터 없음" 메시지(정상 — 설계상 우아하게 처리).

- [ ] **Step 4: Streamlit 수동 확인(사용자)**

Run: `venv/bin/streamlit run app.py`
확인: 'EV 최적 조합' 섹션에서 슬라이더 조정 후 생성 버튼 → 조합·인기도·EV 지수·근거 태그·경고 문구 표시.

- [ ] **Step 5: 최종 커밋(문서 갱신 필요 시)**

CLAUDE.md의 기능 목록에 EV 최적화를 추가할지는 사용자 판단. 필요 시:
```bash
git add CLAUDE.md
git commit -m "docs: EV 최적화 엔진 설명 추가

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review 결과

**Spec 커버리지:**
- ① 모듈 구조 → Task 1~5 ✓
- ② PopularityModel factor 7개 → Task 1 ✓
- ③ EVOptimizer(EV 지수/생성/다양성) → Task 2 ✓
- ④ winner_data 수집 → Task 3 ✓
- ⑤ verify_ev 회귀+보정 → Task 4 ✓
- ⑥ UI 탭(경고 문구 포함) → Task 5 ✓
- 테스트 → 각 Task의 TDD + Task 6 ✓
- 정직성 원칙(확률 개선 없음 명시) → Task 5 st.info ✓

**타입 일관성:** `factor_scores`/`popularity`/`reasons`/`set_weights`(Task1) ↔ EVOptimizer 사용(Task2) ↔ `FACTOR_ORDER=list(DEFAULT_WEIGHTS.keys())`(Task4) 일치. `build_ev_rows` 시그니처(Task2) ↔ app 호출(Task5) 일치. `WINNER_COLUMNS`/`parse_winner_record`(Task3) ↔ `build_dataset` 병합 키 `drwNo`, `firstPrzwnerCo`/`totSellamnt` 사용(Task4) 일치.

**Placeholder 스캔:** 없음. 모든 스텝에 실제 코드/명령/기대출력 포함.
