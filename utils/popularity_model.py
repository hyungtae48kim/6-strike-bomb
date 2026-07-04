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
