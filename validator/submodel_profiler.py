"""서브모델 단위 hit 프로파일러.

Ultimate Ensemble의 각 서브모델(stats/lstm/...)이 walk-forward 평가창에서
top-6 예측만으로 actual에 몇 개 적중하는지를 측정한다. 약한 모델을
식별해 가지치기 후보를 정하기 위해 사용한다.
"""
from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


def top6_from_probs(probs: np.ndarray) -> List[int]:
    """45-dim 확률 분포에서 상위 6개 번호(1-indexed)를 반환."""
    arr = np.asarray(probs, dtype=float)
    idx = np.argsort(-arr)[:6]
    return sorted(int(i) + 1 for i in idx)


def score_top6_hits(top6: List[int], actual: List[int]) -> int:
    """top-6과 actual의 교집합 크기."""
    return len(set(top6) & set(actual))


class SubmodelProfiler:
    """chunk 단위로 submodel 확률 분포를 받아 누적 hit 통계 계산."""

    def __init__(self):
        self._hit_sums: Dict[str, float] = defaultdict(float)
        self._draw_counts: Dict[str, int] = defaultdict(int)

    def record_chunk(
        self,
        submodel_probs: Dict[str, np.ndarray],
        actuals: List[List[int]],
    ) -> None:
        """한 chunk 내 모든 회차에 대해 각 submodel의 hit 누적.

        submodel_probs: {name: 45-dim probs} — chunk 학습 직후 스냅샷
        actuals: chunk 내 회차들의 실제 당첨 번호 리스트
        """
        for name, probs in submodel_probs.items():
            top6 = top6_from_probs(probs)
            for actual in actuals:
                self._hit_sums[name] += score_top6_hits(top6, actual)
                self._draw_counts[name] += 1

    def aggregate(self) -> Dict[str, float]:
        """submodel별 평균 hit 수 반환."""
        out: Dict[str, float] = {}
        for name, total in self._hit_sums.items():
            count = self._draw_counts.get(name, 0)
            if count > 0:
                out[name] = float(total / count)
        return out
