"""B (예측 기반) 및 C (확률분포 기반) 메트릭 계산."""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


BASELINE_EXPECTED_HITS = 6 * 6 / 45  # 0.8


@dataclass
class BMetrics:
    """예측 기반 적중 통계."""
    mean_hits: float
    std_hits: float
    hit_distribution: Dict[int, int]
    high_tier_rate: float
    baseline_improvement_pct: float


def compute_b_metrics(
    predictions: List[List[List[int]]],
    actuals: List[List[int]],
) -> BMetrics:
    """
    predictions: [회차][K][6] 구조의 예측. K는 회차별 예측 횟수.
    actuals: [회차][6] 구조의 실제 당첨 번호.
    """
    if not predictions or not actuals:
        raise ValueError("predictions and actuals must be non-empty")
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals length must match")

    all_hits: List[int] = []
    for preds_k, actual in zip(predictions, actuals):
        actual_set = set(actual)
        for pred in preds_k:
            hits = len(set(pred) & actual_set)
            all_hits.append(hits)

    hits_arr = np.array(all_hits, dtype=float)
    mean_hits = float(hits_arr.mean())
    std_hits = float(hits_arr.std())
    distribution = dict(Counter(int(h) for h in hits_arr))
    high_tier_rate = float((hits_arr >= 5).sum() / len(hits_arr))
    improvement = (mean_hits / BASELINE_EXPECTED_HITS - 1.0) * 100.0

    return BMetrics(
        mean_hits=mean_hits,
        std_hits=std_hits,
        hit_distribution=distribution,
        high_tier_rate=high_tier_rate,
        baseline_improvement_pct=improvement,
    )
