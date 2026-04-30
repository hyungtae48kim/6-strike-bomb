"""B (예측 기반) 및 C (확률분포 기반) 메트릭 계산."""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.stats import rankdata


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


@dataclass
class CMetrics:
    """확률분포 기반 메트릭."""
    top6_prob_sum: float
    top10_hits: float
    mean_rank: float
    log_likelihood: float
    zone_predicted_mass: Dict[str, float]
    zone_actual_rate: Dict[str, float]


ZONES = [("1-10", 1, 10), ("11-20", 11, 20), ("21-30", 21, 30),
         ("31-40", 31, 40), ("41-45", 41, 45)]


def compute_c_metrics(
    probabilities: List[np.ndarray],
    actuals: List[List[int]],
) -> CMetrics:
    """
    probabilities: [회차][45] 확률 분포.
    actuals: [회차][6] 당첨 번호.
    """
    if not probabilities or not actuals:
        raise ValueError("probabilities and actuals must be non-empty")
    if len(probabilities) != len(actuals):
        raise ValueError("probabilities and actuals length must match")

    top6_sums: List[float] = []
    top10_hit_counts: List[float] = []
    ranks_collected: List[float] = []
    log_likes: List[float] = []
    zone_pred_sums = {z: 0.0 for z, _, _ in ZONES}
    zone_actual_counts = {z: 0 for z, _, _ in ZONES}

    for probs, actual in zip(probabilities, actuals):
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (45,):
            raise ValueError(f"probability must be 45-dim, got {probs.shape}")

        actual_idx = [n - 1 for n in actual]
        top6_sums.append(float(probs[actual_idx].sum()))

        # 동순위는 평균 랭크로 처리 (균등분포에서 mean_rank ≈ 23이 되도록)
        all_ranks = rankdata(-probs, method="average")
        ranks = [float(all_ranks[i]) for i in actual_idx]
        ranks_collected.extend(ranks)
        top10_hit_counts.append(float(sum(1 for r in ranks if r <= 10)))

        eps = 1e-12
        log_likes.append(float(np.log(np.clip(probs[actual_idx], eps, None)).sum()))

        for zone_name, lo, hi in ZONES:
            zone_pred_sums[zone_name] += float(probs[lo - 1:hi].sum())
            zone_actual_counts[zone_name] += int(sum(1 for n in actual if lo <= n <= hi))

    n_draws = len(probabilities)
    zone_pred_mass = {k: v / n_draws for k, v in zone_pred_sums.items()}
    zone_actual_rate = {k: v / (n_draws * 6) for k, v in zone_actual_counts.items()}

    return CMetrics(
        top6_prob_sum=float(np.mean(top6_sums)),
        top10_hits=float(np.mean(top10_hit_counts)),
        mean_rank=float(np.mean(ranks_collected)),
        log_likelihood=float(np.mean(log_likes)),
        zone_predicted_mass=zone_pred_mass,
        zone_actual_rate=zone_actual_rate,
    )
