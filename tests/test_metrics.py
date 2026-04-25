import numpy as np
import pytest

from validator.metrics import BMetrics, compute_b_metrics


def test_b_metrics_counts_hits_correctly():
    predictions = [
        [[1, 2, 3, 4, 5, 6]],         # 6 hits
        [[10, 20, 30, 40, 41, 42]],   # 0 hits
        [[1, 2, 3, 40, 41, 42]],      # 3 hits
    ]
    actuals = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
    ]
    m = compute_b_metrics(predictions, actuals)
    assert m.mean_hits == (6 + 0 + 3) / 3
    assert m.hit_distribution[6] == 1
    assert m.hit_distribution[0] == 1
    assert m.hit_distribution[3] == 1
    assert m.high_tier_rate == 1 / 3


def test_b_metrics_averages_across_k_predictions():
    predictions = [
        [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 41, 42]],
    ]
    actuals = [[1, 2, 3, 4, 5, 6]]
    m = compute_b_metrics(predictions, actuals)
    assert m.mean_hits == 3.0


def test_b_metrics_baseline_comparison_is_percent():
    predictions = [[[1, 2, 3, 40, 41, 42]]]  # 3 hits
    actuals = [[1, 2, 3, 4, 5, 6]]
    m = compute_b_metrics(predictions, actuals)
    # baseline = 0.8, 3 / 0.8 - 1 = 2.75 -> 275%
    assert abs(m.baseline_improvement_pct - 275.0) < 0.1


def test_b_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_b_metrics([], [])


from validator.metrics import CMetrics, compute_c_metrics


def _peaked_probs(top_numbers: list, base: float = 0.01) -> np.ndarray:
    """상위 번호에 높은 확률을 준 45-dim 분포 (sums to 1)."""
    probs = np.full(45, base)
    for rank, num in enumerate(top_numbers):
        probs[num - 1] = base + 0.2 - rank * 0.01
    probs = probs / probs.sum()
    return probs


def test_c_metrics_top6_prob_sum_high_when_winners_are_top():
    probs = _peaked_probs([1, 2, 3, 4, 5, 6])
    actual = [1, 2, 3, 4, 5, 6]
    m = compute_c_metrics([probs], [actual])
    assert m.top6_prob_sum > 0.5


def test_c_metrics_mean_rank_near_baseline_for_uniform():
    probs = np.ones(45) / 45
    actual = [1, 2, 3, 4, 5, 6]
    m = compute_c_metrics([probs], [actual])
    assert 20 <= m.mean_rank <= 26


def test_c_metrics_top10_hits_counts_correctly():
    probs = _peaked_probs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    actual = [1, 2, 3, 40, 41, 42]
    m = compute_c_metrics([probs], [actual])
    assert m.top10_hits == 3.0


def test_c_metrics_zone_bias_detects_concentration():
    probs = np.full(45, 0.001)
    probs[0:10] = 0.099
    probs = probs / probs.sum()
    actual = [30, 31, 32, 33, 34, 35]
    m = compute_c_metrics([probs], [actual])
    assert m.zone_predicted_mass["1-10"] > 0.5


def test_c_metrics_log_likelihood_higher_for_better_predictor():
    actual = [1, 2, 3, 4, 5, 6]
    good = _peaked_probs([1, 2, 3, 4, 5, 6])
    uniform = np.ones(45) / 45
    m_good = compute_c_metrics([good], [actual])
    m_uniform = compute_c_metrics([uniform], [actual])
    assert m_good.log_likelihood > m_uniform.log_likelihood
