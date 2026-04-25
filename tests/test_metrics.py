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
