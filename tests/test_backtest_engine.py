import numpy as np
import pandas as pd
import pytest

from validator.backtest_engine import (
    BacktestEngine,
    DrawResult,
    ModelBacktestResult,
)
from validator.config import ValidatorConfig
from validator.model_registry import ModelSpec


class _StubModel:
    """결정적 스텁 모델: 항상 1-6번 예측."""
    def __init__(self):
        self._probs = None

    def train(self, df: pd.DataFrame):
        probs = np.full(45, 0.01)
        probs[0:6] = 0.15
        self._probs = probs / probs.sum()

    def predict(self):
        return [1, 2, 3, 4, 5, 6]

    def predict_multiple(self, n_sets=5):
        return [[1, 2, 3, 4, 5, 6] for _ in range(n_sets)]

    def get_probability_distribution(self):
        return self._probs.copy() if self._probs is not None else np.ones(45) / 45


class _StubSpec(ModelSpec):
    def instantiate(self):
        return _StubModel()


@pytest.fixture
def stub_spec():
    return _StubSpec(
        name="Stub",
        module="__main__",
        class_name="Stub",
        kwargs={},
        strategy="chunk_10",
    )


def test_engine_runs_on_window_and_produces_draw_results(sample_lotto_df, stub_spec):
    cfg = ValidatorConfig(
        eval_window_draws=20,
        predictions_per_draw=2,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    result = engine.run_model(stub_spec)

    assert isinstance(result, ModelBacktestResult)
    assert result.model_name == "Stub"
    assert len(result.draw_results) == 20
    for dr in result.draw_results:
        assert isinstance(dr, DrawResult)
        assert len(dr.predictions) == 2  # K=2
        assert len(dr.actual) == 6
        assert dr.probability.shape == (45,)


def test_engine_respects_chunk_size_by_not_retraining_every_draw(sample_lotto_df):
    instance_count = {"n": 0}

    class CountingSpec(ModelSpec):
        def instantiate(self):
            instance_count["n"] += 1
            return _StubModel()

    cfg = ValidatorConfig(
        eval_window_draws=20,
        predictions_per_draw=1,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    spec = CountingSpec(
        name="Counting", module="x", class_name="x", kwargs={}, strategy="chunk_10",
    )
    engine.run_model(spec)
    assert instance_count["n"] == 4  # 20회차 / 5 청크


def test_engine_auto_shrinks_window_when_insufficient_data(sample_lotto_df, stub_spec):
    # 50회차 데이터에 100회차 요구 → 자동 축소
    cfg = ValidatorConfig(
        eval_window_draws=100,
        predictions_per_draw=1,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    result = engine.run_model(stub_spec)
    assert 0 < len(result.draw_results) < 50


def test_run_all_returns_dict_keyed_by_model_name(sample_lotto_df, stub_spec):
    cfg = ValidatorConfig(
        eval_window_draws=10,
        predictions_per_draw=1,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    results = engine.run_all([stub_spec])
    assert set(results.keys()) == {"Stub"}
    assert isinstance(results["Stub"], ModelBacktestResult)
