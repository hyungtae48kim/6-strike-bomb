"""End-to-end 스모크 테스트: Stub 스펙으로 파이프라인 전체 통과 검증."""
import json

import numpy as np
import pandas as pd

from validator.backtest_engine import BacktestEngine
from validator.config import ValidatorConfig, set_seeds
from validator.metrics import compute_b_metrics, compute_c_metrics
from validator.model_registry import ModelSpec
from validator.report_generator import (
    write_charts,
    write_metrics_summary_json,
    write_raw_predictions_csv,
    write_report_md,
)


class _StubModel:
    def __init__(self):
        self._probs = None

    def train(self, df: pd.DataFrame):
        probs = np.full(45, 0.02)
        probs[:6] = 0.1
        self._probs = probs / probs.sum()

    def predict(self):
        return [1, 2, 3, 4, 5, 6]

    def predict_multiple(self, n_sets=5):
        return [[1, 2, 3, 4, 5, 6] for _ in range(n_sets)]

    def get_probability_distribution(self):
        return self._probs if self._probs is not None else np.ones(45) / 45


class _StubSpec(ModelSpec):
    def instantiate(self):
        return _StubModel()


def test_smoke_full_pipeline(tmp_path, sample_lotto_df):
    cfg = ValidatorConfig(
        eval_window_draws=10,
        predictions_per_draw=2,
        retrain_chunk_size=5,
    )
    set_seeds(cfg.random_seed)

    specs = [
        _StubSpec(name="StubA", module="x", class_name="x", kwargs={},
                  strategy="chunk_10"),
        _StubSpec(name="StubB", module="x", class_name="x", kwargs={},
                  strategy="chunk_10"),
    ]

    engine = BacktestEngine(sample_lotto_df, cfg)
    results = engine.run_all(specs)

    metrics = {}
    for name, result in results.items():
        preds_k = [dr.predictions for dr in result.draw_results]
        probs = [dr.probability for dr in result.draw_results]
        actuals = [dr.actual for dr in result.draw_results]
        metrics[name] = {
            "b": compute_b_metrics(preds_k, actuals),
            "c": compute_c_metrics(probs, actuals),
        }

    out = tmp_path / "smoke_out"
    write_raw_predictions_csv(results, out / "raw_predictions.csv")
    write_metrics_summary_json(metrics, cfg, out / "metrics_summary.json")
    write_report_md(metrics, cfg, out / "report.md")
    write_charts(results, metrics, out / "charts")

    assert (out / "raw_predictions.csv").exists()
    assert (out / "metrics_summary.json").exists()
    assert (out / "report.md").exists()
    assert (out / "charts" / "avg_hits_bar.png").exists()

    summary = json.loads((out / "metrics_summary.json").read_text())
    assert set(summary["models"].keys()) == {"StubA", "StubB"}
