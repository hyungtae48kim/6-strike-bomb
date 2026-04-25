import json
from pathlib import Path

import numpy as np
import pandas as pd

from validator.backtest_engine import DrawResult, ModelBacktestResult
from validator.config import ValidatorConfig
from validator.metrics import compute_b_metrics, compute_c_metrics
from validator.report_generator import (
    write_raw_predictions_csv,
    write_metrics_summary_json,
)


def _fake_result(name: str) -> ModelBacktestResult:
    probs = np.full(45, 0.01)
    probs[:6] = 0.15
    probs = probs / probs.sum()
    draws = [
        DrawResult(
            draw_no=1000 + i,
            predictions=[[1, 2, 3, 4, 5, 6]],
            probability=probs,
            actual=[1, 2, 3, 40, 41, 42],
        )
        for i in range(5)
    ]
    return ModelBacktestResult(model_name=name, draw_results=draws)


def test_write_raw_predictions_csv_has_one_row_per_prediction(tmp_path):
    result = _fake_result("Stub")
    out = tmp_path / "raw.csv"
    write_raw_predictions_csv({"Stub": result}, out)
    df = pd.read_csv(out)
    assert len(df) == 5
    assert {"draw_no", "model", "prediction_idx", "predicted", "actual", "hits"} <= set(df.columns)
    assert (df["hits"] == 3).all()


def test_write_metrics_summary_json_contains_b_and_c(tmp_path):
    result = _fake_result("Stub")
    b = compute_b_metrics(
        [[dr.predictions[0]] for dr in result.draw_results],
        [dr.actual for dr in result.draw_results],
    )
    c = compute_c_metrics(
        [dr.probability for dr in result.draw_results],
        [dr.actual for dr in result.draw_results],
    )
    metrics = {"Stub": {"b": b, "c": c}}
    cfg = ValidatorConfig()
    out = tmp_path / "summary.json"
    write_metrics_summary_json(metrics, cfg, out)

    data = json.loads(out.read_text())
    assert "config" in data
    assert "models" in data
    assert data["models"]["Stub"]["b"]["mean_hits"] == b.mean_hits
    assert data["models"]["Stub"]["c"]["top6_prob_sum"] == c.top6_prob_sum
