"""백테스트 결과를 CSV/JSON/Markdown/PNG로 출력."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from validator.backtest_engine import ModelBacktestResult
from validator.config import ValidatorConfig
from validator.metrics import BMetrics, CMetrics


def write_raw_predictions_csv(
    results: Dict[str, ModelBacktestResult],
    path,
) -> None:
    """회차·모델·예측·적중을 원시 CSV로 기록."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["draw_no", "model", "prediction_idx",
                         "predicted", "actual", "hits"])
        for model_name, result in results.items():
            for dr in result.draw_results:
                actual_set = set(dr.actual)
                for k_idx, pred in enumerate(dr.predictions):
                    hits = len(set(pred) & actual_set)
                    writer.writerow([
                        dr.draw_no,
                        model_name,
                        k_idx,
                        ",".join(str(n) for n in pred),
                        ",".join(str(n) for n in dr.actual),
                        hits,
                    ])


def write_metrics_summary_json(
    metrics: Dict[str, Dict[str, object]],
    config: ValidatorConfig,
    path,
) -> None:
    """B+C 메트릭을 재현성 컨텍스트와 함께 JSON으로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(obj):
        if isinstance(obj, (BMetrics, CMetrics)):
            d = asdict(obj)
            if "hit_distribution" in d:
                d["hit_distribution"] = {str(k): v for k, v in d["hit_distribution"].items()}
            return d
        return obj

    payload = {
        "config": asdict(config),
        "models": {
            name: {k: _to_dict(v) for k, v in model_metrics.items()}
            for name, model_metrics in metrics.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
