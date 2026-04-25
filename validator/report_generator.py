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


def write_report_md(
    metrics: Dict[str, Dict[str, object]],
    config: ValidatorConfig,
    path,
) -> None:
    """사람이 읽을 Markdown 보고서 작성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Lotto Validator Report")
    lines.append("")
    lines.append(f"- 평가 창 (회차 수): {config.eval_window_draws}")
    lines.append(f"- 회차당 예측 반복 (K): {config.predictions_per_draw}")
    lines.append(f"- 재학습 전략: chunk_{config.retrain_chunk_size}")
    lines.append(f"- 난수 시드: {config.random_seed}")
    lines.append(f"- 무작위 기댓값 (baseline): 0.8 hits")
    lines.append("")
    lines.append("## 모델별 B+C 메트릭")
    lines.append("")
    lines.append(
        "| 모델 | 평균 적중 | std | 5+ 적중율 | baseline 대비 | top6 확률합 | "
        "top10 적중 | 평균 순위 | log-like |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in metrics.items():
        b: BMetrics = m["b"]
        c: CMetrics = m["c"]
        lines.append(
            f"| {name} | {b.mean_hits:.3f} | {b.std_hits:.3f} | "
            f"{b.high_tier_rate:.3%} | {b.baseline_improvement_pct:+.1f}% | "
            f"{c.top6_prob_sum:.4f} | {c.top10_hits:.2f} | "
            f"{c.mean_rank:.2f} | {c.log_likelihood:.3f} |"
        )
    lines.append("")
    lines.append("## 적중 개수 분포")
    lines.append("")
    for name, m in metrics.items():
        b: BMetrics = m["b"]
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(b.hit_distribution.items()))
        lines.append(f"- **{name}**: {dist_str}")
    lines.append("")
    lines.append("## 번호대 편향 (예측 질량 vs 실제 당첨 비율)")
    lines.append("")
    for name, m in metrics.items():
        c: CMetrics = m["c"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| 구간 | 예측 질량 | 실제 비율 | 편차 |")
        lines.append("|---|---|---|---|")
        for zone in c.zone_predicted_mass.keys():
            pred = c.zone_predicted_mass[zone]
            actual = c.zone_actual_rate[zone]
            lines.append(
                f"| {zone} | {pred:.3f} | {actual:.3f} | {pred - actual:+.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
