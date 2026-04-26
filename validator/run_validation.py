"""CLI: python -m validator.run_validation

실행 플로우:
1. 설정 로드 및 시드 고정
2. 로또 이력 로드
3. 지정된 앙상블 모델들 Walk-Forward 백테스트
4. B+C 메트릭 계산
5. CSV/JSON/MD/PNG + 체크포인트 저장
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from utils import fetcher

from validator.backtest_engine import BacktestEngine, save_checkpoint
from validator.config import ValidatorConfig, set_seeds
from validator.metrics import compute_b_metrics, compute_c_metrics
from validator.model_registry import get_ensemble_specs
from validator.report_generator import (
    write_charts,
    write_metrics_summary_json,
    write_raw_predictions_csv,
    write_report_md,
)


def _compute_metrics(results):
    """결과에서 B+C 메트릭을 계산합니다."""
    metrics = {}
    for name, result in results.items():
        preds_k = [dr.predictions for dr in result.draw_results]
        probs = [dr.probability for dr in result.draw_results]
        actuals = [dr.actual for dr in result.draw_results]
        metrics[name] = {
            "b": compute_b_metrics(preds_k, actuals),
            "c": compute_c_metrics(probs, actuals),
        }
    return metrics


def main(argv=None) -> int:
    """CLI 진입점."""
    parser = argparse.ArgumentParser(description="Lotto Validator")
    parser.add_argument("--window", type=int, default=None,
                        help="평가 창 (최근 N회차). 기본 104.")
    parser.add_argument("--k", type=int, default=None,
                        help="회차당 예측 반복 수. 기본 5.")
    parser.add_argument("--seed", type=int, default=None,
                        help="난수 시드. 기본 42.")
    parser.add_argument("--out", type=str, default=None,
                        help="결과 디렉토리 (기본 validation_results/<timestamp>/)")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Ultimate Ensemble 통합 분포에 적용할 sharpening τ. "
                             "권장 0.5~0.7. 미지정 시 sharpening 없음.")
    args = parser.parse_args(argv)

    cfg_kwargs = {}
    if args.window is not None:
        cfg_kwargs["eval_window_draws"] = args.window
    if args.k is not None:
        cfg_kwargs["predictions_per_draw"] = args.k
    if args.seed is not None:
        cfg_kwargs["random_seed"] = args.seed
    config = ValidatorConfig(**cfg_kwargs)

    set_seeds(config.random_seed)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = Path(args.out) if args.out else Path(config.results_base_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[validator] 결과 디렉토리: {out_dir}")
    print(f"[validator] 로또 이력 로드 중...")
    df = fetcher.load_data()
    if df.empty:
        print("[validator] 이력 데이터가 없습니다. fetcher로 먼저 데이터를 수집하세요.")
        return 1
    print(f"[validator] 총 {len(df)}회차 데이터 로드")

    specs = get_ensemble_specs()
    if args.temperature is not None:
        for spec in specs:
            if spec.name == "Ultimate Ensemble":
                spec.kwargs = {**spec.kwargs, "sharpening_temperature": args.temperature}
        print(f"[validator] Ultimate Ensemble sharpening τ={args.temperature}")
    print(f"[validator] 평가 대상: {[s.name for s in specs]}")

    engine = BacktestEngine(df, config)
    t0 = time.time()
    results = engine.run_all(specs)
    elapsed = time.time() - t0
    print(f"[validator] 백테스트 완료 ({elapsed:.0f}초)")

    save_checkpoint(results, out_dir / "checkpoint.json")
    metrics = _compute_metrics(results)

    write_raw_predictions_csv(results, out_dir / "raw_predictions.csv")
    write_metrics_summary_json(metrics, config, out_dir / "metrics_summary.json")
    write_report_md(metrics, config, out_dir / "report.md")
    write_charts(results, metrics, out_dir / "charts")

    print(f"[validator] 완료: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
