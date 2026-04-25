"""Walk-Forward 백테스트 엔진."""
import contextlib
import io
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from validator.config import ValidatorConfig, set_seeds
from validator.model_registry import ModelSpec


@dataclass
class DrawResult:
    """단일 회차의 백테스트 결과."""
    draw_no: int
    predictions: List[List[int]]   # 회차당 K개 예측
    probability: np.ndarray        # 45-dim 확률 분포
    actual: List[int]


@dataclass
class ModelBacktestResult:
    """단일 모델의 전체 백테스트 결과."""
    model_name: str
    draw_results: List[DrawResult] = field(default_factory=list)


class BacktestEngine:
    """Walk-Forward 백테스트 엔진."""

    MIN_TRAIN_DRAWS = 100  # 앙상블이 안정적으로 학습되려면 최소 필요

    def __init__(self, df: pd.DataFrame, config: ValidatorConfig):
        self.df = df.sort_values("drwNo").reset_index(drop=True)
        self.config = config

    def _eval_range(self) -> range:
        """평가할 회차의 인덱스 범위. 학습 데이터가 부족하면 자동으로 축소된다."""
        total = len(self.df)
        min_train = min(self.MIN_TRAIN_DRAWS, max(1, total // 2))
        window = min(self.config.eval_window_draws, total - min_train)
        window = max(0, window)
        start = total - window
        return range(start, total)

    def _extract_actual(self, row: pd.Series) -> List[int]:
        return sorted(int(row[f"drwtNo{i}"]) for i in range(1, 7))

    def _predict_k(self, model, k: int) -> List[List[int]]:
        """K개 예측을 얻는다. predict_multiple 사용 가능하면 활용."""
        if hasattr(model, "predict_multiple"):
            try:
                return [sorted(list(p)) for p in model.predict_multiple(n_sets=k)]
            except Exception:
                pass
        return [sorted(list(model.predict())) for _ in range(k)]

    def run_model(self, spec: ModelSpec) -> ModelBacktestResult:
        """단일 모델에 대해 Walk-Forward 백테스트를 수행한다."""
        set_seeds(self.config.random_seed)
        eval_range = self._eval_range()
        result = ModelBacktestResult(model_name=spec.name)

        chunk = self.config.retrain_chunk_size
        eval_indices = list(eval_range)
        current_model = None

        for i, idx in enumerate(eval_indices):
            if i % chunk == 0:
                train_df = self.df.iloc[:idx]
                current_model = spec.instantiate()
                with contextlib.redirect_stdout(io.StringIO()):
                    current_model.train(train_df)

            row = self.df.iloc[idx]
            predictions = self._predict_k(current_model, self.config.predictions_per_draw)
            try:
                probability = np.asarray(
                    current_model.get_probability_distribution(), dtype=float
                )
            except Exception:
                probability = np.ones(45) / 45
            if probability.shape != (45,):
                probability = np.ones(45) / 45

            result.draw_results.append(DrawResult(
                draw_no=int(row["drwNo"]),
                predictions=predictions,
                probability=probability,
                actual=self._extract_actual(row),
            ))

        return result

    def run_all(self, specs: List[ModelSpec]) -> Dict[str, ModelBacktestResult]:
        """여러 모델을 순차적으로 백테스트한다."""
        return {spec.name: self.run_model(spec) for spec in specs}
