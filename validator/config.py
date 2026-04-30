"""Validator 설정 및 시드 고정 유틸리티."""
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidatorConfig:
    """Validator 실행 설정."""
    eval_window_draws: int = 104
    random_seed: int = 42
    predictions_per_draw: int = 5
    retrain_chunk_size: int = 10
    report_ttl_hours: int = 24
    results_base_dir: str = "validation_results"


ENSEMBLE_SPECS = [
    {
        "name": "Ultimate Ensemble",
        "module": "models.ultimate_ensemble_model",
        "class": "UltimateEnsembleModel",
        "kwargs": {},
        "strategy": "chunk_10",
    },
    {
        "name": "Stacking Ensemble",
        "module": "models.stacking_ensemble_model",
        "class": "StackingEnsembleModel",
        "kwargs": {"meta_model_type": "ridge"},
        "strategy": "chunk_10",
    },
]


def set_seeds(seed: int) -> None:
    """모든 주요 난수 라이브러리의 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
