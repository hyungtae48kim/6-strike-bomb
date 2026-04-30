import random
import numpy as np

from validator.config import ValidatorConfig, set_seeds, ENSEMBLE_SPECS


def test_config_has_sensible_defaults():
    cfg = ValidatorConfig()
    assert cfg.eval_window_draws == 104
    assert cfg.random_seed == 42
    assert cfg.predictions_per_draw == 5
    assert cfg.retrain_chunk_size == 10
    assert cfg.report_ttl_hours == 24
    assert cfg.results_base_dir == "validation_results"


def test_set_seeds_is_deterministic():
    set_seeds(123)
    a = (random.random(), np.random.rand())
    set_seeds(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_ensemble_specs_lists_only_ultimate_and_stacking():
    names = sorted(spec["name"] for spec in ENSEMBLE_SPECS)
    assert names == ["Stacking Ensemble", "Ultimate Ensemble"]
    for spec in ENSEMBLE_SPECS:
        assert spec["strategy"] == "chunk_10"
