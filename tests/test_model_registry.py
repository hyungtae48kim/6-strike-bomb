import numpy as np

from validator.model_registry import ModelSpec, get_ensemble_specs


def test_registry_returns_two_ensembles():
    specs = get_ensemble_specs()
    assert len(specs) == 2
    names = sorted(s.name for s in specs)
    assert names == ["Stacking Ensemble", "Ultimate Ensemble"]


def test_every_spec_can_instantiate_model():
    specs = get_ensemble_specs()
    for spec in specs:
        instance = spec.instantiate()
        assert instance is not None
        assert hasattr(instance, "get_probability_distribution")
        probs = instance.get_probability_distribution()
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (45,)


def test_spec_has_chunk_10_strategy():
    for spec in get_ensemble_specs():
        assert spec.strategy == "chunk_10"


def test_spec_name_is_readable():
    for spec in get_ensemble_specs():
        assert " " in spec.name
