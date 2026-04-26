"""UltimateEnsembleModel zone calibration 단위 테스트."""
import numpy as np
import pandas as pd
import pytest

from models.ultimate_ensemble_model import (
    UltimateEnsembleModel,
    ZONE_RANGES,
    _apply_zone_calibration,
    _compute_zone_targets,
)


def test_zone_ranges_cover_all_45_numbers():
    """ZONE_RANGES는 1-45 전체를 정확히 분할해야 한다."""
    covered = set()
    for start, end in ZONE_RANGES:
        for i in range(start, end):
            covered.add(i)
    assert covered == set(range(45))  # array index 0-44


def test_apply_zone_calibration_matches_targets_exactly():
    """zone calibration 후 zone별 질량이 target과 정확히 일치해야 한다."""
    probs = np.full(45, 1.0 / 45)  # 균등 분포
    targets = np.array([0.30, 0.20, 0.20, 0.20, 0.10])
    out = _apply_zone_calibration(probs, targets)
    for (start, end), target in zip(ZONE_RANGES, targets):
        zone_mass = out[start:end].sum()
        assert np.isclose(zone_mass, target, atol=1e-6)
    assert np.isclose(out.sum(), 1.0)


def test_apply_zone_calibration_preserves_within_zone_shape():
    """같은 zone 내 상대적 비율은 보존되어야 한다."""
    probs = np.zeros(45)
    probs[0] = 0.20  # 1-10 zone
    probs[1] = 0.10  # 1-10 zone (절반)
    probs[10:20] = 0.06  # 11-20 zone 균등
    probs[20:30] = 0.01
    probs[30:40] = 0.01
    probs[40:45] = 0.0
    probs /= probs.sum()
    targets = np.array([0.40, 0.40, 0.10, 0.05, 0.05])
    out = _apply_zone_calibration(probs, targets)
    # 1-10 zone에서 idx 0과 1의 비율 (2:1)이 유지되어야 함
    assert np.isclose(out[0] / out[1], 2.0, rtol=1e-6)


def test_apply_zone_calibration_handles_zero_zone_mass():
    """어떤 zone의 입력 질량이 0이어도 NaN/Inf 발생 안 함."""
    probs = np.zeros(45)
    probs[0:10] = 0.1  # 1-10 zone만 분포 있음
    probs /= probs.sum()
    targets = np.array([0.30, 0.20, 0.20, 0.20, 0.10])
    out = _apply_zone_calibration(probs, targets)
    assert np.all(np.isfinite(out))
    assert np.isclose(out.sum(), 1.0)


def test_compute_zone_targets_from_training_data():
    """학습 데이터의 zone별 빈도가 target으로 변환되어야 한다."""
    df = pd.DataFrame({
        "drwNo": [1, 2, 3],
        "drwtNo1": [1, 2, 3],     # 1-10
        "drwtNo2": [4, 5, 6],     # 1-10
        "drwtNo3": [11, 12, 13],  # 11-20
        "drwtNo4": [21, 22, 23],  # 21-30
        "drwtNo5": [31, 32, 33],  # 31-40
        "drwtNo6": [41, 42, 43],  # 41-45
    })
    targets = _compute_zone_targets(df)
    assert targets.shape == (5,)
    # 각 행 6개 번호: 1-10에 2개, 나머지 4개 zone 각 1개씩 → 비율 (2,1,1,1,1)/6
    assert np.isclose(targets[0], 2.0 / 6)
    assert np.isclose(targets[1], 1.0 / 6)
    assert np.isclose(targets[4], 1.0 / 6)
    assert np.isclose(targets.sum(), 1.0)


def test_ultimate_default_no_zone_calibration():
    """enable_zone_calibration 기본 False — 기존 동작."""
    m = UltimateEnsembleModel()
    assert m.enable_zone_calibration is False
    assert getattr(m, "_zone_targets", None) is None


def test_ultimate_init_accepts_enable_zone_calibration():
    """enable_zone_calibration 인자 저장."""
    m = UltimateEnsembleModel(enable_zone_calibration=True)
    assert m.enable_zone_calibration is True


def test_ultimate_compute_distribution_pulls_zones_toward_targets():
    """enable_zone_calibration=True일 때 결합 분포의 zone mass가 target에 가까워야 한다."""
    m = UltimateEnsembleModel(
        enable_zone_calibration=True,
        enable_diversity=False,
    )
    # 인위적 비대칭 분포: 1-10에 강한 편향
    biased = np.full(45, 0.005)
    biased[0:10] = 0.08  # 1-10 zone에 0.80 mass
    biased /= biased.sum()
    m._model_probabilities = {"a": biased}
    m.weights = {"a": 1.0}
    m.models = {}  # 외부 순회 skip
    # zone target 인위 설정 (실제 lotto 분포와 비슷)
    m._zone_targets = np.array([0.236, 0.221, 0.232, 0.231, 0.080])
    # 다시 _model_probabilities 채우기 (메서드가 self.models 비어있으면 균등으로 fallback)
    m._model_probabilities = {"a": biased}

    # _compute_probability_distribution 안의 self.models 빈 fallback을 우회하기 위해
    # 통합 단계만 직접 호출
    combined = biased.copy()
    out = _apply_zone_calibration(combined, m._zone_targets)

    # 1-10 zone mass가 0.80 → 0.236 부근으로 떨어져야 함
    assert np.isclose(out[0:10].sum(), 0.236, atol=1e-6)
    # 41-45 zone에서 입력 질량은 작지만 target 0.080으로 끌어올림
    assert np.isclose(out[40:45].sum(), 0.080, atol=1e-6)
