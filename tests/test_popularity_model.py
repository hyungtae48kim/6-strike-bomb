import pandas as pd
import pytest
from utils.popularity_model import PopularityModel, DEFAULT_WEIGHTS

# 인기 조합: 연속 + 전부 생일범위 + 유명조합
POPULAR = [1, 2, 3, 4, 5, 6]
# 비인기 조합: 고번호 다수 + 비패턴 + 분산
UNPOPULAR = [5, 17, 26, 33, 38, 44]


def test_factor_scores_all_in_range():
    pm = PopularityModel()
    fs = pm.factor_scores(POPULAR)
    assert set(fs.keys()) == set(DEFAULT_WEIGHTS.keys())
    assert all(0.0 <= v <= 1.0 for v in fs.values())


def test_famous_combo_is_highly_popular():
    pm = PopularityModel()
    assert pm.popularity(POPULAR) > 0.7


def test_dispersed_combo_is_unpopular():
    pm = PopularityModel()
    assert pm.popularity(UNPOPULAR) < 0.35


def test_popular_greater_than_unpopular():
    pm = PopularityModel()
    assert pm.popularity(POPULAR) > pm.popularity(UNPOPULAR)


def test_calendar_bias_high_for_low_numbers():
    pm = PopularityModel()
    assert pm.factor_scores([1, 2, 3, 4, 5, 6])["calendar_bias"] == pytest.approx(1.0)
    assert pm.factor_scores([32, 35, 38, 41, 44, 45])["calendar_bias"] == pytest.approx(0.0)


def test_recent_reuse_uses_recent_draws():
    df = pd.DataFrame([{
        "drwNo": 1, "drwtNo1": 10, "drwtNo2": 11, "drwtNo3": 12,
        "drwtNo4": 13, "drwtNo5": 14, "drwtNo6": 15, "bnusNo": 40,
    }])
    pm = PopularityModel(df, recent_window=1)
    assert pm.factor_scores([10, 11, 12, 13, 14, 15])["recent_reuse"] == pytest.approx(1.0)
    assert pm.factor_scores([20, 21, 22, 23, 24, 25])["recent_reuse"] == pytest.approx(0.0)


def test_reasons_returns_tags_for_high_factors():
    pm = PopularityModel()
    reasons = pm.reasons(POPULAR)
    assert "연속 번호" in reasons
