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


# ---- 경계값 테스트 (boundary tests) ----

def test_low_sum_boundaries():
    pm = PopularityModel()
    # 합계 21 (<= 90) → 1.0
    assert pm.factor_scores([1, 2, 3, 4, 5, 6])["low_sum"] == pytest.approx(1.0)
    # 합계 219 (>= 170) → 0.0
    assert pm.factor_scores([25, 30, 35, 40, 44, 45])["low_sum"] == pytest.approx(0.0)


def test_arithmetic_detects_exact_progression():
    pm = PopularityModel()
    # 완전 등차수열: [5,10,15,20,25,30] → arithmetic == 1.0
    assert pm.factor_scores([5, 10, 15, 20, 25, 30])["arithmetic"] == pytest.approx(1.0)
    # 비패턴 조합: [3,8,14,27,33,41] → diffs=[5,6,13,6,8], 연속 동일 diff 없음 → 0.0
    assert pm.factor_scores([3, 8, 14, 27, 33, 41])["arithmetic"] < 0.5


def test_lucky_partial_signals():
    pm = PopularityModel()
    # 7 포함, 3 미포함, 유명 조합 아님 → lucky == 0.5
    assert pm.factor_scores([7, 13, 22, 29, 34, 41])["lucky"] == pytest.approx(0.5)
    # 7·3 모두 미포함, 유명 조합 아님 → lucky == 0.0
    assert pm.factor_scores([11, 18, 24, 29, 35, 44])["lucky"] == pytest.approx(0.0)


def test_grid_pattern_detects_column():
    pm = PopularityModel()
    # 1,8,15,22,29,36: 모두 (n-1)%7==0 → 동일 열 6개 → max_line=6 → 1.0
    assert pm.factor_scores([1, 8, 15, 22, 29, 36])["grid_pattern"] == pytest.approx(1.0)
    # 분산 조합 → grid_pattern < 1.0
    assert pm.factor_scores([2, 10, 19, 27, 33, 44])["grid_pattern"] < 1.0


def test_set_weights_rejects_bad_keys():
    pm = PopularityModel()
    # 부분 키 집합 (subset) → ValueError
    with pytest.raises(ValueError):
        pm.set_weights({"calendar_bias": 1.0})
    # 알 수 없는 키 추가 (superset) → ValueError
    bad_weights = dict(DEFAULT_WEIGHTS)
    bad_weights["unknown_factor"] = 0.1
    with pytest.raises(ValueError):
        pm.set_weights(bad_weights)
