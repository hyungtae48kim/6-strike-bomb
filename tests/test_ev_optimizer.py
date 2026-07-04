from utils.popularity_model import PopularityModel
from utils.ev_optimizer import EVOptimizer, build_ev_rows

POPULAR = [1, 2, 3, 4, 5, 6]
UNPOPULAR = [5, 17, 26, 33, 38, 44]


def test_ev_index_higher_for_unpopular():
    opt = EVOptimizer(PopularityModel(), seed=1)
    assert opt.ev_index(UNPOPULAR) > opt.ev_index(POPULAR)


def test_ev_index_above_one_for_unpopular():
    opt = EVOptimizer(PopularityModel(), seed=1)
    # 평균보다 비인기면 EV 지수 > 1
    assert opt.ev_index(UNPOPULAR) > 1.0


def test_generate_returns_valid_tickets():
    opt = EVOptimizer(PopularityModel(), seed=1)
    rows = opt.generate(n_tickets=5, pool_size=3000)
    assert len(rows) == 5
    for r in rows:
        combo = r["combo"]
        assert len(combo) == 6
        assert len(set(combo)) == 6
        assert combo == sorted(combo)
        assert all(1 <= n <= 45 for n in combo)
        assert 0.0 <= r["popularity"] <= 1.0
        assert r["ev_index"] > 0
        assert isinstance(r["reasons"], list)


def test_generate_prefers_low_popularity():
    opt = EVOptimizer(PopularityModel(), seed=1)
    rows = opt.generate(n_tickets=5, pool_size=3000)
    avg_pop = sum(r["popularity"] for r in rows) / len(rows)
    assert avg_pop < opt._reference_mean()


def test_generate_diversity():
    opt = EVOptimizer(PopularityModel(), seed=1)
    rows = opt.generate(n_tickets=5, pool_size=3000)
    combos = [r["combo"] for r in rows]
    for i in range(len(combos)):
        for j in range(i + 1, len(combos)):
            assert len(set(combos[i]) & set(combos[j])) <= 3


def test_build_ev_rows_helper():
    import pandas as pd
    df = pd.DataFrame(columns=["drwNo", "drwtNo1", "drwtNo2", "drwtNo3",
                               "drwtNo4", "drwtNo5", "drwtNo6", "bnusNo"])
    rows = build_ev_rows(df, n_tickets=3, pool_size=2000)
    assert len(rows) == 3
