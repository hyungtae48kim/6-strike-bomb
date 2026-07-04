import numpy as np
import pandas as pd
import verify_ev


def _lotto_df():
    rows = []
    combos = [
        [1, 2, 3, 4, 5, 6],       # 인기 조합
        [5, 17, 26, 33, 38, 44],  # 비인기 조합
        [2, 7, 12, 19, 24, 31],   # 생일 편중
        [13, 22, 29, 34, 41, 45], # 분산
    ]
    for i, c in enumerate(combos, start=1):
        rows.append({
            "drwNo": i, "drwtNo1": c[0], "drwtNo2": c[1], "drwtNo3": c[2],
            "drwtNo4": c[3], "drwtNo5": c[4], "drwtNo6": c[5], "bnusNo": 40,
        })
    return pd.DataFrame(rows)


def _winner_df():
    return pd.DataFrame([
        {"drwNo": 1, "firstPrzwnerCo": 20, "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
        {"drwNo": 2, "firstPrzwnerCo": 3,  "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
        {"drwNo": 3, "firstPrzwnerCo": 18, "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
        {"drwNo": 4, "firstPrzwnerCo": 5,  "firstWinamnt": 1, "totSellamnt": 90_000_000_000},
    ])


def test_build_dataset_shapes():
    X, y, sales = verify_ev.build_dataset(_lotto_df(), _winner_df())
    assert X.shape == (4, len(verify_ev.FACTOR_ORDER))
    assert y.shape == (4,)
    assert sales.shape == (4,)


def test_run_regression_returns_metrics():
    X, y, sales = verify_ev.build_dataset(_lotto_df(), _winner_df())
    result = verify_ev.run_regression(X, y, sales)
    assert "r2" in result
    assert set(result["factor_coefs"].keys()) == set(verify_ev.FACTOR_ORDER)
    assert isinstance(result["r2"], float)


def test_calibrate_weights_normalizes_positive():
    coefs = {f: 1.0 for f in verify_ev.FACTOR_ORDER}
    w = verify_ev.calibrate_weights(coefs)
    assert abs(sum(w.values()) - 1.0) < 1e-9
    assert all(v > 0 for v in w.values())


def test_calibrate_weights_falls_back_on_all_negative():
    from utils.popularity_model import DEFAULT_WEIGHTS
    coefs = {f: -1.0 for f in verify_ev.FACTOR_ORDER}
    w = verify_ev.calibrate_weights(coefs)
    assert w == DEFAULT_WEIGHTS
