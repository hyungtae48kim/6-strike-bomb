import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_lotto_df() -> pd.DataFrame:
    """50회차 가상의 로또 이력 DataFrame."""
    rng = np.random.default_rng(0)
    rows = []
    for draw_no in range(1, 51):
        nums = sorted(rng.choice(range(1, 46), size=6, replace=False).tolist())
        bonus = int(rng.integers(1, 46))
        while bonus in nums:
            bonus = int(rng.integers(1, 46))
        rows.append({
            "drwNo": draw_no,
            "drwNoDate": f"2020-01-{draw_no:02d}",
            "drwtNo1": nums[0], "drwtNo2": nums[1], "drwtNo3": nums[2],
            "drwtNo4": nums[3], "drwtNo5": nums[4], "drwtNo6": nums[5],
            "bnusNo": bonus,
        })
    return pd.DataFrame(rows)
