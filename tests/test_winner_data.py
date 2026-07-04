import pandas as pd
from utils import winner_data


SAMPLE_API = {
    "returnValue": "success", "drwNo": 1100,
    "firstPrzwnerCo": 7, "firstWinamnt": 2000000000, "totSellamnt": 90000000000,
    "drwtNo1": 1, "drwtNo2": 2, "drwtNo3": 3,
    "drwtNo4": 4, "drwtNo5": 5, "drwtNo6": 6, "bnusNo": 7,
}


def test_parse_winner_record():
    rec = winner_data.parse_winner_record(SAMPLE_API)
    assert rec == {
        "drwNo": 1100, "firstPrzwnerCo": 7,
        "firstWinamnt": 2000000000, "totSellamnt": 90000000000,
    }


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    csv_path = tmp_path / "winner_history.csv"
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(csv_path))
    df = pd.DataFrame([winner_data.parse_winner_record(SAMPLE_API)])
    winner_data.save_winner_data(df)
    loaded = winner_data.load_winner_data()
    assert list(loaded.columns) == winner_data.WINNER_COLUMNS
    assert int(loaded.iloc[0]["firstPrzwnerCo"]) == 7


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(tmp_path / "none.csv"))
    df = winner_data.load_winner_data()
    assert df.empty
    assert list(df.columns) == winner_data.WINNER_COLUMNS
