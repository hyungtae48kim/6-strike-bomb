"""1등 당첨자 수/판매액 수집 (EV 휴리스틱 검증용).

동행복권 API 응답에는 firstPrzwnerCo(1등 당첨자 수),
firstWinamnt(1등 인당 상금), totSellamnt(총 판매액)가 포함된다.
각 회차 당첨자 수는 '그 회차 당첨조합의 실제 인기도' 실측치로 쓰인다.
"""
import os
import pandas as pd
import requests
from utils.fetcher import DHLOTTERY_API_URL, ensure_data_dir

WINNER_CSV = os.path.join("data", "winner_history.csv")
WINNER_COLUMNS = ["drwNo", "firstPrzwnerCo", "firstWinamnt", "totSellamnt"]


def parse_winner_record(data: dict) -> dict:
    """API JSON dict → 1등 당첨자 레코드."""
    return {
        "drwNo": int(data["drwNo"]),
        "firstPrzwnerCo": int(data["firstPrzwnerCo"]),
        "firstWinamnt": int(data["firstWinamnt"]),
        "totSellamnt": int(data["totSellamnt"]),
    }


def load_winner_data() -> pd.DataFrame:
    if os.path.exists(WINNER_CSV):
        return pd.read_csv(WINNER_CSV)
    return pd.DataFrame(columns=WINNER_COLUMNS)


def save_winner_data(df: pd.DataFrame):
    ensure_data_dir()
    df.sort_values("drwNo").to_csv(WINNER_CSV, index=False)


def fetch_winner_data(start: int, end: int, session=None) -> pd.DataFrame:
    """[start, end] 회차의 1등 당첨자 데이터를 수집·병합·저장.

    차단/오류 회차는 건너뛴다(fetcher.py와 동일한 방어 로직).
    """
    session = session or requests.Session()
    existing = load_winner_data()
    known = set(int(x) for x in existing["drwNo"].tolist()) if not existing.empty else set()
    records = existing.to_dict("records")
    for draw_no in range(start, end + 1):
        if draw_no in known:
            continue
        try:
            resp = session.get(DHLOTTERY_API_URL.format(draw_no), timeout=5)
        except Exception:
            continue
        if resp.status_code != 200:
            continue
        if "application/json" not in resp.headers.get("content-type", "").lower():
            continue  # 차단 시 HTML 반환
        try:
            data = resp.json()
        except ValueError:
            continue
        if data.get("returnValue") != "success":
            continue
        records.append(parse_winner_record(data))
    df = pd.DataFrame(records, columns=WINNER_COLUMNS)
    save_winner_data(df)
    return df
