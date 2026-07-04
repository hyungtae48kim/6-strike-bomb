import pandas as pd
from utils import winner_data


SAMPLE_API = {
    "returnValue": "success", "drwNo": 1100,
    "firstPrzwnerCo": 7, "firstWinamnt": 2000000000, "totSellamnt": 90000000000,
    "drwtNo1": 1, "drwtNo2": 2, "drwtNo3": 3,
    "drwtNo4": 4, "drwtNo5": 5, "drwtNo6": 6, "bnusNo": 7,
}


# ---------------------------------------------------------------------------
# 공통 페이크 헬퍼
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, content_type="application/json", payload=None, raise_json=False):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self._payload = payload
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    """draw_no → FakeResponse 매핑으로 응답을 흉내낸다."""
    def __init__(self, responses):
        self.responses = responses  # dict: draw_no -> FakeResponse
        self.calls = []

    def get(self, url, timeout=None):
        # URL 끝의 drwNo 추출 (예: ...&drwNo=1 → 1)
        draw_no = int(url.rsplit("=", 1)[-1])
        self.calls.append(draw_no)
        resp = self.responses.get(draw_no)
        if resp is None:
            raise RuntimeError("network error")  # 예외 분기 테스트용
        return resp


def _success_payload(draw_no, first_prize_winner_co=3):
    """성공 응답 페이로드 생성."""
    return {
        "returnValue": "success",
        "drwNo": draw_no,
        "firstPrzwnerCo": first_prize_winner_co,
        "firstWinamnt": 2000000000,
        "totSellamnt": 90000000000,
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


# ---------------------------------------------------------------------------
# fetch_winner_data 방어 분기 단위 테스트
# ---------------------------------------------------------------------------

def test_fetch_collects_success_records(tmp_path, monkeypatch):
    """성공 응답 2개 → DataFrame 2행, WINNER_COLUMNS 일치."""
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(tmp_path / "winner_history.csv"))

    session = FakeSession({
        1: FakeResponse(payload=_success_payload(1, first_prize_winner_co=5)),
        2: FakeResponse(payload=_success_payload(2, first_prize_winner_co=9)),
    })

    df = winner_data.fetch_winner_data(1, 2, session=session)

    assert len(df) == 2
    assert list(df.columns) == winner_data.WINNER_COLUMNS
    winners = sorted(df["firstPrzwnerCo"].tolist())
    assert winners == [5, 9]


def test_fetch_skips_blocked_and_errors(tmp_path, monkeypatch):
    """HTML 차단 응답 및 5xx 오류 → 성공 1행만 반환."""
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(tmp_path / "winner_history.csv"))

    session = FakeSession({
        1: FakeResponse(payload=_success_payload(1, first_prize_winner_co=3)),
        2: FakeResponse(content_type="text/html", payload=None),        # 차단
        3: FakeResponse(status_code=500, payload=None),                  # 서버 오류
    })

    df = winner_data.fetch_winner_data(1, 3, session=session)

    assert len(df) == 1
    assert int(df.iloc[0]["drwNo"]) == 1


def test_fetch_skips_bad_returnvalue_and_json_error(tmp_path, monkeypatch):
    """returnValue!='success' 및 JSON 파싱 오류 → 0행 반환."""
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(tmp_path / "winner_history.csv"))

    session = FakeSession({
        1: FakeResponse(payload=_success_payload(1) | {"returnValue": "fail"}),
        2: FakeResponse(raise_json=True),
    })

    df = winner_data.fetch_winner_data(1, 2, session=session)

    assert len(df) == 0
    assert list(df.columns) == winner_data.WINNER_COLUMNS


def test_fetch_skips_network_exception(tmp_path, monkeypatch):
    """네트워크 예외 발생 → 크래시 없이 0행 반환."""
    monkeypatch.setattr(winner_data, "WINNER_CSV", str(tmp_path / "winner_history.csv"))

    # responses에 키가 없으면 FakeSession.get()이 RuntimeError를 발생시킴
    session = FakeSession({})

    df = winner_data.fetch_winner_data(1, 2, session=session)

    assert len(df) == 0
    assert list(df.columns) == winner_data.WINNER_COLUMNS


def test_fetch_incremental_skips_known(tmp_path, monkeypatch):
    """기존 CSV에 drwNo=1이 있으면 세션을 호출하지 않고, 결과는 2행."""
    csv_path = str(tmp_path / "winner_history.csv")
    monkeypatch.setattr(winner_data, "WINNER_CSV", csv_path)

    # drwNo=1 미리 저장
    existing = pd.DataFrame([winner_data.parse_winner_record(_success_payload(1, first_prize_winner_co=4))])
    winner_data.save_winner_data(existing)

    # 세션에는 drwNo=2만 제공
    session = FakeSession({
        2: FakeResponse(payload=_success_payload(2, first_prize_winner_co=6)),
    })

    df = winner_data.fetch_winner_data(1, 2, session=session)

    # drwNo=1은 이미 알려진 회차 → 세션 호출 없어야 함
    assert 1 not in session.calls
    # 기존 1행 + 새로운 1행 = 총 2행
    assert len(df) == 2
