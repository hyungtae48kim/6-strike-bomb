import re
import requests
import pandas as pd
import os
from datetime import datetime

DATA_DIR = 'data'
DATA_FILE = os.path.join(DATA_DIR, 'lotto_history.csv')
DHLOTTERY_API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"
NAVER_SEARCH_URL = "https://search.naver.com/search.naver?query=%EB%A1%9C%EB%98%90{}%ED%9A%8C"  # "로또{N}회"

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)


def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=['drwNo', 'drwNoDate', 'drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo'])


def _fetch_one_dhlottery(draw_no, session):
    """동행복권 JSON API에서 단일 회차 조회.

    반환: dict | "BLOCKED" | "NOT_FOUND" | "HTTP_ERROR:<code>" | "EXCEPTION:<msg>"
    """
    try:
        resp = session.get(DHLOTTERY_API_URL.format(draw_no), timeout=5)
    except Exception as e:
        return f"EXCEPTION:{e}"

    if resp.status_code != 200:
        return f"HTTP_ERROR:{resp.status_code}"

    # 차단 시 서버는 200 + HTML(메인 페이지)을 반환함
    ctype = resp.headers.get('content-type', '').lower()
    if 'application/json' not in ctype:
        return "BLOCKED"
    try:
        data = resp.json()
    except ValueError:
        return "BLOCKED"

    if data.get('returnValue') != 'success':
        return "NOT_FOUND"

    return {
        'drwNo': int(data['drwNo']),
        'drwNoDate': data['drwNoDate'],
        'drwtNo1': int(data['drwtNo1']),
        'drwtNo2': int(data['drwtNo2']),
        'drwtNo3': int(data['drwtNo3']),
        'drwtNo4': int(data['drwtNo4']),
        'drwtNo5': int(data['drwtNo5']),
        'drwtNo6': int(data['drwtNo6']),
        'bnusNo': int(data['bnusNo']),
    }


def _fetch_one_naver(draw_no):
    """Naver 검색 결과의 로또 위젯에서 단일 회차 파싱.

    반환: dict | "NOT_FOUND" | "PARSE_ERROR" | "HTTP_ERROR:<code>" | "EXCEPTION:<msg>"
    """
    try:
        resp = requests.get(
            NAVER_SEARCH_URL.format(draw_no),
            headers={"User-Agent": DEFAULT_UA},
            timeout=10,
        )
    except Exception as e:
        return f"EXCEPTION:{e}"

    if resp.status_code != 200:
        return f"HTTP_ERROR:{resp.status_code}"

    html = resp.text

    # 회차/날짜 마커: "1221회차 (2026.04.25.)"
    m_meta = re.search(rf'>\s*{draw_no}회차\s*\(([\d.]+)\)', html)
    if not m_meta:
        return "NOT_FOUND"
    date_str = m_meta.group(1).rstrip('.').replace('.', '-')

    # 본 번호 6개: <div class="winning_number"> ... </div> 내부의 ball span
    m_win = re.search(
        r'class="winning_number"[^>]*>(.*?)</div>',
        html, flags=re.DOTALL,
    )
    if not m_win:
        return "PARSE_ERROR"
    nums = re.findall(r'>\s*(\d{1,2})\s*<', m_win.group(1))
    if len(nums) < 6:
        return "PARSE_ERROR"
    main_nums = [int(x) for x in nums[:6]]

    # 보너스 번호: winning_number 직후 가장 먼저 등장하는 ball span
    after = html[m_win.end():m_win.end() + 2000]
    m_bonus = re.search(r'>\s*(\d{1,2})\s*<', after)
    if not m_bonus:
        return "PARSE_ERROR"
    bonus = int(m_bonus.group(1))

    return {
        'drwNo': draw_no,
        'drwNoDate': date_str,
        'drwtNo1': main_nums[0],
        'drwtNo2': main_nums[1],
        'drwtNo3': main_nums[2],
        'drwtNo4': main_nums[3],
        'drwtNo5': main_nums[4],
        'drwtNo6': main_nums[5],
        'bnusNo': bonus,
    }


def fetch_latest_data():
    ensure_data_dir()
    df = load_data()

    start_draw = 1
    if not df.empty:
        start_draw = int(df['drwNo'].max()) + 1

    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_UA})

    new_data = []
    used_naver = False
    dhlottery_blocked = False
    last_error_msg = ""
    sources_used = set()

    current_draw = start_draw
    print(f"Checking for new data starting from draw {start_draw}...")

    while True:
        # 1차: dhlottery (이미 차단 확인된 세션에서는 건너뜀)
        if not dhlottery_blocked:
            result = _fetch_one_dhlottery(current_draw, session)
            if isinstance(result, dict):
                new_data.append(result)
                sources_used.add('dhlottery')
                if current_draw % 10 == 0:
                    print(f"Fetched draw {current_draw} (dhlottery)")
                current_draw += 1
                continue
            if result == "NOT_FOUND":
                print(f"No more data found. Last valid draw was {current_draw - 1}.")
                break
            if result == "BLOCKED":
                print(f"dhlottery blocked at draw {current_draw}; falling back to Naver.")
                dhlottery_blocked = True
                # fallthrough into Naver
            else:
                last_error_msg = f"dhlottery 오류 ({result}) — Draw {current_draw}."
                print(last_error_msg)
                # fallthrough into Naver as last resort

        # 2차: Naver fallback
        result = _fetch_one_naver(current_draw)
        if isinstance(result, dict):
            new_data.append(result)
            sources_used.add('naver')
            used_naver = True
            if current_draw % 10 == 0:
                print(f"Fetched draw {current_draw} (naver)")
            current_draw += 1
            continue
        if result == "NOT_FOUND":
            print(f"No more data found via Naver. Last valid draw was {current_draw - 1}.")
            break
        # PARSE_ERROR / HTTP_ERROR / EXCEPTION
        last_error_msg = f"백업 소스(Naver) 오류 ({result}) — Draw {current_draw}."
        print(last_error_msg)
        break

    if new_data:
        new_df = pd.DataFrame(new_data)
        if not df.empty:
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df

        df.drop_duplicates(subset=['drwNo'], inplace=True)
        df.sort_values(by='drwNo', inplace=True)
        df.to_csv(DATA_FILE, index=False)
        print(f"Saved {len(new_data)} new records.")
        source_label = "+".join(sorted(sources_used)) or "unknown"
        return df, f"업데이트 성공! {len(new_data)}건 추가됨 (소스: {source_label})."

    # 새 데이터 없음
    if last_error_msg:
        if dhlottery_blocked and not used_naver:
            return df, (
                "자동 업데이트 실패: dhlottery 접근 차단 + 백업 소스도 실패. "
                "아래 [수동 입력] 메뉴를 이용해주세요. "
                f"(상세: {last_error_msg})"
            )
        return df, f"자동 업데이트 실패: {last_error_msg} 아래 [수동 입력]을 이용해주세요."

    return df, "새로운 데이터가 없습니다 (다음 추첨 대기 중)."


def add_manual_data(drwNo, drwNoDate, numbers, bnusNo):
    ensure_data_dir()
    df = load_data()

    if not df.empty and drwNo in df['drwNo'].values:
        return False, f"{drwNo}회차는 이미 존재합니다."

    new_row = {
        'drwNo': drwNo,
        'drwNoDate': drwNoDate,
        'drwtNo1': numbers[0],
        'drwtNo2': numbers[1],
        'drwtNo3': numbers[2],
        'drwtNo4': numbers[3],
        'drwtNo5': numbers[4],
        'drwtNo6': numbers[5],
        'bnusNo': bnusNo,
    }

    new_df = pd.DataFrame([new_row])
    if not df.empty:
        df = pd.concat([df, new_df], ignore_index=True)
    else:
        df = new_df

    df.drop_duplicates(subset=['drwNo'], inplace=True)
    df.sort_values(by='drwNo', inplace=True)
    df.to_csv(DATA_FILE, index=False)
    return True, f"{drwNo}회차 저장 완료!"


if __name__ == "__main__":
    fetch_latest_data()
