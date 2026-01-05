import requests
import pandas as pd
import os
from datetime import datetime

DATA_DIR = 'data'
DATA_FILE = os.path.join(DATA_DIR, 'lotto_history.csv')
API_URL = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={}"

def ensure_data_dir():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

def load_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame(columns=['drwNo', 'drwNoDate', 'drwtNo1', 'drwtNo2', 'drwtNo3', 'drwtNo4', 'drwtNo5', 'drwtNo6', 'bnusNo'])

def fetch_latest_data():
    ensure_data_dir()
    df = load_data()
    
    start_draw = 1
    if not df.empty:
        start_draw = int(df['drwNo'].max()) + 1
    
    current_draw = start_draw
    new_data = []
    message = ""
    
    print(f"Checking for new data starting from draw {start_draw}...")
    
    while True:
        try:
            response = requests.get(API_URL.format(current_draw), timeout=5)
            if response.status_code != 200:
                print(f"Failed to fetch draw {current_draw}")
                message = f"업데이트 실패 (Draw {current_draw}, Status: {response.status_code}). 아래 [수동 입력]을 이용해주세요."
                break
            
            # Check if response is JSON (API might block and return HTML)
            try:
                data = response.json()
            except ValueError:
                print(f"Response for draw {current_draw} is not valid JSON. Possibly blocked.")
                if not new_data:
                     # Only return error if we haven't fetched anything yet
                     return df, "자동 업데이트 실패 (사이트 차단됨). 아래 [수동 입력] 메뉴를 이용해주세요."
                break # Stop fetching but save what we have

            if data.get('returnValue') == 'fail':
                # Reached the end or invalid draw number
                print(f"No more data found. Last valid draw was {current_draw - 1}.")
                break
                
            row = {
                'drwNo': data['drwNo'],
                'drwNoDate': data['drwNoDate'],
                'drwtNo1': data['drwtNo1'],
                'drwtNo2': data['drwtNo2'],
                'drwtNo3': data['drwtNo3'],
                'drwtNo4': data['drwtNo4'],
                'drwtNo5': data['drwtNo5'],
                'drwtNo6': data['drwtNo6'],
                'bnusNo': data['bnusNo']
            }
            new_data.append(row)
            if current_draw % 10 == 0:
                print(f"Fetched draw {current_draw}")
            current_draw += 1
            
        except Exception as e:
            print(f"Error fetching draw {current_draw}: {e}")
            message = f"오류 발생: {e}. 아래 [수동 입력]을 이용해주세요."
            break
            
    if new_data:
        new_df = pd.DataFrame(new_data)
        if not df.empty:
            df = pd.concat([df, new_df], ignore_index=True)
        else:
            df = new_df
        
        # Remove duplicates just in case
        df.drop_duplicates(subset=['drwNo'], inplace=True)
        df.sort_values(by='drwNo', inplace=True)
        df.to_csv(DATA_FILE, index=False)
        print(f"Saved {len(new_data)} new records.")
        return df, f"업데이트 성공! {len(new_data)}건 추가됨."
    else:
        print("No new data to save.")
        if message:
            return df, message
        return df, "새로운 데이터가 없습니다."

def add_manual_data(drwNo, drwNoDate, numbers, bnusNo):
    ensure_data_dir()
    df = load_data()
    
    # Check if exists
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
        'bnusNo': bnusNo
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
