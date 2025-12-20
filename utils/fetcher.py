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
    
    print(f"Checking for new data starting from draw {start_draw}...")
    
    while True:
        try:
            response = requests.get(API_URL.format(current_draw), timeout=5)
            if response.status_code != 200:
                print(f"Failed to fetch draw {current_draw}")
                break
                
            data = response.json()
            
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
    else:
        print("No new data to save.")

    return df

if __name__ == "__main__":
    fetch_latest_data()
