import pandas as pd
from utils.fetcher import add_manual_data, load_data

def verify_manual_update():
    print("Starting verification of manual update...")
    
    # 1. Use a dummy draw number
    dummy_draw = 9999
    dummy_date = "2099-12-31"
    dummy_nums = [1, 2, 3, 4, 5, 6]
    dummy_bonus = 7
    
    # 2. Add manual data
    print(f"Adding dummy draw {dummy_draw}...")
    success, msg = add_manual_data(dummy_draw, dummy_date, dummy_nums, dummy_bonus)
    print(f"Result: {success}, {msg}")
    
    if not success:
        print("Failed to add data!")
        return
        
    # 3. Verify it exists in CSV
    df = load_data()
    row = df[df['drwNo'] == dummy_draw]
    
    if not row.empty:
        print("Verification Successful: Record found in CSV.")
        print(row)
        
        # 4. Clean up
        print("Cleaning up dummy data...")
        df = df[df['drwNo'] != dummy_draw]
        df.to_csv('data/lotto_history.csv', index=False)
        print("Cleanup complete.")
    else:
        print("Verification Failed: Record NOT found in CSV.")

if __name__ == "__main__":
    verify_manual_update()
