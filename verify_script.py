import sys
import os
import pandas as pd

# Add current dir to path
sys.path.append(os.getcwd())

from utils.fetcher import fetch_latest_data
from models.stats_model import StatsModel
from models.gnn_model import GNNModel

def test_system():
    print("1. Testing Data Fetcher...")
    try:
        # Fetch just a small amount of data if possible, but our fetcher is 'smart'
        # For test, we might just check if we can reach the API.
        # But let's just run it; it saves to CSV.
        df = fetch_latest_data()
        print(f"   Fetched/Loaded {len(df)} rows.")
    except Exception as e:
        print(f"   Fetcher FAILED: {e}")
        return

    print("\n2. Testing Stats Model...")
    try:
        model = StatsModel()
        model.train(df)
        pred = model.predict()
        print(f"   Prediction: {pred}")
        assert len(pred) == 6
        assert all(1 <= x <= 45 for x in pred)
        print("   Stats Model OK.")
    except Exception as e:
        print(f"   Stats Model FAILED: {e}")

    print("\n3. Testing GNN Model...")
    try:
        model = GNNModel()
        model.train(df)
        pred = model.predict()
        print(f"   Prediction: {pred}")
        assert len(pred) == 6
        assert all(1 <= x <= 45 for x in pred)
        print("   GNN Model OK.")
    except Exception as e:
        print(f"   GNN Model FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()
