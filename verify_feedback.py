import pandas as pd
import os
import shutil
from utils.history_manager import HistoryManager, HISTORY_FILE
from models.enums import AlgorithmType
from models.weighted_ensemble_model import WeightedEnsembleModel

def test_feedback_loop():
    print("Testing Feedback Loop...")
    
    # Backup existing history if any
    backup_path = HISTORY_FILE + ".bak"
    if os.path.exists(HISTORY_FILE):
        shutil.copy(HISTORY_FILE, backup_path)
        os.remove(HISTORY_FILE)
        
    try:
        # 1. Initialize Manager
        manager = HistoryManager()
        
        # 2. Mock a prediction for Draw 1000 (Past)
        # Let's assume winning numbers for 1000 were: 1, 2, 3, 4, 5, 6
        # We predict 1, 2, 3, 10, 11, 12 -> 3 hits
        
        draw_no = 1000
        predicted_nums = [1, 2, 3, 10, 11, 12]
        algo = AlgorithmType.STATS
        
        print(f"Saving prediction for Draw {draw_no}: {predicted_nums}")
        manager.save_prediction(draw_no, algo, predicted_nums)
        
        # Verify saved
        df = pd.read_csv(HISTORY_FILE)
        assert len(df) == 1
        assert df.iloc[0]['hit_count'] == -1
        print("Prediction saved successfully.")
        
        # 3. Update Hit Counts with Mock Data
        # Create a mock lotto dataframe
        mock_lotto_data = {
            'drwNo': [1000],
            'drwtNo1': [1], 'drwtNo2': [2], 'drwtNo3': [3],
            'drwtNo4': [4], 'drwtNo5': [5], 'drwtNo6': [6],
            'bnusNo': [7],
            'drwNoDate': ['2023-01-01']
        }
        lotto_df = pd.DataFrame(mock_lotto_data)
        
        print("Updating hit counts with mock win data...")
        manager.update_hit_counts(lotto_df)
        
        # Verify update
        df = pd.read_csv(HISTORY_FILE)
        hit_count = df.iloc[0]['hit_count']
        print(f"Hit Count updated to: {hit_count}")
        assert hit_count == 3
        
        # 4. Check Weights
        weights = manager.get_weights()
        print(f"Weights: {weights}")
        assert weights[algo.value] == 3.0
        
        # 5. Test Ensemble Model
        print("Testing Ensemble Model...")
        # Since Stats has weight 3.0, its suggestions should be prioritized.
        ensemble = WeightedEnsembleModel(weights)
        # We can't easily predict deterministic output without mocking sub-models,
        # but we can check if it runs.
        # Create a tiny dummy df for training
        dummy_df = pd.DataFrame(mock_lotto_data) 
        ensemble.train(dummy_df)
        pred = ensemble.predict()
        print(f"Ensemble Prediction: {pred}")
        assert len(pred) == 6
        
        print("All tests passed!")
        
    finally:
        # Restore backup
        if os.path.exists(backup_path):
            shutil.move(backup_path, HISTORY_FILE)
        elif os.path.exists(HISTORY_FILE):
             # If we created a file but didn't have a backup (clean state), remove it
             os.remove(HISTORY_FILE)

if __name__ == "__main__":
    test_feedback_loop()
