import pandas as pd
import os
from datetime import datetime
from typing import List, Dict
from models.enums import AlgorithmType

HISTORY_FILE = 'data/prediction_history.csv'

class HistoryManager:
    def __init__(self):
        self._ensure_history_file()

    def _ensure_history_file(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        
        if not os.path.exists(HISTORY_FILE):
            df = pd.DataFrame(columns=[
                'draw_no', 'algorithm', 'predicted_numbers', 'hit_count', 'timestamp'
            ])
            df.to_csv(HISTORY_FILE, index=False)

    def save_prediction(self, draw_no: int, algorithm: AlgorithmType, numbers: List[int]):
        """
        Save a prediction to the history file.
        initial hit_count is -1 (unknown).
        """
        # Convert list to string "1,2,3,4,5,6"
        numbers_str = ",".join(map(str, sorted(numbers)))
        
        new_row = {
            'draw_no': draw_no,
            'algorithm': algorithm.value,
            'predicted_numbers': numbers_str,
            'hit_count': -1,
            'timestamp': datetime.now().isoformat()
        }
        
        df = pd.read_csv(HISTORY_FILE)
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(HISTORY_FILE, index=False)

    def update_hit_counts(self, lotto_df: pd.DataFrame):
        """
        Update hit_count for past predictions based on actual results.
        lotto_df should have 'drwNo', 'drwtNo1'...'drwtNo6'.
        """
        if not os.path.exists(HISTORY_FILE):
            return

        hist_df = pd.read_csv(HISTORY_FILE)
        
        # Filter rows where hit_count is still -1 (unverified)
        pending_mask = hist_df['hit_count'] == -1
        if not pending_mask.any():
            return

        updated = False
        
        # Create a lookup for draw results
        # { draw_no: {1, 2, 3, 4, 5, 6} }
        results_map = {}
        for _, row in lotto_df.iterrows():
            draw_no = int(row['drwNo'])
            winning_numbers = {
                int(row[f'drwtNo{i}']) for i in range(1, 7)
            }
            results_map[draw_no] = winning_numbers

        # Iterate only pending rows
        for idx in hist_df[pending_mask].index:
            draw_no = int(hist_df.at[idx, 'draw_no'])
            
            if draw_no in results_map:
                pred_str = str(hist_df.at[idx, 'predicted_numbers'])
                pred_nums = set(map(int, pred_str.split(',')))
                
                winning_nums = results_map[draw_no]
                hits = len(pred_nums.intersection(winning_nums))
                
                hist_df.at[idx, 'hit_count'] = hits
                updated = True
        
        if updated:
            hist_df.to_csv(HISTORY_FILE, index=False)
            print("Updated hit counts for past predictions.")

    def get_weights(self) -> Dict[str, float]:
        """
        Calculate weights for each algorithm based on average hit_count.
        Base weight is 1.0.
        Weight = 1.0 + (Average Hit Count / 6.0) * Scale Factor?
        Or simply Average Hit Count (if 0, then epsilon).
        
        Let's try a simple approach:
        Average Hit Count. If no history, default to 1.0.
        """
        if not os.path.exists(HISTORY_FILE):
            return {alg.value: 1.0 for alg in AlgorithmType if alg != AlgorithmType.ENSEMBLE}

        df = pd.read_csv(HISTORY_FILE)
        
        # Filter checked rows
        checked_df = df[df['hit_count'] != -1]
        
        weights = {}
        all_algs = [alg.value for alg in AlgorithmType if alg != AlgorithmType.ENSEMBLE]
        
        if checked_df.empty:
            return {alg: 1.0 for alg in all_algs}

        # Calculate average hit count per algorithm
        avg_hits = checked_df.groupby('algorithm')['hit_count'].mean()
        
        for alg in all_algs:
            # Default to 1.0 if no data
            val = avg_hits.get(alg, 1.0)
            # Ensure it's not 0 to avoid completely ignoring it? 
            # Or strict feedback: if it sucks, it gets low weight.
            # Let's give a small epsilon floor.
            weights[alg] = max(float(val), 0.1)
            
        return weights
