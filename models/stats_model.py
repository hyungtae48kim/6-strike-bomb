from .base_model import LottoModel
import pandas as pd
import random
from collections import Counter
from typing import List

class StatsModel(LottoModel):
    def __init__(self):
        self.frequencies = Counter()
        self.hot_numbers = []
    
    def train(self, df: pd.DataFrame):
        # Flatten all winning numbers into a single list
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(df[f'drwtNo{i}'].tolist())
        
        # Calculate frequency
        self.frequencies = Counter(all_numbers)
        
        # Determine "hot" numbers (top 20 most frequent)
        self.hot_numbers = [num for num, count in self.frequencies.most_common(20)]

    def predict(self) -> List[int]:
        # Simple Logic: 
        # Pick 3 numbers from "hot" numbers (weighted by frequency)
        # Pick 3 numbers randomly from the rest to add variety
        
        all_nums = list(range(1, 46))
        
        # Weighted selection for hot numbers
        weights = [self.frequencies[n] for n in self.hot_numbers]
        # Normalize weights
        total_w = sum(weights)
        weights = [w/total_w for w in weights]
        
        full_prediction = list(np_choice(self.hot_numbers, size=6, replace=False, p=weights))
        return sorted(full_prediction)

# Helper to avoid numpy dependency in this simple file if possible, 
# but we have numpy in requirements so let's use it for weighted choice
import numpy as np

def np_choice(a, size, replace, p):
    return np.random.choice(a, size=size, replace=replace, p=p)
