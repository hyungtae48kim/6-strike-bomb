import numpy as np
import pandas as pd
from typing import List
from .base_model import LottoModel
from collections import Counter

class StatsModel(LottoModel):
    def __init__(self):
        self.z_scores = {}
        self.frequencies = Counter()
    
    def train(self, df: pd.DataFrame):
        # Flatten all winning numbers into a single list
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(df[f'drwtNo{i}'].tolist())
        
        # 1. Theoretical Expectations
        total_draws_count = len(df)
        
        # Probability of a specific number appearing in one draw (6 balls out of 45)
        p_occurrence = 6 / 45
        
        # Expected frequency for a specific number over N draws
        expected_freq = total_draws_count * p_occurrence
        
        # Standard Deviation for Binomial Distribution B(n, p)
        std_dev = np.sqrt(total_draws_count * p_occurrence * (1 - p_occurrence))
        
        # 2. Calculate Z-Scores
        self.frequencies = Counter(all_numbers)
        self.z_scores = {}
        
        for num in range(1, 46):
            obs_freq = self.frequencies[num]
            if std_dev > 0:
                z = (obs_freq - expected_freq) / std_dev
            else:
                z = 0
            self.z_scores[num] = z
            
    def predict(self) -> List[int]:
        if not self.z_scores:
            # Fallback if not trained: random selection
            return sorted(np.random.choice(range(1, 46), size=6, replace=False).tolist())
        
        numbers = list(range(1, 46))
        z_values = np.array([self.z_scores[n] for n in numbers])
        
        # 3. Weighting using Softmax on Z-Scores to highlight "hot" numbers
        # Numerical stability shift
        e_z = np.exp(z_values - np.max(z_values)) 
        weights = e_z / e_z.sum()
        
        # Select 6 numbers without replacement
        selected = np.random.choice(numbers, size=6, replace=False, p=weights)
        
        return sorted(selected.tolist())
