from typing import List, Dict
import pandas as pd
import numpy as np
from .base_model import LottoModel
from .stats_model import StatsModel
from .gnn_model import GNNModel
from .bayes_model import BayesModel
from .enums import AlgorithmType

class WeightedEnsembleModel(LottoModel):
    def __init__(self, weights: Dict[str, float]):
        """
        weights: Dict with keys matching AlgorithmType values, e.g., "Stats Based": 1.5
        """
        self.weights = weights
        self.models = {
            AlgorithmType.STATS.value: StatsModel(),
            AlgorithmType.GNN.value: GNNModel(),
            AlgorithmType.BAYES.value: BayesModel()
        }
        
    def train(self, df: pd.DataFrame):
        # Train all models
        for name, model in self.models.items():
            model.train(df)

    def predict(self) -> List[int]:
        # Gather predictions from all models
        # We will use valid weights to score numbers
        
        number_scores = {}  # {number: score}
        
        for name, model in self.models.items():
            weight = self.weights.get(name, 1.0)
            
            # Predict
            try:
                pred_nums = model.predict()
            except Exception as e:
                print(f"Model {name} failed: {e}")
                pred_nums = []

            # Scoring Strategy:
            # If a model predicts a number, that number gets +weight score.
            # If multiple models predict it, scores add up.
            
            for num in pred_nums:
                number_scores[num] = number_scores.get(num, 0) + weight
                
        # Sort numbers by score descending
        sorted_nums = sorted(number_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top 6
        top_6 = [num for num, score in sorted_nums[:6]]
        
        # If less than 6 (unlikely), fill with random valid numbers not already chosen
        if len(top_6) < 6:
            remaining = [n for n in range(1, 46) if n not in top_6]
            fill_count = 6 - len(top_6)
            top_6.extend(np.random.choice(remaining, size=fill_count, replace=False))
            
        return sorted(top_6)
