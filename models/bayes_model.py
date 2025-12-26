import numpy as np
import pandas as pd
from typing import List
from .base_model import LottoModel

class BayesModel(LottoModel):
    def __init__(self):
        # Hyperparameters for Beta distribution (Uniform prior)
        self.alpha_prior = 1.0
        self.beta_prior = 1.0
        self.posterior_probs = {}

    def train(self, df: pd.DataFrame):
        """
        Train using Beta-Binomial conjugate prior.
        Each number (1-45) is treated as a Bernoulli trial.
        
        Likelihood: Binomial
        Prior: Beta(alpha, beta)
        Posterior: Beta(alpha + successes, beta + failures)
        
        Expected Value of Posterior = (alpha + successes) / (alpha + beta + total_trials)
        """
        total_draws = len(df)
        
        # Flatten all winning numbers to count occurrences
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(df[f'drwtNo{i}'].tolist())
        
        counts = pd.Series(all_numbers).value_counts()

        self.posterior_probs = {}
        
        for num in range(1, 46):
            successes = counts.get(num, 0)
            # Each draw has 1 chance for a specific number to appear? 
            # Actually, in one draw, 6 numbers are picked.
            # So "trials" for a number is total_draws.
            # "Success" is if the number was one of the 6.
            # "Failure" is if it wasn't.
            
            failures = total_draws - successes
            
            post_alpha = self.alpha_prior + successes
            post_beta = self.beta_prior + failures
            
            # Expected probability from Posterior Beta distribution
            expected_prob = post_alpha / (post_alpha + post_beta)
            self.posterior_probs[num] = expected_prob

    def predict(self) -> List[int]:
        if not self.posterior_probs:
            # Fallback random
            return sorted(np.random.choice(range(1, 46), size=6, replace=False).tolist())

        numbers = list(self.posterior_probs.keys())
        probs = list(self.posterior_probs.values())
        
        # Normalize probabilities to sum to 1 for sampling
        probs_sum = sum(probs)
        normalized_probs = [p / probs_sum for p in probs]
        
        # Select 6 numbers weighted by their posterior probability
        selected = np.random.choice(numbers, size=6, replace=False, p=normalized_probs)
        
        return sorted(selected.tolist())
