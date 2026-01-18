import numpy as np
import pandas as pd
from typing import List
from .base_model import LottoModel

class BayesModel(LottoModel):
    """
    베이즈 정리 기반 로또 예측 모델.
    Beta-Binomial 켤레 사전분포를 사용합니다.
    """

    def __init__(self):
        # Beta 분포의 하이퍼파라미터 (균등 사전분포)
        self.alpha_prior = 1.0
        self.beta_prior = 1.0
        self.posterior_probs = {}
        self._probability_dist = None

    def train(self, df: pd.DataFrame):
        """
        Beta-Binomial 켤레 사전분포로 학습.

        우도 (Likelihood): Binomial
        사전분포 (Prior): Beta(alpha, beta)
        사후분포 (Posterior): Beta(alpha + successes, beta + failures)

        사후 기대값 = (alpha + successes) / (alpha + beta + total_trials)
        """
        total_draws = len(df)

        # 모든 당첨 번호 빈도 계산
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(df[f'drwtNo{i}'].tolist())

        counts = pd.Series(all_numbers).value_counts()

        self.posterior_probs = {}

        for num in range(1, 46):
            successes = counts.get(num, 0)
            failures = total_draws - successes

            post_alpha = self.alpha_prior + successes
            post_beta = self.beta_prior + failures

            # 사후분포의 기대 확률
            expected_prob = post_alpha / (post_alpha + post_beta)
            self.posterior_probs[num] = expected_prob

        # 확률 분포 계산
        self._compute_probability_distribution()

    def _compute_probability_distribution(self):
        """사후확률 기반 정규화된 확률 분포 계산"""
        if not self.posterior_probs:
            self._probability_dist = np.ones(45) / 45
            return

        probs = np.array([self.posterior_probs[n] for n in range(1, 46)])
        self._probability_dist = probs / probs.sum()

    def get_probability_distribution(self) -> np.ndarray:
        """45차원 확률 벡터 반환"""
        if self._probability_dist is None:
            return np.ones(45) / 45
        return self._probability_dist.copy()

    def predict(self) -> List[int]:
        if not self.posterior_probs:
            # 학습되지 않은 경우: 랜덤 선택
            return sorted(np.random.choice(range(1, 46), size=6, replace=False).tolist())

        numbers = list(range(1, 46))
        weights = self.get_probability_distribution()

        # 가중 샘플링으로 6개 선택
        selected = np.random.choice(numbers, size=6, replace=False, p=weights)

        return sorted(selected.tolist())
