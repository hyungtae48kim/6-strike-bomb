import numpy as np
import pandas as pd
from typing import List
from .base_model import LottoModel
from collections import Counter

class StatsModel(LottoModel):
    """
    통계 기반 로또 예측 모델.
    Z-score와 softmax 가중치를 사용하여 "핫넘버"를 식별합니다.
    """

    def __init__(self):
        self.z_scores = {}
        self.frequencies = Counter()
        self._probability_dist = None

    def train(self, df: pd.DataFrame):
        # 모든 당첨 번호를 하나의 리스트로 평탄화
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(df[f'drwtNo{i}'].tolist())

        # 1. 이론적 기대값 계산
        total_draws_count = len(df)

        # 한 회차에서 특정 번호가 나올 확률 (6/45)
        p_occurrence = 6 / 45

        # N 회차에서 특정 번호의 예상 빈도
        expected_freq = total_draws_count * p_occurrence

        # 이항분포의 표준편차
        std_dev = np.sqrt(total_draws_count * p_occurrence * (1 - p_occurrence))

        # 2. Z-Score 계산
        self.frequencies = Counter(all_numbers)
        self.z_scores = {}

        for num in range(1, 46):
            obs_freq = self.frequencies[num]
            if std_dev > 0:
                z = (obs_freq - expected_freq) / std_dev
            else:
                z = 0
            self.z_scores[num] = z

        # 3. 확률 분포 계산 및 저장
        self._compute_probability_distribution()

    def _compute_probability_distribution(self):
        """Z-score 기반 확률 분포 계산"""
        if not self.z_scores:
            self._probability_dist = np.ones(45) / 45
            return

        z_values = np.array([self.z_scores[n] for n in range(1, 46)])

        # Softmax로 확률 분포 변환 (수치 안정성 적용)
        e_z = np.exp(z_values - np.max(z_values))
        self._probability_dist = e_z / e_z.sum()

    def get_probability_distribution(self) -> np.ndarray:
        """45차원 확률 벡터 반환"""
        if self._probability_dist is None:
            return np.ones(45) / 45
        return self._probability_dist.copy()

    def predict(self) -> List[int]:
        if not self.z_scores:
            # 학습되지 않은 경우: 랜덤 선택
            return sorted(np.random.choice(range(1, 46), size=6, replace=False).tolist())

        numbers = list(range(1, 46))
        weights = self.get_probability_distribution()

        # 가중 샘플링으로 6개 선택 (중복 없음)
        selected = np.random.choice(numbers, size=6, replace=False, p=weights)

        return sorted(selected.tolist())
