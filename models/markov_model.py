import numpy as np
import pandas as pd
from typing import List
from .base_model import LottoModel


class MarkovModel(LottoModel):
    """
    마르코프 체인 기반 로또 예측 모델.
    번호 간 전이 확률을 학습하여 다음 회차 번호를 예측합니다.
    """

    def __init__(self, order=1, smoothing=0.1):
        self.order = order  # 마르코프 체인 차수
        self.smoothing = smoothing  # 라플라스 스무딩
        self.transition_matrix = None
        self.last_numbers = None
        self._probability_dist = None

    def train(self, df: pd.DataFrame):
        """전이 확률 행렬 학습"""
        print("Markov 모델 학습 시작...")

        df_sorted = df.sort_values(by='drwNo', ascending=True)

        # 45x45 전이 행렬 초기화 (라플라스 스무딩 적용)
        self.transition_matrix = np.ones((45, 45)) * self.smoothing

        # 연속된 회차 간의 전이 학습
        prev_nums = None
        for _, row in df_sorted.iterrows():
            curr_nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]

            if prev_nums is not None:
                # 이전 회차의 각 번호에서 현재 회차의 각 번호로의 전이
                for prev_num in prev_nums:
                    for curr_num in curr_nums:
                        self.transition_matrix[prev_num - 1, curr_num - 1] += 1

            prev_nums = curr_nums

        # 행 정규화 (각 행의 합이 1이 되도록)
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = self.transition_matrix / row_sums

        # 마지막 회차 번호 저장
        last_row = df_sorted.iloc[-1]
        self.last_numbers = [int(last_row[f'drwtNo{i}']) for i in range(1, 7)]

        # 확률 분포 계산
        self._compute_probability_distribution()

        print("Markov 학습 완료.")

    def _compute_probability_distribution(self):
        """마지막 상태에서의 전이 확률 기반 분포 계산"""
        if self.transition_matrix is None or self.last_numbers is None:
            self._probability_dist = np.ones(45) / 45
            return

        # 마지막 회차 번호들에서 다음 번호로의 전이 확률 평균
        probs = np.zeros(45)
        for num in self.last_numbers:
            probs += self.transition_matrix[num - 1]

        probs = probs / len(self.last_numbers)

        # 정규화
        self._probability_dist = probs / probs.sum()

    def get_probability_distribution(self) -> np.ndarray:
        """45차원 확률 벡터 반환"""
        if self._probability_dist is None:
            return np.ones(45) / 45
        return self._probability_dist.copy()

    def predict(self) -> List[int]:
        """다음 회차 6개 번호 예측"""
        probs = self.get_probability_distribution()

        numbers = list(range(1, 46))
        selected = np.random.choice(numbers, size=6, replace=False, p=probs)

        return sorted(selected.tolist())
