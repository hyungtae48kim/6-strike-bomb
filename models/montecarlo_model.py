import numpy as np
import pandas as pd
from typing import List, Tuple
from collections import Counter
from .base_model import LottoModel


class MonteCarloModel(LottoModel):
    """
    몬테카를로 시뮬레이션 기반 로또 예측 모델.
    다수의 조합을 생성하고 품질 점수를 기반으로 최적의 조합을 선택합니다.
    """

    def __init__(self, n_simulations=10000, top_k=100):
        self.n_simulations = n_simulations
        self.top_k = top_k
        self.frequency = None
        self.pair_frequency = None
        self.ideal_sum_range = None
        self.ideal_odd_count = None
        self.best_combination = None
        self._probability_dist = None

    def _calculate_frequencies(self, df: pd.DataFrame):
        """번호 및 쌍 빈도 계산"""
        # 단일 번호 빈도
        all_numbers = []
        for i in range(1, 7):
            all_numbers.extend(df[f'drwtNo{i}'].tolist())

        self.frequency = Counter(all_numbers)

        # 쌍 빈도
        pair_counts = Counter()
        for _, row in df.iterrows():
            nums = sorted([int(row[f'drwtNo{i}']) for i in range(1, 7)])
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    pair_counts[(nums[i], nums[j])] += 1

        self.pair_frequency = pair_counts

    def _calculate_ideal_patterns(self, df: pd.DataFrame):
        """이상적인 패턴 계산"""
        sums = []
        odd_counts = []

        for _, row in df.iterrows():
            nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]
            sums.append(sum(nums))
            odd_counts.append(sum(1 for n in nums if n % 2 == 1))

        # 합계 범위 (평균 ± 1 표준편차)
        sum_mean = np.mean(sums)
        sum_std = np.std(sums)
        self.ideal_sum_range = (sum_mean - sum_std, sum_mean + sum_std)

        # 가장 흔한 홀수 개수
        self.ideal_odd_count = Counter(odd_counts).most_common(1)[0][0]

    def _score_combination(self, combo: List[int]) -> float:
        """조합의 품질 점수 계산"""
        score = 0.0

        # 1. 빈도 점수 (각 번호가 자주 나온 정도)
        for num in combo:
            score += self.frequency.get(num, 0)

        # 2. 쌍 점수 (함께 나온 쌍의 빈도)
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                pair = (combo[i], combo[j])
                score += self.pair_frequency.get(pair, 0) * 0.5

        # 3. 합계 점수
        combo_sum = sum(combo)
        if self.ideal_sum_range[0] <= combo_sum <= self.ideal_sum_range[1]:
            score += 50
        else:
            # 범위에서 벗어난 정도에 따라 감점
            distance = min(
                abs(combo_sum - self.ideal_sum_range[0]),
                abs(combo_sum - self.ideal_sum_range[1])
            )
            score -= distance * 0.5

        # 4. 홀짝 균형 점수
        odd_count = sum(1 for n in combo if n % 2 == 1)
        if odd_count == self.ideal_odd_count:
            score += 30
        else:
            score -= abs(odd_count - self.ideal_odd_count) * 10

        # 5. 번호 분포 점수 (1-15, 16-30, 31-45 각 그룹에서 균등 분포)
        low = sum(1 for n in combo if n <= 15)
        mid = sum(1 for n in combo if 16 <= n <= 30)
        high = sum(1 for n in combo if n >= 31)
        if low >= 1 and mid >= 1 and high >= 1:
            score += 20

        # 6. 연속 번호 패널티 (너무 많은 연속 번호는 감점)
        consecutive = 0
        sorted_combo = sorted(combo)
        for i in range(len(sorted_combo) - 1):
            if sorted_combo[i + 1] - sorted_combo[i] == 1:
                consecutive += 1
        if consecutive > 2:
            score -= (consecutive - 2) * 15

        return score

    def train(self, df: pd.DataFrame):
        """통계 계산 및 시뮬레이션"""
        print("Monte Carlo 모델 학습 시작...")

        self._calculate_frequencies(df)
        self._calculate_ideal_patterns(df)

        # 몬테카를로 시뮬레이션
        print(f"  {self.n_simulations}개 조합 시뮬레이션 중...")

        combinations = []
        for _ in range(self.n_simulations):
            combo = sorted(np.random.choice(range(1, 46), size=6, replace=False).tolist())
            score = self._score_combination(combo)
            combinations.append((combo, score))

        # 상위 조합 선택
        combinations.sort(key=lambda x: x[1], reverse=True)
        top_combos = combinations[:self.top_k]

        # 최고 조합 저장
        self.best_combination = top_combos[0][0]

        # 확률 분포 계산 (상위 조합에서 각 번호의 출현 빈도)
        self._compute_probability_distribution(top_combos)

        print(f"Monte Carlo 학습 완료. 최고 점수: {top_combos[0][1]:.2f}")

    def _compute_probability_distribution(self, top_combos: List[Tuple[List[int], float]]):
        """상위 조합 기반 확률 분포 계산"""
        freq = np.zeros(45)

        for combo, score in top_combos:
            # 점수에 비례하여 가중치 부여
            weight = score / max(top_combos[0][1], 1)
            for num in combo:
                freq[num - 1] += weight

        if freq.sum() > 0:
            self._probability_dist = freq / freq.sum()
        else:
            self._probability_dist = np.ones(45) / 45

    def get_probability_distribution(self) -> np.ndarray:
        """45차원 확률 벡터 반환"""
        if self._probability_dist is None:
            return np.ones(45) / 45
        return self._probability_dist.copy()

    def predict(self) -> List[int]:
        """다음 회차 6개 번호 예측"""
        # 옵션 1: 최고 점수 조합 반환
        if self.best_combination:
            return self.best_combination

        # 옵션 2: 확률 분포 기반 샘플링
        probs = self.get_probability_distribution()
        numbers = list(range(1, 46))
        selected = np.random.choice(numbers, size=6, replace=False, p=probs)
        return sorted(selected.tolist())
