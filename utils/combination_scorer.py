"""
조합 수준 확률 계산기.
개별 번호 확률을 넘어 조합의 조건부 확률을 평가합니다.
번호 간 상관관계와 조건부 확률을 이용한 순차적 샘플링을 제공합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple
from collections import defaultdict


class CombinationScorer:
    """
    조합 수준 확률 계산기.
    번호 간 공출현 패턴을 분석하여 조합의 조건부 확률을 평가합니다.
    """

    def __init__(self, df: pd.DataFrame, smoothing: float = 0.5):
        """
        역대 데이터로 상관관계 행렬과 조건부 확률을 구축합니다.

        Args:
            df: 로또 히스토리 데이터프레임
            smoothing: 라플라스 스무딩 계수
        """
        self.smoothing = smoothing
        self.n_numbers = 45
        self._build_statistics(df)

    def _build_statistics(self, df: pd.DataFrame):
        """통계 데이터 구축"""
        n_draws = len(df)

        # 각 번호의 출현 횟수
        self.frequency = np.zeros(self.n_numbers)
        # 쌍별 공출현 횟수
        self.pair_count = np.zeros((self.n_numbers, self.n_numbers))

        for _, row in df.iterrows():
            nums = [int(row[f'drwtNo{i}']) - 1 for i in range(1, 7)]  # 0-indexed
            for n in nums:
                self.frequency[n] += 1
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    self.pair_count[nums[i]][nums[j]] += 1
                    self.pair_count[nums[j]][nums[i]] += 1

        # 조건부 확률: P(j | i) = (공출현 횟수 + 스무딩) / (i 출현 횟수 + 45 * 스무딩)
        self.cond_probs = np.zeros((self.n_numbers, self.n_numbers))
        for i in range(self.n_numbers):
            denom = self.frequency[i] + self.n_numbers * self.smoothing
            if denom > 0:
                for j in range(self.n_numbers):
                    if i == j:
                        self.cond_probs[i][j] = 0  # 자기 자신 제외
                    else:
                        self.cond_probs[i][j] = (self.pair_count[i][j] + self.smoothing) / denom

        # 상관관계 행렬 (피어슨)
        self._build_correlation_matrix(df)

    def _build_correlation_matrix(self, df: pd.DataFrame):
        """45x45 피어슨 상관계수 행렬 구축"""
        n_draws = len(df)
        # 각 추첨을 45-dim 이진 벡터로 변환
        binary_matrix = np.zeros((n_draws, self.n_numbers))
        for idx, (_, row) in enumerate(df.iterrows()):
            for i in range(1, 7):
                num = int(row[f'drwtNo{i}']) - 1
                binary_matrix[idx, num] = 1.0

        # 피어슨 상관계수 행렬 계산
        self.correlation = np.corrcoef(binary_matrix.T)
        # NaN 처리 (분산이 0인 경우)
        self.correlation = np.nan_to_num(self.correlation, nan=0.0)

    def score_combination(self, combo: List[int], base_probs: np.ndarray) -> float:
        """
        조합 점수 계산.
        기본 확률에 쌍별 상관관계 조정을 적용합니다.

        Args:
            combo: 6개 번호 리스트 (1-indexed)
            base_probs: 45차원 기본 확률 벡터

        Returns:
            조합 점수 (높을수록 유리)
        """
        indices = [n - 1 for n in combo]  # 0-indexed

        # 개별 확률 곱
        prob_score = sum(np.log(base_probs[i] + 1e-10) for i in indices)

        # 쌍별 상관관계 보정
        corr_score = 0.0
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                # 양의 상관관계면 가산, 음의 상관관계면 감산
                corr_score += self.correlation[indices[i]][indices[j]]

        return prob_score + corr_score * 0.1

    def adjusted_sampling(self, base_probs: np.ndarray) -> List[int]:
        """
        순차적 조건부 샘플링.
        이미 선택된 번호와의 조건부 확률로 다음 번호 확률을 조정합니다.

        1. P(n1)에서 첫 번호 샘플
        2. P(n2|n1) 조정된 확률에서 두 번째 샘플
        3. 6개까지 반복

        Args:
            base_probs: 45차원 기본 확률 벡터

        Returns:
            정렬된 6개 번호 리스트 (1-indexed)
        """
        selected = []  # 0-indexed
        current_probs = base_probs.copy()

        for step in range(6):
            # 이미 선택된 번호 제외
            for s in selected:
                current_probs[s] = 0

            # 선택된 번호와의 조건부 확률로 가중치 조정
            if selected:
                adjustment = np.ones(self.n_numbers)
                for s in selected:
                    cond = self.cond_probs[s]
                    # 조건부 확률의 영향을 반영 (0인 자기 자신은 이미 제외됨)
                    adjustment *= (1.0 + cond)
                current_probs *= adjustment

            # 정규화
            total = current_probs.sum()
            if total > 0:
                current_probs = current_probs / total
            else:
                # fallback: 균일 분포
                remaining = [i for i in range(self.n_numbers) if i not in selected]
                current_probs = np.zeros(self.n_numbers)
                for r in remaining:
                    current_probs[r] = 1.0 / len(remaining)

            chosen = np.random.choice(self.n_numbers, p=current_probs)
            selected.append(chosen)
            current_probs = base_probs.copy()  # 다음 단계를 위해 기본 확률 복원

        return sorted([s + 1 for s in selected])

    def generate_optimal_combinations(self, base_probs: np.ndarray,
                                       n_candidates: int = 5000,
                                       n_output: int = 5) -> List[List[int]]:
        """
        후보 조합 생성 후 점수 매기기로 상위 N개 반환.

        Args:
            base_probs: 45차원 기본 확률 벡터
            n_candidates: 후보 조합 생성 수
            n_output: 반환할 최종 조합 수

        Returns:
            상위 N개 조합 리스트
        """
        candidates = []
        numbers = list(range(1, 46))

        for _ in range(n_candidates):
            # 조건부 샘플링으로 후보 생성
            combo = self.adjusted_sampling(base_probs)
            score = self.score_combination(combo, base_probs)
            candidates.append((combo, score))

        # 점수 기준 정렬 후 상위 선택
        candidates.sort(key=lambda x: x[1], reverse=True)

        # 중복 제거
        seen = set()
        results = []
        for combo, score in candidates:
            key = tuple(combo)
            if key not in seen:
                seen.add(key)
                results.append(combo)
                if len(results) >= n_output:
                    break

        return results
