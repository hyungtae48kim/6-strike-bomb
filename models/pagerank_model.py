import numpy as np
import pandas as pd
import networkx as nx
from typing import List
from collections import defaultdict
from .base_model import LottoModel


class PageRankModel(LottoModel):
    """
    PageRank 기반 로또 예측 모델.
    번호 간 동시 출현 그래프에서 PageRank 알고리즘을 적용하여
    "영향력 있는" 번호를 식별합니다.
    """

    def __init__(self, damping_factor=0.85, recent_weight=2.0, recent_draws=50):
        self.damping_factor = damping_factor
        self.recent_weight = recent_weight
        self.recent_draws = recent_draws
        self.pagerank_scores = {}
        self._probability_dist = None

    def _build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """동시 출현 그래프 구축"""
        G = nx.Graph()

        # 모든 노드 추가 (1-45)
        G.add_nodes_from(range(1, 46))

        # 엣지 가중치 계산
        edge_weights = defaultdict(float)
        df_sorted = df.sort_values(by='drwNo', ascending=True)
        total_draws = len(df_sorted)

        for idx, (_, row) in enumerate(df_sorted.iterrows()):
            # 최근 회차에 더 높은 가중치 부여
            if idx >= total_draws - self.recent_draws:
                weight = self.recent_weight
            else:
                weight = 1.0

            nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]

            # 모든 쌍에 대해 엣지 추가
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    edge_key = tuple(sorted([nums[i], nums[j]]))
                    edge_weights[edge_key] += weight

        # 그래프에 엣지 추가
        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)

        return G

    def train(self, df: pd.DataFrame):
        """PageRank 점수 계산"""
        print("PageRank 모델 학습 시작...")

        G = self._build_graph(df)

        # PageRank 계산
        self.pagerank_scores = nx.pagerank(
            G,
            alpha=self.damping_factor,
            weight='weight'
        )

        # 확률 분포 계산
        self._compute_probability_distribution()

        print("PageRank 학습 완료.")

    def _compute_probability_distribution(self):
        """PageRank 점수 기반 확률 분포 계산"""
        if not self.pagerank_scores:
            self._probability_dist = np.ones(45) / 45
            return

        scores = np.array([self.pagerank_scores.get(i, 0) for i in range(1, 46)])

        # 정규화
        if scores.sum() > 0:
            self._probability_dist = scores / scores.sum()
        else:
            self._probability_dist = np.ones(45) / 45

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
