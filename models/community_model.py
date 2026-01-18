import numpy as np
import pandas as pd
import networkx as nx
from typing import List, Dict
from collections import defaultdict
from .base_model import LottoModel

# python-louvain 패키지가 없을 경우 대체 구현 사용
try:
    import community as community_louvain
    HAS_LOUVAIN = True
except ImportError:
    HAS_LOUVAIN = False


class CommunityModel(LottoModel):
    """
    커뮤니티 탐지 기반 로또 예측 모델.
    번호들을 클러스터로 그룹화하고, 최근 활성화된 클러스터에서
    번호를 선택합니다.
    """

    def __init__(self, recent_draws=20):
        self.recent_draws = recent_draws
        self.communities = {}
        self.community_scores = {}
        self._probability_dist = None

    def _build_graph(self, df: pd.DataFrame) -> nx.Graph:
        """동시 출현 그래프 구축"""
        G = nx.Graph()
        G.add_nodes_from(range(1, 46))

        edge_weights = defaultdict(int)

        for _, row in df.iterrows():
            nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]
            for i in range(len(nums)):
                for j in range(i + 1, len(nums)):
                    edge_key = tuple(sorted([nums[i], nums[j]]))
                    edge_weights[edge_key] += 1

        for (u, v), weight in edge_weights.items():
            G.add_edge(u, v, weight=weight)

        return G

    def _detect_communities(self, G: nx.Graph) -> Dict[int, int]:
        """커뮤니티 탐지 (Louvain 알고리즘 또는 대체)"""
        if HAS_LOUVAIN:
            # Louvain 알고리즘 사용
            partition = community_louvain.best_partition(G, weight='weight')
            return partition
        else:
            # 대체: Label Propagation 사용
            communities = nx.community.label_propagation_communities(G)
            partition = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    partition[node] = idx
            return partition

    def _calculate_community_activity(self, df: pd.DataFrame) -> Dict[int, float]:
        """최근 회차에서 각 커뮤니티의 활성도 계산"""
        df_sorted = df.sort_values(by='drwNo', ascending=False)
        recent_df = df_sorted.head(self.recent_draws)

        community_counts = defaultdict(int)
        total_numbers = 0

        for _, row in recent_df.iterrows():
            for i in range(1, 7):
                num = int(row[f'drwtNo{i}'])
                if num in self.communities:
                    community_counts[self.communities[num]] += 1
                    total_numbers += 1

        # 활성도 정규화
        activity = {}
        for comm_id in set(self.communities.values()):
            activity[comm_id] = community_counts[comm_id] / max(total_numbers, 1)

        return activity

    def train(self, df: pd.DataFrame):
        """커뮤니티 탐지 및 활성도 계산"""
        print("Community 모델 학습 시작...")

        G = self._build_graph(df)
        self.communities = self._detect_communities(G)
        self.community_scores = self._calculate_community_activity(df)

        # 확률 분포 계산
        self._compute_probability_distribution(df)

        print(f"Community 학습 완료. {len(set(self.communities.values()))}개 커뮤니티 발견.")

    def _compute_probability_distribution(self, df: pd.DataFrame):
        """커뮤니티 활성도 기반 확률 분포 계산"""
        if not self.communities:
            self._probability_dist = np.ones(45) / 45
            return

        # 각 번호의 확률 = 해당 커뮤니티의 활성도
        probs = np.zeros(45)
        for num in range(1, 46):
            comm_id = self.communities.get(num, 0)
            probs[num - 1] = self.community_scores.get(comm_id, 0.1)

        # 번호 자체의 빈도도 고려
        df_sorted = df.sort_values(by='drwNo', ascending=False)
        recent_df = df_sorted.head(self.recent_draws)

        freq = np.zeros(45)
        for _, row in recent_df.iterrows():
            for i in range(1, 7):
                num = int(row[f'drwtNo{i}'])
                freq[num - 1] += 1

        freq = freq / max(freq.sum(), 1)

        # 커뮤니티 활성도와 빈도 결합
        combined = probs * 0.6 + freq * 0.4 + 0.01  # 최소값 보장

        self._probability_dist = combined / combined.sum()

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
