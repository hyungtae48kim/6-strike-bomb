from enum import Enum

class AlgorithmType(Enum):
    """로또 예측 알고리즘 유형"""
    # Tier 1: 기존 모델
    STATS = "Stats Based"
    GNN = "GNN"
    BAYES = "Bayes Theorem"
    ENSEMBLE = "Weighted Ensemble"

    # Tier 2: 딥러닝 모델
    LSTM = "LSTM"
    TRANSFORMER = "Transformer"

    # Tier 3: 그래프 알고리즘
    PAGERANK = "PageRank"
    COMMUNITY = "Community"

    # Tier 4: 확률/패턴 모델
    MARKOV = "Markov Chain"
    PATTERN = "Pattern Analysis"
    MONTECARLO = "Monte Carlo"

    # Ultimate: 메타 앙상블
    ULTIMATE = "Ultimate Ensemble"
