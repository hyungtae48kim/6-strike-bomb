import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from .base_model import LottoModel
from .stats_model import StatsModel
from .bayes_model import BayesModel
from .gnn_model import GNNModel
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .pagerank_model import PageRankModel
from .community_model import CommunityModel
from .markov_model import MarkovModel
from .pattern_model import PatternModel
from .montecarlo_model import MonteCarloModel
from .deepsets_model import DeepSetsModel
from .enums import AlgorithmType


class UltimateEnsembleModel(LottoModel):
    """
    Ultimate 메타 앙상블 로또 예측 모델.

    10개의 개별 모델을 통합하여 최종 예측을 생성합니다:
    - Tier 1 (기존): Stats, Bayes, GNN
    - Tier 2 (딥러닝): LSTM, Transformer, DeepSets
    - Tier 3 (그래프): PageRank, Community
    - Tier 4 (확률/패턴): Markov, Pattern, MonteCarlo

    특징:
    - 모든 모델의 45차원 확률 분포를 수집
    - 동적 가중치로 확률 통합
    - 다양성 보장 메커니즘
    - 조건부 확률 기반 조합 샘플링 (CombinationScorer)
    - 비현실적 조합 필터링 (CombinationFilter)
    """

    # 앙상블에 포함되지 않는 알고리즘 목록
    EXCLUDED_ALGORITHMS = [
        AlgorithmType.ENSEMBLE,
        AlgorithmType.ULTIMATE
    ]

    def __init__(self, weights: Optional[Dict[str, float]] = None,
                 enable_diversity: bool = True,
                 diversity_weight: float = 0.1):
        self.weights = weights or {}
        self.enable_diversity = enable_diversity
        self.diversity_weight = diversity_weight

        # 모든 서브 모델 초기화 (DeepSets 추가)
        self.models = {
            AlgorithmType.STATS.value: StatsModel(),
            AlgorithmType.BAYES.value: BayesModel(),
            AlgorithmType.GNN.value: GNNModel(),
            AlgorithmType.LSTM.value: LSTMModel(epochs=50),  # 빠른 학습
            AlgorithmType.TRANSFORMER.value: TransformerModel(epochs=50),
            AlgorithmType.DEEPSETS.value: DeepSetsModel(epochs=50),
            AlgorithmType.PAGERANK.value: PageRankModel(),
            AlgorithmType.COMMUNITY.value: CommunityModel(),
            AlgorithmType.MARKOV.value: MarkovModel(),
            AlgorithmType.PATTERN.value: PatternModel(),
            AlgorithmType.MONTECARLO.value: MonteCarloModel(n_simulations=5000),
        }

        self._probability_dist = None
        self._model_predictions = {}
        self._model_probabilities = {}
        self._combination_scorer = None
        self._combination_filter = None
        self._train_df = None

    def train(self, df: pd.DataFrame):
        """모든 서브 모델 학습"""
        print("=" * 50)
        print("Ultimate Ensemble 학습 시작")
        print("=" * 50)

        self._train_df = df

        for name, model in self.models.items():
            print(f"\n[{name}] 학습 중...")
            try:
                model.train(df)
                print(f"[{name}] 완료")
            except Exception as e:
                print(f"[{name}] 오류 발생: {e}")

        # 조합 스코어러 및 필터 초기화
        try:
            from utils.combination_scorer import CombinationScorer
            from utils.analysis import CombinationFilter
            self._combination_scorer = CombinationScorer(df)
            self._combination_filter = CombinationFilter(df)
            print("\n[조합 스코어러/필터] 초기화 완료")
        except Exception as e:
            print(f"\n[조합 스코어러/필터] 초기화 오류: {e}")

        # 확률 분포 통합
        self._compute_probability_distribution()

        print("\n" + "=" * 50)
        print("Ultimate Ensemble 학습 완료")
        print("=" * 50)

    def _get_model_weight(self, model_name: str) -> float:
        """모델별 가중치 반환"""
        # 사전 정의된 가중치가 있으면 사용
        if model_name in self.weights:
            return max(self.weights[model_name], 0.1)

        # 기본 가중치 (모델 유형별 차등)
        default_weights = {
            AlgorithmType.STATS.value: 1.0,
            AlgorithmType.BAYES.value: 1.0,
            AlgorithmType.GNN.value: 1.2,
            AlgorithmType.LSTM.value: 1.5,
            AlgorithmType.TRANSFORMER.value: 1.5,
            AlgorithmType.DEEPSETS.value: 1.5,
            AlgorithmType.PAGERANK.value: 1.1,
            AlgorithmType.COMMUNITY.value: 1.0,
            AlgorithmType.MARKOV.value: 0.9,
            AlgorithmType.PATTERN.value: 1.2,
            AlgorithmType.MONTECARLO.value: 1.3,
        }

        # 캐시된 메타 학습 가중치 시도
        try:
            from utils.meta_learner import MetaLearner
            ml = MetaLearner()
            cached = ml.load_cached_weights()
            if cached and model_name in cached:
                return max(cached[model_name], 0.1)
        except Exception:
            pass

        return default_weights.get(model_name, 1.0)

    def _compute_diversity_bonus(self) -> np.ndarray:
        """
        다양성 보너스 계산.
        모델 간 예측이 다양할수록 보너스 부여.
        """
        if not self._model_probabilities:
            return np.zeros(45)

        probs_list = list(self._model_probabilities.values())
        if len(probs_list) < 2:
            return np.zeros(45)

        # 각 번호에 대해 모델 간 표준편차 계산
        stacked = np.stack(probs_list)
        std_per_number = np.std(stacked, axis=0)

        # 표준편차가 높은 번호 = 의견이 분분함 = 탐색 가치 있음
        # 정규화하여 보너스로 사용
        diversity_bonus = std_per_number / (std_per_number.sum() + 1e-10)

        return diversity_bonus

    def _compute_probability_distribution(self):
        """모든 모델의 확률 분포 통합"""
        self._model_probabilities = {}

        # 각 모델의 확률 분포 수집
        for name, model in self.models.items():
            try:
                probs = model.get_probability_distribution()
                if probs is not None and len(probs) == 45:
                    self._model_probabilities[name] = probs
            except Exception as e:
                print(f"[{name}] 확률 분포 수집 오류: {e}")

        if not self._model_probabilities:
            self._probability_dist = np.ones(45) / 45
            return

        # 가중 평균 계산
        weighted_sum = np.zeros(45)
        total_weight = 0

        for name, probs in self._model_probabilities.items():
            weight = self._get_model_weight(name)
            weighted_sum += probs * weight
            total_weight += weight

        if total_weight > 0:
            combined_probs = weighted_sum / total_weight
        else:
            combined_probs = np.ones(45) / 45

        # 다양성 보너스 적용
        if self.enable_diversity:
            diversity_bonus = self._compute_diversity_bonus()
            combined_probs = (1 - self.diversity_weight) * combined_probs + \
                             self.diversity_weight * diversity_bonus

        # 정규화
        self._probability_dist = combined_probs / combined_probs.sum()

    def get_probability_distribution(self) -> np.ndarray:
        """45차원 확률 벡터 반환"""
        if self._probability_dist is None:
            return np.ones(45) / 45
        return self._probability_dist.copy()

    def get_model_contributions(self) -> Dict[str, float]:
        """
        각 모델의 기여도 반환.
        디버깅 및 분석용.
        """
        contributions = {}
        total_weight = sum(self._get_model_weight(name)
                          for name in self._model_probabilities.keys())

        for name in self._model_probabilities.keys():
            weight = self._get_model_weight(name)
            contributions[name] = weight / total_weight if total_weight > 0 else 0

        return contributions

    def predict(self) -> List[int]:
        """다음 회차 6개 번호 예측 (조건부 확률 + 필터링 적용)"""
        probs = self.get_probability_distribution()

        # 조건부 확률 기반 샘플링 (번호 간 상관관계 반영)
        if self._combination_scorer is not None:
            combo = self._combination_scorer.adjusted_sampling(probs)
            # 필터 통과 확인
            if self._combination_filter is not None and self._combination_filter.filter(combo):
                return combo

        # 필터링된 샘플링 (필터 통과할 때까지 반복)
        if self._combination_filter is not None:
            return self._combination_filter.filtered_sampling(probs)

        # fallback: 기본 샘플링
        numbers = list(range(1, 46))
        selected = np.random.choice(numbers, size=6, replace=False, p=probs)
        return sorted(selected.tolist())

    def predict_multiple(self, n_sets: int = 5) -> List[List[int]]:
        """
        여러 세트의 번호 예측.
        각 세트는 서로 다양하도록, 필터링을 적용하여 생성합니다.
        """
        probs = self.get_probability_distribution()
        numbers = list(range(1, 46))

        results = []
        used_numbers = set()

        for _ in range(n_sets):
            # 이미 많이 선택된 번호에 페널티 적용
            adjusted_probs = probs.copy()
            for num in used_numbers:
                adjusted_probs[num - 1] *= 0.7

            adjusted_probs = adjusted_probs / adjusted_probs.sum()

            # 조합 필터 적용 샘플링
            if self._combination_filter is not None:
                combo = self._combination_filter.filtered_sampling(adjusted_probs)
            else:
                selected = np.random.choice(numbers, size=6, replace=False, p=adjusted_probs)
                combo = sorted(selected.tolist())

            results.append(combo)
            used_numbers.update(combo)

        return results

    def get_top_numbers(self, n: int = 10) -> List[tuple]:
        """
        상위 N개 번호와 확률 반환.
        """
        probs = self.get_probability_distribution()
        indexed = [(i + 1, probs[i]) for i in range(45)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:n]
