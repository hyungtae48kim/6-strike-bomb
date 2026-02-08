"""
스태킹 앙상블 로또 예측 모델.
각 서브 모델의 45차원 확률 분포를 특성(feature)으로 사용하여
메타 모델(Ridge Regression)로 최종 확률을 예측합니다.

단순 가중 평균과 달리, 모델 간 상관관계를 자동으로 처리하고
비선형 결합을 학습할 수 있습니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from .base_model import LottoModel
from .stats_model import StatsModel
from .bayes_model import BayesModel
from .pagerank_model import PageRankModel
from .community_model import CommunityModel
from .markov_model import MarkovModel
from .pattern_model import PatternModel
from .montecarlo_model import MonteCarloModel
from .enums import AlgorithmType


class StackingEnsembleModel(LottoModel):
    """
    스태킹 앙상블 모델.
    빠른 모델 7개의 확률 분포를 특성으로 사용하여
    메타 모델로 최종 확률을 예측합니다.

    학습 과정:
    1. 시간적 분할로 메타 특성 구축 (과적합 방지)
    2. 서브 모델 확률 분포 수집 → 315-dim 피처 (7모델 × 45번호)
    3. Ridge Regression으로 메타 모델 학습
    """

    # 스태킹에 포함되지 않는 알고리즘
    EXCLUDED_ALGORITHMS = [
        AlgorithmType.ENSEMBLE,
        AlgorithmType.ULTIMATE,
        AlgorithmType.STACKING,
    ]

    def __init__(self, meta_model_type: str = 'ridge'):
        """
        Args:
            meta_model_type: 메타 모델 유형 ('ridge' | 'logistic')
        """
        self.meta_model_type = meta_model_type

        # 빠른 모델만 사용 (LSTM/Transformer/GNN/DeepSets 제외)
        self.sub_models = {
            AlgorithmType.STATS.value: StatsModel(),
            AlgorithmType.BAYES.value: BayesModel(),
            AlgorithmType.PAGERANK.value: PageRankModel(),
            AlgorithmType.COMMUNITY.value: CommunityModel(),
            AlgorithmType.MARKOV.value: MarkovModel(),
            AlgorithmType.PATTERN.value: PatternModel(),
            AlgorithmType.MONTECARLO.value: MonteCarloModel(n_simulations=3000),
        }

        self.meta_model = None
        self.scaler = None
        self._probability_dist = None
        self._feature_dim = len(self.sub_models) * 45  # 7 × 45 = 315

    def _build_meta_features(self, df: pd.DataFrame):
        """
        시간적 분할로 메타 특성을 구축합니다.
        과적합을 방지하기 위해, 각 시점에서 과거 데이터만으로 학습합니다.

        Returns:
            (X, y): X = (N, 315) 메타 특성, y = (N, 45) 이진 타겟
        """
        df_sorted = df.sort_values('drwNo').reset_index(drop=True)
        features = []
        targets = []

        # 200회차부터 50회차 간격으로 메타 특성 구축
        min_train = 200
        step = 50

        for idx in range(min_train, len(df_sorted) - 1, step):
            train_df = df_sorted.iloc[:idx]
            test_row = df_sorted.iloc[idx]

            feature_row = []
            valid = True

            for name, model in self.sub_models.items():
                try:
                    # 과거 데이터만으로 학습
                    model_copy = self._create_model_copy(name)
                    model_copy.train(train_df)
                    probs = model_copy.get_probability_distribution()

                    if probs is not None and len(probs) == 45:
                        feature_row.extend(probs)
                    else:
                        feature_row.extend(np.ones(45) / 45)
                except Exception:
                    feature_row.extend(np.ones(45) / 45)

            if len(feature_row) == self._feature_dim:
                features.append(feature_row)

                # 타겟: 45-dim 이진 벡터
                target = np.zeros(45)
                for i in range(1, 7):
                    target[int(test_row[f'drwtNo{i}']) - 1] = 1.0
                targets.append(target)

        return np.array(features), np.array(targets)

    def _create_model_copy(self, name: str):
        """서브 모델의 새 인스턴스 생성"""
        model_map = {
            AlgorithmType.STATS.value: lambda: StatsModel(),
            AlgorithmType.BAYES.value: lambda: BayesModel(),
            AlgorithmType.PAGERANK.value: lambda: PageRankModel(),
            AlgorithmType.COMMUNITY.value: lambda: CommunityModel(),
            AlgorithmType.MARKOV.value: lambda: MarkovModel(),
            AlgorithmType.PATTERN.value: lambda: PatternModel(),
            AlgorithmType.MONTECARLO.value: lambda: MonteCarloModel(n_simulations=1000),
        }
        factory = model_map.get(name)
        return factory() if factory else None

    def train(self, df: pd.DataFrame):
        """
        스태킹 앙상블 학습.
        1단계: 모든 서브 모델을 전체 데이터로 학습
        2단계: 메타 특성 구축 (시간적 분할)
        3단계: 메타 모델 학습
        """
        print("=" * 50)
        print("Stacking Ensemble 학습 시작")
        print("=" * 50)

        # 1단계: 서브 모델 학습 (전체 데이터)
        for name, model in self.sub_models.items():
            print(f"\n[{name}] 서브 모델 학습 중...")
            try:
                model.train(df)
                print(f"[{name}] 완료")
            except Exception as e:
                print(f"[{name}] 오류: {e}")

        # 2단계: 메타 특성 구축
        print("\n[메타 특성] 시간적 분할로 구축 중...")
        X_meta, y_meta = self._build_meta_features(df)
        print(f"  메타 특성 크기: {X_meta.shape}")

        if len(X_meta) < 5:
            print("  메타 특성이 부족합니다. 기본 확률 분포 사용.")
            self._compute_simple_ensemble()
            return

        # 3단계: 메타 모델 학습
        print(f"\n[메타 모델] {self.meta_model_type} 학습 중...")
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_meta)

        if self.meta_model_type == 'ridge':
            from sklearn.linear_model import Ridge
            self.meta_model = Ridge(alpha=1.0)
            self.meta_model.fit(X_scaled, y_meta)
        elif self.meta_model_type == 'logistic':
            from sklearn.linear_model import LogisticRegression
            from sklearn.multioutput import MultiOutputClassifier
            base_clf = LogisticRegression(max_iter=1000, C=0.1)
            self.meta_model = MultiOutputClassifier(base_clf)
            y_binary = (y_meta > 0.5).astype(int)
            self.meta_model.fit(X_scaled, y_binary)

        print("Stacking Ensemble 학습 완료")
        print("=" * 50)

        # 확률 분포 계산
        self._compute_probability_distribution()

    def _compute_simple_ensemble(self):
        """메타 특성 부족 시 단순 앙상블 사용"""
        probs_list = []
        for name, model in self.sub_models.items():
            try:
                probs = model.get_probability_distribution()
                if probs is not None and len(probs) == 45:
                    probs_list.append(probs)
            except Exception:
                pass

        if probs_list:
            combined = np.mean(probs_list, axis=0)
            self._probability_dist = combined / combined.sum()
        else:
            self._probability_dist = np.ones(45) / 45

    def _compute_probability_distribution(self):
        """메타 모델 기반 확률 분포 계산"""
        if self.meta_model is None or self.scaler is None:
            self._compute_simple_ensemble()
            return

        # 현재 서브 모델의 확률 분포 수집
        feature_row = []
        for name, model in self.sub_models.items():
            try:
                probs = model.get_probability_distribution()
                if probs is not None and len(probs) == 45:
                    feature_row.extend(probs)
                else:
                    feature_row.extend(np.ones(45) / 45)
            except Exception:
                feature_row.extend(np.ones(45) / 45)

        if len(feature_row) != self._feature_dim:
            self._compute_simple_ensemble()
            return

        X = self.scaler.transform([feature_row])

        if self.meta_model_type == 'ridge':
            meta_probs = self.meta_model.predict(X).flatten()
        elif self.meta_model_type == 'logistic':
            # predict_proba로 확률 추출
            try:
                proba_list = []
                for estimator in self.meta_model.estimators_:
                    proba_list.append(estimator.predict_proba(X)[0, 1])
                meta_probs = np.array(proba_list)
            except Exception:
                meta_probs = self.meta_model.predict(X).flatten().astype(float)

        # 음수 클리핑 후 정규화
        meta_probs = np.clip(meta_probs, 1e-10, None)
        self._probability_dist = meta_probs / meta_probs.sum()

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

    def predict_multiple(self, n_sets: int = 5) -> List[List[int]]:
        """
        여러 세트의 번호 예측.
        각 세트는 서로 다양하도록 생성됩니다.
        """
        probs = self.get_probability_distribution()
        numbers = list(range(1, 46))

        results = []
        used_numbers = set()

        for _ in range(n_sets):
            adjusted_probs = probs.copy()
            for num in used_numbers:
                adjusted_probs[num - 1] *= 0.7

            adjusted_probs = adjusted_probs / adjusted_probs.sum()

            selected = np.random.choice(numbers, size=6, replace=False, p=adjusted_probs)
            results.append(sorted(selected.tolist()))
            used_numbers.update(selected)

        return results

    def get_top_numbers(self, n: int = 10) -> List[tuple]:
        """상위 N개 번호와 확률 반환"""
        probs = self.get_probability_distribution()
        indexed = [(i + 1, probs[i]) for i in range(45)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        return indexed[:n]
