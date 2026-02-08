"""
메타 학습 기반 모델 가중치 최적화.
Walk-Forward 교차검증 결과로 각 모델의 최적 가중치를 학습합니다.
Bayesian Model Averaging과 동적 가중치 조정을 지원합니다.
"""

import numpy as np
import pandas as pd
import json
import os
from typing import Dict, List, Any, Optional


CACHE_DIR = 'data'
WEIGHTS_CACHE_FILE = os.path.join(CACHE_DIR, 'meta_weights_cache.json')


class MetaLearner:
    """
    메타 학습 기반 모델 가중치 최적화기.
    교차검증 성능에 기반하여 각 모델의 최적 가중치를 산출합니다.
    """

    def __init__(self):
        self.optimal_weights = None
        self._cached_weights = None

    def cross_validation_weights(self, model_configs: Dict[str, Dict[str, Any]],
                                  df: pd.DataFrame,
                                  initial_train_size: int = 500,
                                  test_size: int = 5,
                                  step_size: int = 100) -> Dict[str, float]:
        """
        교차검증 기반 가중치 산출.
        각 모델을 Walk-Forward 검증하여 성능 기반 가중치를 계산합니다.

        Args:
            model_configs: {모델명: {"class": ModelClass, "kwargs": {...}}}
            df: 로또 히스토리 데이터프레임
            initial_train_size: 최초 학습 세트 크기
            test_size: 테스트 세트 크기
            step_size: 윈도우 이동 크기

        Returns:
            {모델명: 가중치}
        """
        from utils.validation import WalkForwardValidator

        validator = WalkForwardValidator(
            initial_train_size=initial_train_size,
            test_size=test_size,
            step_size=step_size
        )

        scores = {}
        for name, config in model_configs.items():
            model_class = config["class"]
            kwargs = config.get("kwargs", {})

            print(f"[MetaLearner] {name} 모델 교차검증 중...")
            try:
                result = validator.validate(model_class, df, **kwargs)
                scores[name] = max(result.avg_hits, 0.01)  # 최소 0.01
                print(f"  → 평균 적중: {result.avg_hits:.3f}")
            except Exception as e:
                print(f"  → 오류: {e}, 기본값 사용")
                scores[name] = 0.8  # 무작위 기대값

        # Softmax 정규화
        weights = self._softmax_weights(scores)
        self.optimal_weights = weights

        # 캐시 저장
        self.save_cached_weights(weights)

        return weights

    def _softmax_weights(self, scores: Dict[str, float],
                         temperature: float = 1.0) -> Dict[str, float]:
        """
        성능 점수를 softmax로 정규화하여 가중치 산출.

        Args:
            scores: {모델명: 성능 점수}
            temperature: softmax 온도 (낮을수록 극단적)

        Returns:
            {모델명: 정규화된 가중치}
        """
        names = list(scores.keys())
        values = np.array([scores[n] for n in names])

        # softmax
        scaled = values / temperature
        exp_values = np.exp(scaled - np.max(scaled))
        softmax = exp_values / exp_values.sum()

        # 모델 수에 비례하도록 스케일링
        scaled_weights = softmax * len(names)

        return {name: float(w) for name, w in zip(names, scaled_weights)}

    def bayesian_model_averaging(self, model_probs: Dict[str, np.ndarray],
                                  model_scores: Dict[str, float]) -> np.ndarray:
        """
        베이지안 모델 평균화.
        P(number | data) = Σ P(number | model_k) × P(model_k | data)

        Args:
            model_probs: {모델명: 45차원 확률 벡터}
            model_scores: {모델명: 검증 점수 (log-likelihood 근사)}

        Returns:
            45차원 베이지안 평균 확률 벡터
        """
        if not model_probs:
            return np.ones(45) / 45

        # 모델 사후 확률: P(model_k | data) ∝ score_k
        names = list(model_probs.keys())
        scores = np.array([model_scores.get(n, 1.0) for n in names])

        # 점수를 양수로 보정
        scores = np.clip(scores, 0.01, None)
        model_posteriors = scores / scores.sum()

        # 베이지안 평균
        bma_probs = np.zeros(45)
        for i, name in enumerate(names):
            bma_probs += model_probs[name] * model_posteriors[i]

        # 정규화
        bma_probs = np.clip(bma_probs, 1e-10, None)
        return bma_probs / bma_probs.sum()

    def dynamic_adjust(self, current_weights: Dict[str, float],
                       recent_performance: Dict[str, List[int]],
                       alpha: float = 0.3) -> Dict[str, float]:
        """
        최근 실제 성능 기반 동적 가중치 조정.
        EMA(Exponential Moving Average) 방식으로 가중치를 업데이트합니다.

        Args:
            current_weights: 현재 가중치
            recent_performance: {모델명: [최근 적중 수 리스트]}
            alpha: EMA 계수 (0~1, 높을수록 최근에 민감)

        Returns:
            조정된 가중치
        """
        adjusted = {}

        for name, weight in current_weights.items():
            if name in recent_performance and recent_performance[name]:
                recent_avg = np.mean(recent_performance[name])
                # EMA 업데이트
                new_weight = alpha * recent_avg + (1 - alpha) * weight
                adjusted[name] = max(float(new_weight), 0.1)
            else:
                adjusted[name] = weight

        return adjusted

    def save_cached_weights(self, weights: Dict[str, float],
                            filepath: str = WEIGHTS_CACHE_FILE):
        """가중치를 JSON 파일로 캐싱"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(weights, f, ensure_ascii=False, indent=2)

    def load_cached_weights(self, filepath: str = WEIGHTS_CACHE_FILE) -> Optional[Dict[str, float]]:
        """캐시된 가중치 로드"""
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    self._cached_weights = json.load(f)
                    return self._cached_weights
            except (json.JSONDecodeError, IOError):
                return None
        return None

    def get_weights(self, model_configs: Dict[str, Dict[str, Any]] = None,
                    df: pd.DataFrame = None,
                    use_cache: bool = True) -> Dict[str, float]:
        """
        최적 가중치 반환.
        캐시가 있으면 캐시를 사용하고, 없으면 교차검증을 실행합니다.

        Args:
            model_configs: 모델 설정 (캐시 미스 시 필요)
            df: 데이터프레임 (캐시 미스 시 필요)
            use_cache: 캐시 사용 여부

        Returns:
            {모델명: 가중치}
        """
        if use_cache:
            cached = self.load_cached_weights()
            if cached:
                return cached

        if model_configs and df is not None:
            return self.cross_validation_weights(model_configs, df)

        return {}
