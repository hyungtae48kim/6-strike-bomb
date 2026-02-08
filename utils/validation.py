"""
Walk-Forward 시간적 교차검증 시스템.
모델의 실제 예측력을 측정하고 과적합을 탐지합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class FoldResult:
    """개별 폴드(시점) 결과"""
    draw_no: int
    hits: int
    prediction: List[int]
    actual: List[int] = field(default_factory=list)


@dataclass
class ValidationResult:
    """모델 검증 결과 요약"""
    model_name: str
    avg_hits: float
    std_hits: float
    hit_distribution: Dict[int, int]
    train_avg_hits: float
    test_avg_hits: float
    fold_results: List[FoldResult]
    n_folds: int = 0

    @property
    def overfit_gap(self) -> float:
        """과적합 지표: 학습 적중률 - 테스트 적중률"""
        return self.train_avg_hits - self.test_avg_hits


class WalkForwardValidator:
    """
    Walk-Forward 시간적 교차검증기.
    과거 데이터의 일부로 학습 → 미래 데이터로 테스트를 반복합니다.
    """

    def __init__(self, initial_train_size: int = 500,
                 test_size: int = 10,
                 step_size: int = 50):
        """
        Args:
            initial_train_size: 최초 학습 세트 크기 (회차 수)
            test_size: 각 폴드의 테스트 세트 크기
            step_size: 슬라이딩 윈도우 이동 크기
        """
        self.initial_train_size = initial_train_size
        self.test_size = test_size
        self.step_size = step_size

    def validate(self, model_class, df: pd.DataFrame,
                 **model_kwargs) -> ValidationResult:
        """
        단일 모델의 Walk-Forward 검증 수행.

        Args:
            model_class: 모델 클래스 (예: StatsModel)
            df: 전체 로또 히스토리 데이터프레임
            **model_kwargs: 모델 생성자 인자 (예: epochs=10)

        Returns:
            ValidationResult
        """
        df_sorted = df.sort_values('drwNo').reset_index(drop=True)
        total_draws = len(df_sorted)

        fold_results = []
        train_hits_list = []

        model_name = model_class.__name__

        for start in range(self.initial_train_size,
                           total_draws - self.test_size + 1,
                           self.step_size):
            train_df = df_sorted.iloc[:start]
            test_df = df_sorted.iloc[start:start + self.test_size]

            try:
                model = model_class(**model_kwargs)
                model.train(train_df)
            except Exception as e:
                print(f"[{model_name}] 학습 오류 (train_size={start}): {e}")
                continue

            # 학습 세트에서의 적중 수 (과적합 비교용)
            train_test_df = train_df.tail(self.test_size)
            for _, row in train_test_df.iterrows():
                try:
                    pred = model.predict()
                    actual = {int(row[f'drwtNo{i}']) for i in range(1, 7)}
                    hits = len(set(pred) & actual)
                    train_hits_list.append(hits)
                except Exception:
                    pass

            # 테스트 세트에서의 적중 수
            for _, row in test_df.iterrows():
                try:
                    pred = model.predict()
                    actual_set = {int(row[f'drwtNo{i}']) for i in range(1, 7)}
                    actual_list = sorted(list(actual_set))
                    hits = len(set(pred) & actual_set)

                    fold_results.append(FoldResult(
                        draw_no=int(row['drwNo']),
                        hits=hits,
                        prediction=pred,
                        actual=actual_list
                    ))
                except Exception:
                    pass

        if not fold_results:
            return ValidationResult(
                model_name=model_name,
                avg_hits=0.0,
                std_hits=0.0,
                hit_distribution={},
                train_avg_hits=0.0,
                test_avg_hits=0.0,
                fold_results=[],
                n_folds=0
            )

        test_hits = [fr.hits for fr in fold_results]
        hit_dist = dict(Counter(test_hits))

        return ValidationResult(
            model_name=model_name,
            avg_hits=float(np.mean(test_hits)),
            std_hits=float(np.std(test_hits)),
            hit_distribution=hit_dist,
            train_avg_hits=float(np.mean(train_hits_list)) if train_hits_list else 0.0,
            test_avg_hits=float(np.mean(test_hits)),
            fold_results=fold_results,
            n_folds=len(fold_results)
        )

    def validate_all_models(self, df: pd.DataFrame,
                             model_configs: Dict[str, Dict[str, Any]]
                             ) -> Dict[str, ValidationResult]:
        """
        여러 모델을 한 번에 검증합니다.

        Args:
            df: 로또 히스토리 데이터프레임
            model_configs: {모델명: {"class": ModelClass, "kwargs": {...}}}

        Returns:
            {모델명: ValidationResult}
        """
        results = {}

        for name, config in model_configs.items():
            model_class = config["class"]
            kwargs = config.get("kwargs", {})

            print(f"\n[검증] {name} 모델...")
            result = self.validate(model_class, df, **kwargs)
            results[name] = result
            print(f"  → 평균 적중: {result.avg_hits:.3f}, "
                  f"과적합 갭: {result.overfit_gap:.3f}")

        return results

    @staticmethod
    def detect_overfit(result: ValidationResult) -> Dict[str, Any]:
        """
        과적합 탐지 리포트 생성.

        Args:
            result: ValidationResult

        Returns:
            과적합 분석 딕셔너리
        """
        gap = result.overfit_gap

        report = {
            "model_name": result.model_name,
            "train_avg_hits": result.train_avg_hits,
            "test_avg_hits": result.test_avg_hits,
            "overfit_gap": gap,
            "n_folds": result.n_folds,
            "is_overfit": gap > 0.3,  # 갭이 0.3 이상이면 과적합 의심
            "severity": "없음"
        }

        if gap > 0.5:
            report["severity"] = "심각"
            report["recommendation"] = "모델 복잡도를 줄이거나 정규화를 강화하세요."
        elif gap > 0.3:
            report["severity"] = "경미"
            report["recommendation"] = "학습 데이터 크기를 늘리거나 조기 종료를 적용하세요."
        else:
            report["severity"] = "없음"
            report["recommendation"] = "과적합 징후가 없습니다."

        return report
