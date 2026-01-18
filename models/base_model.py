from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from typing import List

class LottoModel(ABC):
    """
    로또 예측 모델의 기본 추상 클래스.
    모든 예측 모델은 이 클래스를 상속받아야 합니다.
    """

    @abstractmethod
    def train(self, df: pd.DataFrame):
        """
        모델 학습.
        df: drwtNo1..drwtNo6, bnusNo, drwNoDate 등의 컬럼을 포함하는 DataFrame
        """
        pass

    @abstractmethod
    def predict(self) -> List[int]:
        """
        다음 회차의 6개 번호를 예측합니다.
        Returns: 1-45 범위의 정수 6개 리스트 (정렬됨)
        """
        pass

    def get_probability_distribution(self) -> np.ndarray:
        """
        각 번호(1-45)의 선택 확률 분포를 반환합니다.
        Returns: 45차원 확률 벡터 (합계 = 1.0)

        기본 구현은 균등 분포를 반환합니다.
        각 모델은 이 메서드를 오버라이드하여 학습된 확률 분포를 반환해야 합니다.
        """
        return np.ones(45) / 45  # 균등 분포
