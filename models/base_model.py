from abc import ABC, abstractmethod
import pandas as pd
from typing import List

class LottoModel(ABC):
    @abstractmethod
    def train(self, df: pd.DataFrame):
        """
        Train the model using the provided dataframe.
        df should have columns: drwtNo1..drwtNo6, bnusNo, drwNoDate, etc.
        """
        pass

    @abstractmethod
    def predict(self) -> List[int]:
        """
        Predict the next 6 numbers.
        Returns a list of 6 integers between 1 and 45.
        """
        pass
