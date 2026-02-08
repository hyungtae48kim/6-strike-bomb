import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List
from .base_model import LottoModel


class LottoLSTM(nn.Module):
    """
    LSTM 기반 로또 번호 예측 신경망.
    시퀀스 입력을 받아 45차원 확률 벡터를 출력합니다.
    """

    def __init__(self, input_size=45, hidden_size=64, num_layers=2, dropout=0.2):
        super(LottoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 45)
            # Softmax 제거: BCEWithLogitsLoss 사용을 위해 raw logits 출력
        )

    def forward(self, x):
        # x: (batch, seq_len, 45)
        lstm_out, _ = self.lstm(x)
        # 마지막 시퀀스의 출력 사용
        last_output = lstm_out[:, -1, :]
        out = self.fc(last_output)
        return out


class LSTMModel(LottoModel):
    """
    LSTM 기반 시계열 로또 예측 모델.
    과거 N회차의 번호 시퀀스를 학습하여 다음 회차를 예측합니다.
    """

    def __init__(self, seq_length=20, hidden_size=64, num_layers=2, epochs=100, lr=0.001):
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self._probability_dist = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, df: pd.DataFrame):
        """데이터프레임에서 시퀀스 생성"""
        # 각 회차를 45차원 원-핫 벡터로 변환
        df_sorted = df.sort_values(by='drwNo', ascending=True)

        all_vectors = []
        for _, row in df_sorted.iterrows():
            vec = np.zeros(45)
            for i in range(1, 7):
                num = int(row[f'drwtNo{i}']) - 1  # 0-indexed
                vec[num] = 1.0
            all_vectors.append(vec)

        all_vectors = np.array(all_vectors)

        # 시퀀스 생성 (sliding window)
        X, y = [], []
        for i in range(len(all_vectors) - self.seq_length):
            X.append(all_vectors[i:i + self.seq_length])
            y.append(all_vectors[i + self.seq_length])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame):
        """LSTM 모델 학습"""
        print("LSTM 모델 학습 시작...")

        if len(df) < self.seq_length + 10:
            print(f"데이터 부족: 최소 {self.seq_length + 10}개 필요")
            self._probability_dist = np.ones(45) / 45
            return

        X, y = self._create_sequences(df)

        if len(X) == 0:
            self._probability_dist = np.ones(45) / 45
            return

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # 모델 초기화
        self.model = LottoLSTM(
            input_size=45,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        # 학습 루프
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)

            # 원본 이진 타겟 그대로 사용 (정규화 제거)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        print("LSTM 학습 완료.")

        # 마지막 시퀀스로 확률 분포 계산
        self._compute_probability_distribution(df)

    def _compute_probability_distribution(self, df: pd.DataFrame):
        """학습된 모델로 확률 분포 계산"""
        if self.model is None:
            self._probability_dist = np.ones(45) / 45
            return

        df_sorted = df.sort_values(by='drwNo', ascending=True)

        # 마지막 seq_length개의 시퀀스 생성
        last_vectors = []
        for _, row in df_sorted.tail(self.seq_length).iterrows():
            vec = np.zeros(45)
            for i in range(1, 7):
                num = int(row[f'drwtNo{i}']) - 1
                vec[num] = 1.0
            last_vectors.append(vec)

        last_seq = np.array([last_vectors])
        last_seq_tensor = torch.FloatTensor(last_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(last_seq_tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

        # 음수 방지 후 정규화
        probs = np.clip(probs, 1e-10, None)
        self._probability_dist = probs / probs.sum()

    def get_probability_distribution(self) -> np.ndarray:
        """45차원 확률 벡터 반환"""
        if self._probability_dist is None:
            return np.ones(45) / 45
        return self._probability_dist.copy()

    def predict(self) -> List[int]:
        """다음 회차 6개 번호 예측"""
        probs = self.get_probability_distribution()

        # 확률 기반 가중 샘플링
        numbers = list(range(1, 46))
        selected = np.random.choice(numbers, size=6, replace=False, p=probs)

        return sorted(selected.tolist())
