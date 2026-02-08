import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
from typing import List
from .base_model import LottoModel


class PositionalEncoding(nn.Module):
    """Transformer를 위한 위치 인코딩"""

    def __init__(self, d_model, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LottoTransformer(nn.Module):
    """
    Transformer 기반 로또 번호 예측 신경망.
    Self-Attention을 통해 시퀀스 패턴을 학습합니다.
    """

    def __init__(self, input_size=45, d_model=64, nhead=4, num_layers=2, dropout=0.1):
        super(LottoTransformer, self).__init__()

        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 45)
            # Softmax 제거: BCEWithLogitsLoss 사용을 위해 raw logits 출력
        )

    def forward(self, x):
        # x: (batch, seq_len, 45)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # 마지막 토큰의 출력 사용
        x = x[:, -1, :]
        out = self.fc(x)
        return out


class TransformerModel(LottoModel):
    """
    Transformer 기반 시계열 로또 예측 모델.
    Self-Attention 메커니즘으로 중요한 패턴을 자동으로 포착합니다.
    """

    def __init__(self, seq_length=20, d_model=64, nhead=4, num_layers=2, epochs=100, lr=0.001):
        self.seq_length = seq_length
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self._probability_dist = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, df: pd.DataFrame):
        """데이터프레임에서 시퀀스 생성"""
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
        """Transformer 모델 학습"""
        print("Transformer 모델 학습 시작...")

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
        self.model = LottoTransformer(
            input_size=45,
            d_model=self.d_model,
            nhead=self.nhead,
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

        print("Transformer 학습 완료.")

        # 확률 분포 계산
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

        numbers = list(range(1, 46))
        selected = np.random.choice(numbers, size=6, replace=False, p=probs)

        return sorted(selected.tolist())
