"""
DeepSets + GRU 기반 순서 불변(Permutation Invariant) 로또 예측 모델.
각 추첨을 순서 없는 집합으로 처리하여, LSTM/Transformer의 순서 가정 문제를 해결합니다.

핵심 아이디어:
- SetEncoder: 6개 번호를 임베딩 → phi 함수 적용 → 합산 (순서 불변)
- GRU: 시간적 패턴 학습 (회차 간 순서는 의미 있음)
- 출력: 45차원 로짓 → sigmoid로 각 번호 출현 확률
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import List
from .base_model import LottoModel


class SetEncoder(nn.Module):
    """
    DeepSets 기반 집합 인코더.
    순서 불변 함수: phi(x_i) → sum → rho
    6개 번호를 순서에 무관하게 하나의 벡터로 인코딩합니다.
    """

    def __init__(self, num_numbers: int = 45, embed_dim: int = 16, hidden_dim: int = 64):
        super(SetEncoder, self).__init__()
        # 번호 임베딩 (1-45 → embed_dim 차원)
        # 0은 패딩용으로 예약
        self.embedding = nn.Embedding(num_numbers + 1, embed_dim, padding_idx=0)

        # phi: 각 원소에 독립적으로 적용
        self.phi = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # rho: 합산된 표현에 적용
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        """
        Args:
            x: (batch, 6) - 6개 번호 인덱스 (1-45)
        Returns:
            (batch, hidden_dim) - 순서 불변 집합 표현
        """
        embedded = self.embedding(x)        # (batch, 6, embed_dim)
        phi_out = self.phi(embedded)         # (batch, 6, hidden_dim)
        pooled = phi_out.sum(dim=1)          # (batch, hidden_dim) - 순서 불변!
        return self.rho(pooled)              # (batch, hidden_dim)


class DeepSetsTemporalNet(nn.Module):
    """
    DeepSets + GRU 시간적 모델.
    각 추첨을 SetEncoder로 인코딩한 후, GRU로 시간적 패턴을 학습합니다.
    """

    def __init__(self, num_numbers: int = 45, embed_dim: int = 16,
                 set_hidden: int = 64, gru_hidden: int = 128,
                 num_gru_layers: int = 2, dropout: float = 0.2):
        super(DeepSetsTemporalNet, self).__init__()

        self.set_encoder = SetEncoder(num_numbers, embed_dim, set_hidden)

        self.gru = nn.GRU(
            input_size=set_hidden,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )

        self.output = nn.Sequential(
            nn.Linear(gru_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_numbers)
            # raw logits 출력 (BCEWithLogitsLoss 사용)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 6) - 시퀀스의 각 시점에서 6개 번호
        Returns:
            (batch, 45) - 각 번호의 출현 로짓
        """
        batch, seq_len, _ = x.shape

        # 각 시점의 번호 집합을 인코딩
        x_flat = x.reshape(batch * seq_len, 6)      # (batch*seq_len, 6)
        encoded = self.set_encoder(x_flat)            # (batch*seq_len, set_hidden)
        encoded = encoded.reshape(batch, seq_len, -1) # (batch, seq_len, set_hidden)

        # GRU로 시간적 패턴 학습
        gru_out, _ = self.gru(encoded)               # (batch, seq_len, gru_hidden)
        last_output = gru_out[:, -1, :]              # (batch, gru_hidden)

        return self.output(last_output)               # (batch, 45)


class DeepSetsModel(LottoModel):
    """
    DeepSets + GRU 기반 로또 예측 모델.
    순서 불변 집합 인코딩으로 각 추첨을 표현하고,
    GRU로 시간적 패턴을 학습하여 다음 회차를 예측합니다.
    """

    def __init__(self, seq_length: int = 20, embed_dim: int = 16,
                 set_hidden: int = 64, gru_hidden: int = 128,
                 epochs: int = 100, lr: float = 0.001):
        self.seq_length = seq_length
        self.embed_dim = embed_dim
        self.set_hidden = set_hidden
        self.gru_hidden = gru_hidden
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self._probability_dist = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _create_sequences(self, df: pd.DataFrame):
        """
        데이터프레임에서 시퀀스 생성.
        각 추첨을 6개 번호 인덱스(1-45)로 표현합니다.
        (45-dim 바이너리 벡터가 아님)
        """
        df_sorted = df.sort_values(by='drwNo', ascending=True)

        # 각 회차를 6개 번호로 표현
        all_draws = []
        for _, row in df_sorted.iterrows():
            nums = sorted([int(row[f'drwtNo{i}']) for i in range(1, 7)])
            all_draws.append(nums)

        all_draws = np.array(all_draws)  # (total_draws, 6)

        # 타겟: 45-dim 이진 벡터
        all_targets = []
        for _, row in df_sorted.iterrows():
            target = np.zeros(45)
            for i in range(1, 7):
                target[int(row[f'drwtNo{i}']) - 1] = 1.0
            all_targets.append(target)
        all_targets = np.array(all_targets)

        # 시퀀스 생성 (sliding window)
        X, y = [], []
        for i in range(len(all_draws) - self.seq_length):
            X.append(all_draws[i:i + self.seq_length])
            y.append(all_targets[i + self.seq_length])

        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame):
        """DeepSets + GRU 모델 학습"""
        print("DeepSets 모델 학습 시작...")

        if len(df) < self.seq_length + 10:
            print(f"데이터 부족: 최소 {self.seq_length + 10}개 필요")
            self._probability_dist = np.ones(45) / 45
            return

        X, y = self._create_sequences(df)

        if len(X) == 0:
            self._probability_dist = np.ones(45) / 45
            return

        # LongTensor (임베딩 인덱스용)
        X_tensor = torch.LongTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # 모델 초기화
        self.model = DeepSetsTemporalNet(
            num_numbers=45,
            embed_dim=self.embed_dim,
            set_hidden=self.set_hidden,
            gru_hidden=self.gru_hidden
        ).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        # 학습 루프
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch [{epoch+1}/{self.epochs}], Loss: {loss.item():.4f}")

        print("DeepSets 학습 완료.")

        # 확률 분포 계산
        self._compute_probability_distribution(df)

    def _compute_probability_distribution(self, df: pd.DataFrame):
        """학습된 모델로 확률 분포 계산"""
        if self.model is None:
            self._probability_dist = np.ones(45) / 45
            return

        df_sorted = df.sort_values(by='drwNo', ascending=True)

        # 마지막 seq_length개의 추첨 시퀀스 생성
        last_draws = []
        for _, row in df_sorted.tail(self.seq_length).iterrows():
            nums = sorted([int(row[f'drwtNo{i}']) for i in range(1, 7)])
            last_draws.append(nums)

        last_seq = np.array([last_draws])
        last_seq_tensor = torch.LongTensor(last_seq).to(self.device)

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
