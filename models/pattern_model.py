import numpy as np
import pandas as pd
from typing import List, Dict
from collections import Counter
from .base_model import LottoModel

# scipy가 있으면 FFT 사용, 없으면 대체 로직
try:
    from scipy.fft import fft
    from scipy.signal import find_peaks
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PatternModel(LottoModel):
    """
    패턴 분석 기반 로또 예측 모델.
    주기성, 간격, 합계, 홀짝 비율 등 다양한 패턴을 분석합니다.
    """

    def __init__(self, recent_draws=100):
        self.recent_draws = recent_draws
        self.number_gaps = {}  # 각 번호의 평균 출현 간격
        self.number_due = {}   # 각 번호가 "나올 때가 된" 정도
        self.sum_pattern = {}  # 합계 패턴
        self.odd_even_pattern = None
        self.high_low_pattern = None
        self._probability_dist = None

    def _analyze_gaps(self, df: pd.DataFrame):
        """각 번호의 출현 간격 분석"""
        df_sorted = df.sort_values(by='drwNo', ascending=True)

        # 각 번호의 마지막 출현 회차
        last_seen = {i: 0 for i in range(1, 46)}
        gaps = {i: [] for i in range(1, 46)}

        for _, row in df_sorted.iterrows():
            draw_no = int(row['drwNo'])
            nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]

            for num in nums:
                if last_seen[num] > 0:
                    gap = draw_no - last_seen[num]
                    gaps[num].append(gap)
                last_seen[num] = draw_no

        # 평균 간격 계산
        current_draw = int(df_sorted.iloc[-1]['drwNo'])
        for num in range(1, 46):
            if gaps[num]:
                avg_gap = np.mean(gaps[num])
                self.number_gaps[num] = avg_gap

                # "나올 때가 된" 정도 계산
                draws_since = current_draw - last_seen[num]
                self.number_due[num] = draws_since / avg_gap if avg_gap > 0 else 1.0
            else:
                self.number_gaps[num] = 10
                self.number_due[num] = 1.0

    def _analyze_sum_pattern(self, df: pd.DataFrame):
        """당첨 번호 합계 패턴 분석"""
        sums = []
        for _, row in df.iterrows():
            total = sum(int(row[f'drwtNo{i}']) for i in range(1, 7))
            sums.append(total)

        # 합계 분포의 평균과 표준편차
        self.sum_pattern = {
            'mean': np.mean(sums),
            'std': np.std(sums),
            'min': min(sums),
            'max': max(sums)
        }

    def _analyze_odd_even(self, df: pd.DataFrame):
        """홀짝 비율 분석"""
        df_sorted = df.sort_values(by='drwNo', ascending=False)
        recent_df = df_sorted.head(self.recent_draws)

        odd_counts = []
        for _, row in recent_df.iterrows():
            nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]
            odd_count = sum(1 for n in nums if n % 2 == 1)
            odd_counts.append(odd_count)

        self.odd_even_pattern = {
            'avg_odd': np.mean(odd_counts),
            'most_common': Counter(odd_counts).most_common(1)[0][0]
        }

    def _analyze_high_low(self, df: pd.DataFrame):
        """고저 비율 분석 (1-22 vs 23-45)"""
        df_sorted = df.sort_values(by='drwNo', ascending=False)
        recent_df = df_sorted.head(self.recent_draws)

        low_counts = []
        for _, row in recent_df.iterrows():
            nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]
            low_count = sum(1 for n in nums if n <= 22)
            low_counts.append(low_count)

        self.high_low_pattern = {
            'avg_low': np.mean(low_counts),
            'most_common': Counter(low_counts).most_common(1)[0][0]
        }

    def _analyze_periodicity(self, df: pd.DataFrame):
        """주기성 분석 (FFT 사용)"""
        if not HAS_SCIPY:
            return {}

        df_sorted = df.sort_values(by='drwNo', ascending=True)

        periodicity = {}
        for num in range(1, 46):
            # 각 회차에서 해당 번호 출현 여부 (이진 시계열)
            appearances = []
            for _, row in df_sorted.iterrows():
                nums = [int(row[f'drwtNo{i}']) for i in range(1, 7)]
                appearances.append(1 if num in nums else 0)

            appearances = np.array(appearances)

            if len(appearances) > 10:
                # FFT 수행
                fft_result = fft(appearances)
                power = np.abs(fft_result) ** 2

                # 주요 주기 찾기
                peaks, _ = find_peaks(power[:len(power)//2], height=np.mean(power))
                if len(peaks) > 0:
                    dominant_period = len(appearances) / peaks[0] if peaks[0] > 0 else None
                    periodicity[num] = dominant_period

        return periodicity

    def train(self, df: pd.DataFrame):
        """패턴 분석 수행"""
        print("Pattern 모델 학습 시작...")

        self._analyze_gaps(df)
        self._analyze_sum_pattern(df)
        self._analyze_odd_even(df)
        self._analyze_high_low(df)

        # 확률 분포 계산
        self._compute_probability_distribution(df)

        print("Pattern 학습 완료.")

    def _compute_probability_distribution(self, df: pd.DataFrame):
        """패턴 기반 확률 분포 계산"""
        probs = np.zeros(45)

        # 1. 출현 간격 기반 확률 (나올 때가 된 번호에 높은 확률)
        due_scores = np.array([self.number_due.get(i, 1.0) for i in range(1, 46)])
        due_scores = np.clip(due_scores, 0.1, 3.0)  # 극단값 제한

        # 2. 최근 빈도 기반 확률
        df_sorted = df.sort_values(by='drwNo', ascending=False)
        recent_df = df_sorted.head(self.recent_draws)

        freq = np.zeros(45)
        for _, row in recent_df.iterrows():
            for i in range(1, 7):
                num = int(row[f'drwtNo{i}'])
                freq[num - 1] += 1

        freq = freq / max(freq.sum(), 1)

        # 3. 합계 패턴 기반 조정
        # 이상적인 합계 범위의 번호 조합에 보너스
        ideal_sum = self.sum_pattern.get('mean', 135)

        # 종합 확률 계산
        # - 간격 기반 50%
        # - 빈도 기반 30%
        # - 기본 확률 20%
        probs = due_scores * 0.5 + freq * 0.3 + np.ones(45) * 0.2 / 45

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
