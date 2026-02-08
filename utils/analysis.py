"""
로또 번호 분석 유틸리티.
AC값, 끝수 분포, 합계 범위, 연번 제약 등 다양한 분석 기능 제공.
조합 필터링으로 비현실적 조합을 제거합니다.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from collections import Counter


class LottoAnalyzer:
    """
    로또 번호 조합 분석기.
    개별 조합의 품질을 다양한 지표로 평가합니다.
    """

    @staticmethod
    def ac_value(combo: List[int]) -> int:
        """
        AC값 (Arithmetic Complexity) 계산.
        6개 번호의 모든 쌍 차이값 중 고유한 값의 개수 - (n-1).
        C(6,2) = 15쌍, AC = unique_differences - 5.
        역대 당첨번호의 AC값은 주로 7-10 사이.
        """
        diffs = set()
        for i in range(len(combo)):
            for j in range(i + 1, len(combo)):
                diffs.add(abs(combo[i] - combo[j]))
        return len(diffs) - (len(combo) - 1)

    @staticmethod
    def last_digit_distribution(combo: List[int]) -> Dict[int, int]:
        """끝수 분포: 각 끝수(0-9)의 출현 횟수"""
        return dict(Counter(n % 10 for n in combo))

    @staticmethod
    def sum_value(combo: List[int]) -> int:
        """조합의 합계"""
        return sum(combo)

    @staticmethod
    def sum_range_check(combo: List[int], mean: float = 135.0, std: float = 35.0) -> bool:
        """합계 범위 필터: 평균 +-1 표준편차 내인지 확인"""
        s = sum(combo)
        return mean - std <= s <= mean + std

    @staticmethod
    def consecutive_count(combo: List[int]) -> int:
        """연번 쌍의 개수 (연속하는 번호 쌍 수)"""
        sorted_c = sorted(combo)
        return sum(1 for i in range(len(sorted_c) - 1)
                   if sorted_c[i + 1] - sorted_c[i] == 1)

    @staticmethod
    def odd_even_ratio(combo: List[int]) -> Tuple[int, int]:
        """홀짝 비율 (홀수 개수, 짝수 개수)"""
        odd = sum(1 for n in combo if n % 2 == 1)
        return (odd, 6 - odd)

    @staticmethod
    def high_low_ratio(combo: List[int], boundary: int = 23) -> Tuple[int, int]:
        """고저 비율 (저번호 개수, 고번호 개수). 경계: 1-22=저, 23-45=고"""
        low = sum(1 for n in combo if n <= boundary)
        return (low, 6 - low)

    @staticmethod
    def decade_distribution(combo: List[int]) -> Dict[str, int]:
        """번호대 분포 (1-9, 10-19, 20-29, 30-39, 40-45)"""
        ranges = {"1-9": 0, "10-19": 0, "20-29": 0, "30-39": 0, "40-45": 0}
        for n in combo:
            if n <= 9:
                ranges["1-9"] += 1
            elif n <= 19:
                ranges["10-19"] += 1
            elif n <= 29:
                ranges["20-29"] += 1
            elif n <= 39:
                ranges["30-39"] += 1
            else:
                ranges["40-45"] += 1
        return ranges

    def comprehensive_score(self, combo: List[int], df: pd.DataFrame = None) -> float:
        """
        종합 분석 점수.
        모든 분석 지표를 결합하여 조합의 '품질' 점수를 산출합니다.
        점수가 높을수록 역대 당첨번호 패턴에 부합합니다.
        """
        score = 0.0

        # AC값: 7-10이면 가산점 (역대 당첨번호의 주요 분포)
        ac = self.ac_value(combo)
        if 7 <= ac <= 10:
            score += 20
        elif ac == 6:
            score += 10
        else:
            score -= 10

        # 합계 범위: 100-170 사이 (역대 평균 ~135)
        s = self.sum_value(combo)
        if 100 <= s <= 170:
            score += 15
        elif 80 <= s <= 190:
            score += 5
        else:
            score -= 15

        # 연번: 0-2개면 정상 (대부분의 당첨번호)
        consec = self.consecutive_count(combo)
        if consec <= 2:
            score += 10
        else:
            score -= (consec - 2) * 10

        # 홀짝: 2:4, 3:3, 4:2가 가장 빈번 (전체의 ~80%)
        odd, even = self.odd_even_ratio(combo)
        if 2 <= odd <= 4:
            score += 10
        else:
            score -= 10

        # 고저: 2:4, 3:3, 4:2가 가장 빈번
        low, high = self.high_low_ratio(combo)
        if 2 <= low <= 4:
            score += 10
        else:
            score -= 10

        # 끝수: 동일 끝수 3개 이상이면 감점
        ld = self.last_digit_distribution(combo)
        max_same_digit = max(ld.values()) if ld else 0
        if max_same_digit >= 3:
            score -= 15
        elif max_same_digit <= 2:
            score += 5

        # 번호대 분포: 최소 3개 이상 번호대에 분산
        decades = self.decade_distribution(combo)
        active_decades = sum(1 for v in decades.values() if v > 0)
        if active_decades >= 4:
            score += 10
        elif active_decades >= 3:
            score += 5
        else:
            score -= 10

        return score


class CombinationFilter:
    """
    조합 필터: 역대 당첨번호 통계 기준에 부합하지 않는 조합을 제거합니다.
    predict() 후 후처리로 사용 가능합니다.
    """

    def __init__(self, df: pd.DataFrame):
        """
        역대 데이터로 통계 기준을 계산합니다.

        Args:
            df: 로또 히스토리 데이터프레임 (drwtNo1-6 컬럼 포함)
        """
        self.analyzer = LottoAnalyzer()
        self._compute_historical_stats(df)

    def _compute_historical_stats(self, df: pd.DataFrame):
        """역대 데이터에서 필터 기준 통계를 계산"""
        sums = []
        ac_values = []
        odd_counts = []

        for _, row in df.iterrows():
            nums = sorted([int(row[f'drwtNo{i}']) for i in range(1, 7)])
            sums.append(sum(nums))
            ac_values.append(self.analyzer.ac_value(nums))
            odd_counts.append(sum(1 for n in nums if n % 2 == 1))

        self.sum_mean = np.mean(sums)
        self.sum_std = np.std(sums)
        self.ac_min = int(np.percentile(ac_values, 5))   # 하위 5% 컷오프
        self.ac_max = int(np.percentile(ac_values, 95))   # 상위 95% 컷오프
        self.odd_mode = Counter(odd_counts).most_common(1)[0][0]

    def filter(self, combo: List[int]) -> bool:
        """
        이 조합이 필터를 통과하면 True 반환.
        역대 당첨번호 패턴에서 크게 벗어나는 조합을 제거합니다.
        """
        # AC값 범위 (보통 5-10)
        ac = self.analyzer.ac_value(combo)
        if ac < self.ac_min or ac > self.ac_max:
            return False

        # 합계 범위 (평균 +-1.5 표준편차)
        s = sum(combo)
        if s < self.sum_mean - 1.5 * self.sum_std or s > self.sum_mean + 1.5 * self.sum_std:
            return False

        # 연번 4개 이상 제외 (3쌍까지는 드물지만 발생)
        if self.analyzer.consecutive_count(combo) > 3:
            return False

        # 홀짝 극단 제외 (0:6 또는 6:0, 1:5 또는 5:1)
        odd, _ = self.analyzer.odd_even_ratio(combo)
        if odd < 1 or odd > 5:
            return False

        # 동일 끝수 4개 이상 제외 (3개까지는 실제로 발생)
        ld = self.analyzer.last_digit_distribution(combo)
        if max(ld.values()) >= 4:
            return False

        # 최소 2개 번호대에 분산 (한 번호대 집중은 매우 드물지만 2개는 가능)
        decades = self.analyzer.decade_distribution(combo)
        active_decades = sum(1 for v in decades.values() if v > 0)
        if active_decades < 2:
            return False

        return True

    def filtered_sampling(self, probs: np.ndarray, max_attempts: int = 100) -> List[int]:
        """
        필터를 통과하는 조합이 나올 때까지 반복 샘플링.

        Args:
            probs: 45차원 확률 벡터
            max_attempts: 최대 시도 횟수

        Returns:
            필터를 통과한 6개 번호 리스트 (정렬됨)
        """
        numbers = list(range(1, 46))

        for _ in range(max_attempts):
            selected = np.random.choice(numbers, size=6, replace=False, p=probs)
            combo = sorted(selected.tolist())
            if self.filter(combo):
                return combo

        # 최대 시도 초과 시 필터 없이 반환
        selected = np.random.choice(numbers, size=6, replace=False, p=probs)
        return sorted(selected.tolist())
