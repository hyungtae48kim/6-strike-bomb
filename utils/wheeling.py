"""
수학적 커버리지 보장 휠링 시스템.
N개 선정 번호에서 최소 티켓 수로 K-매치를 보장하는 조합 세트를 생성합니다.

휠링 시스템 종류:
- 완전 휠 (Full Wheel): N개에서 가능한 모든 C(N,6) 조합
- 축약 휠 (Abbreviated Wheel): 보장 조건을 만족하는 최소 조합 집합
"""

from itertools import combinations
from typing import List, Dict
import numpy as np


class WheelingSystem:
    """
    수학적 커버리지 보장 휠링 시스템.
    선정된 후보 번호에서 최적의 조합 세트를 생성합니다.
    """

    def __init__(self, selected_numbers: List[int],
                 guarantee_match: int = 3,
                 combination_size: int = 6):
        """
        Args:
            selected_numbers: 선정된 후보 번호 리스트 (예: [3, 7, 12, 18, 25, 31, 37, 42])
            guarantee_match: 보장할 최소 적중 수 (3=5등, 4=4등, 5=3등)
            combination_size: 한 조합의 크기 (6)
        """
        self.numbers = sorted(selected_numbers)
        self.n = len(self.numbers)
        self.k = guarantee_match
        self.combo_size = combination_size

        if self.n < self.combo_size:
            raise ValueError(f"후보 번호({self.n}개)가 조합 크기({self.combo_size})보다 적습니다.")
        if self.k > self.combo_size:
            raise ValueError(f"보장 적중({self.k})이 조합 크기({self.combo_size})보다 큽니다.")

    def generate_full_wheel(self) -> List[List[int]]:
        """
        완전 휠: N개에서 가능한 모든 6개 조합 생성.
        C(N, 6) 개의 조합을 반환합니다.

        주의: N이 크면 조합 수가 폭발적으로 증가합니다.
        예: N=15 → C(15,6) = 5,005조합
        """
        return [list(c) for c in combinations(self.numbers, self.combo_size)]

    def generate_abbreviated_wheel(self) -> List[List[int]]:
        """
        축약 휠: 보장 조건을 만족하는 최소 조합 집합.

        탐욕적 집합 커버 알고리즘:
        1. 커버해야 할 모든 k-부분집합 목록 생성
        2. 각 6개 조합이 커버하는 k-부분집합 계산
        3. 가장 많이 커버하는 조합을 반복 선택
        4. 모든 k-부분집합이 커버될 때까지 반복

        보장: 선정된 N개 번호 중 k개가 당첨이면,
              최소 하나의 티켓이 k개 이상 적중합니다.

        Returns:
            조합 리스트 (각 조합은 정렬된 6개 번호)
        """
        # 커버해야 할 모든 k-부분집합
        k_subsets = list(combinations(self.numbers, self.k))
        uncovered = set(range(len(k_subsets)))

        # 모든 가능한 6개 조합
        all_combos = list(combinations(self.numbers, self.combo_size))

        # 각 조합이 커버하는 k-부분집합 사전 계산
        combo_covers = {}
        for i, combo in enumerate(all_combos):
            combo_set = set(combo)
            covers = set()
            for j, ks in enumerate(k_subsets):
                if set(ks).issubset(combo_set):
                    covers.add(j)
            combo_covers[i] = covers

        # 탐욕적 선택
        selected_indices = []
        while uncovered:
            # 아직 커버되지 않은 부분집합을 가장 많이 커버하는 조합 선택
            best_idx = max(combo_covers.keys(),
                          key=lambda i: len(combo_covers[i] & uncovered))

            newly_covered = combo_covers[best_idx] & uncovered
            if not newly_covered:
                break  # 더 이상 커버할 수 없음

            selected_indices.append(best_idx)
            uncovered -= newly_covered

        return [list(all_combos[i]) for i in selected_indices]

    def get_coverage_report(self, wheel: List[List[int]]) -> Dict:
        """
        커버리지 분석 리포트 생성.

        Args:
            wheel: 생성된 휠 (조합 리스트)

        Returns:
            커버리지 분석 딕셔너리
        """
        # 모든 k-부분집합
        k_subsets = list(combinations(self.numbers, self.k))
        total_k_subsets = len(k_subsets)

        # 커버된 k-부분집합 계산
        covered = set()
        for combo in wheel:
            combo_set = set(combo)
            for j, ks in enumerate(k_subsets):
                if set(ks).issubset(combo_set):
                    covered.add(j)

        coverage_rate = len(covered) / total_k_subsets if total_k_subsets > 0 else 0

        # 보장 등수 매핑
        guarantee_prize = {3: "5등", 4: "4등", 5: "3등", 6: "2등/1등"}

        report = {
            "후보_번호": self.numbers,
            "후보_번호_수": self.n,
            "보장_적중": self.k,
            "보장_등수": guarantee_prize.get(self.k, f"{self.k}개 적중"),
            "총_티켓_수": len(wheel),
            "완전_휠_티켓_수": len(list(combinations(self.numbers, self.combo_size))),
            "절감_비율": f"{(1 - len(wheel) / max(len(list(combinations(self.numbers, self.combo_size))), 1)) * 100:.1f}%",
            "커버_부분집합": len(covered),
            "전체_부분집합": total_k_subsets,
            "커버리지": f"{coverage_rate * 100:.1f}%",
            "티켓_목록": wheel,
        }

        return report


def generate_optimal_wheel(top_numbers: List[int],
                           n_select: int = 10,
                           guarantee: int = 3) -> Dict:
    """
    편의 함수: 상위 번호에서 최적 휠 생성.

    Args:
        top_numbers: 확률 순 정렬된 번호 리스트
        n_select: 후보로 선정할 번호 수
        guarantee: 보장 적중 수

    Returns:
        커버리지 리포트 딕셔너리
    """
    selected = top_numbers[:n_select]
    ws = WheelingSystem(selected, guarantee_match=guarantee)
    wheel = ws.generate_abbreviated_wheel()
    return ws.get_coverage_report(wheel)
