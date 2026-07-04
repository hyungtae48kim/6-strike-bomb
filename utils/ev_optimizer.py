"""EV(기대 수령액) 최적화 엔진.

당첨 확률은 모든 조합이 동일(1/8,145,060)하므로, '비인기 조합'을 골라
당첨 시 분배 인원을 줄여 기대 수령액을 최대화한다.
EV ∝ 1 / (1 + K * 인기도).
"""
from typing import List, Dict
import numpy as np
import pandas as pd
from utils.popularity_model import PopularityModel

# EV 지수 산출용 확산 상수(기대 동반당첨자 민감도). verify_ev로 보정 가능.
DEFAULT_K = 8.0


class EVOptimizer:
    def __init__(self, popularity_model: PopularityModel, k: float = DEFAULT_K,
                 seed: int = 42):
        self.pm = popularity_model
        self.k = k
        self.rng = np.random.default_rng(seed)
        self._ref_rng = np.random.default_rng(seed)  # reference mean 전용 RNG (Fix 1)
        self._p_bar = None  # 평균 인기도(기준선)

    def _random_combo(self) -> List[int]:
        return sorted(int(x) for x in
                      self._ref_rng.choice(range(1, 46), size=6, replace=False))

    def _reference_mean(self, n_sample: int = 2000) -> float:
        if self._p_bar is None:
            pops = [self.pm.popularity(self._random_combo()) for _ in range(n_sample)]
            self._p_bar = float(np.mean(pops))
        return self._p_bar

    def _ev_index_from_pop(self, p: float) -> float:
        """캐시된 인기도로 EV 지수 산출 (DRY 헬퍼, Fix 3)."""
        p_bar = self._reference_mean()
        return (1 + self.k * p_bar) / (1 + self.k * p)

    def ev_index(self, combo: List[int]) -> float:
        """평균 조합 대비 기대 수령액 배수. >1이면 유리."""
        return self._ev_index_from_pop(self.pm.popularity(combo))

    def _weighted_combo(self, aggressiveness: float) -> List[int]:
        base = np.ones(45)
        if aggressiveness > 0:  # 고번호(>31)에 가중 → 비인기 쪽 후보 확대
            for n in range(32, 46):
                base[n - 1] += aggressiveness * 1.5
        probs = base / base.sum()
        combo = self.rng.choice(range(1, 46), size=6, replace=False, p=probs)
        return sorted(int(x) for x in combo)

    def generate(self, n_tickets: int = 5, aggressiveness: float = 1.0,
                 pool_size: int = 20000, max_overlap: int = 3) -> List[Dict]:
        """비인기 조합 n_tickets세트 생성(다양성 보장).

        그리디 다양성 패스로 n_tickets를 채우지 못하면, 남은 슬롯을
        max_overlap 보장 없이 fallback으로 채운다(겹침이 더 클 수 있음).
        pool_size가 너무 작아 n_tickets를 채울 수 없으면 ValueError를 발생시킨다.
        """
        candidates = []
        seen = set()
        for _ in range(pool_size):
            combo = tuple(self._weighted_combo(aggressiveness))
            if combo in seen:
                continue
            seen.add(combo)
            candidates.append((self.pm.popularity(list(combo)), combo))
        candidates.sort(key=lambda x: x[0])  # 인기도 오름차순

        # 후보 인기도 캐시 — popularity() 중복 호출 방지 (Fix 3)
        pop_cache: Dict[tuple, float] = {combo: pop for pop, combo in candidates}

        selected: List[tuple] = []
        for _, combo in candidates:  # 그리디 다양성 선택
            if all(len(set(combo) & set(s)) <= max_overlap for s in selected):
                selected.append(combo)
            if len(selected) >= n_tickets:
                break
        i = 0  # 부족하면 겹침 제약 완화 (Fix 4: fallback은 max_overlap 미보장)
        while len(selected) < n_tickets and i < len(candidates):
            combo = candidates[i][1]
            if combo not in selected:
                selected.append(combo)
            i += 1

        if len(selected) < n_tickets:
            raise ValueError(
                f"요청한 {n_tickets}세트를 생성할 수 없습니다. pool_size를 늘리세요 "
                f"(could not generate {n_tickets} tickets from pool)"
            )

        results = []
        for combo in selected[:n_tickets]:
            pop = pop_cache[combo]
            results.append({
                "combo": list(combo),
                "popularity": round(pop, 4),
                "ev_index": round(self._ev_index_from_pop(pop), 4),
                "reasons": self.pm.reasons(list(combo)),
            })
        return results


def build_ev_rows(df: pd.DataFrame, n_tickets: int = 5,
                  aggressiveness: float = 1.0, pool_size: int = 20000,
                  weights: Dict[str, float] = None, seed: int = 42) -> List[Dict]:
    """app/스크립트용 헬퍼: df로 모델 구성 후 EV 조합 생성."""
    pm = PopularityModel(df, weights=weights)
    opt = EVOptimizer(pm, seed=seed)
    return opt.generate(n_tickets=n_tickets, aggressiveness=aggressiveness,
                        pool_size=pool_size)
