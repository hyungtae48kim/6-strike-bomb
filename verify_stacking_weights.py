"""
예측 피드백 루프 구멍 메우기 검증 스크립트.

검증 시나리오:
1. StackingEnsembleModel이 weights를 받아 분포에 반영하는가
2. update_hit_counts() 후 meta_weights_cache.json이 자동 갱신되는가
3. UltimateEnsembleModel._get_model_weight() 우선순위가 정확한가
"""
import os
import json
import shutil
import time
import numpy as np
import pandas as pd

from utils.history_manager import HistoryManager, HISTORY_FILE
from utils.fetcher import load_data
from utils.meta_learner import WEIGHTS_CACHE_FILE
from models.enums import AlgorithmType
from models.ultimate_ensemble_model import UltimateEnsembleModel


def test_stacking_weights_affect_distribution():
    """시나리오 1: weights가 Stacking 분포에 영향을 주는가"""
    print("\n[Test 1] Stacking weights → 분포 변화")

    from models.stacking_ensemble_model import StackingEnsembleModel

    df = load_data()
    if df.empty or len(df) < 250:
        print("  SKIP: 데이터 부족 (250회차 미만)")
        return

    weights_low = {alg.value: 0.1 for alg in AlgorithmType}
    weights_high_stats = dict(weights_low)
    weights_high_stats["Stats Based"] = 5.0

    np.random.seed(42)
    m1 = StackingEnsembleModel(weights=weights_low)
    m1.train(df)
    p1 = m1.get_probability_distribution()

    np.random.seed(42)
    m2 = StackingEnsembleModel(weights=weights_high_stats)
    m2.train(df)
    p2 = m2.get_probability_distribution()

    diff = float(np.abs(p1 - p2).sum())
    print(f"  분포 L1 차이: {diff:.6f}")
    assert diff > 1e-4, f"weights가 분포에 영향을 줘야 함 (diff={diff})"
    print("  PASS")


def test_cache_auto_refresh():
    """시나리오 2: update_hit_counts가 캐시를 자동 갱신"""
    print("\n[Test 2] meta_weights_cache.json 자동 갱신")

    history_backup = HISTORY_FILE + ".bak_test"
    cache_backup = WEIGHTS_CACHE_FILE + ".bak_test"

    if os.path.exists(HISTORY_FILE):
        shutil.copy(HISTORY_FILE, history_backup)
    if os.path.exists(WEIGHTS_CACHE_FILE):
        shutil.copy(WEIGHTS_CACHE_FILE, cache_backup)

    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        if os.path.exists(WEIGHTS_CACHE_FILE):
            os.remove(WEIGHTS_CACHE_FILE)

        manager = HistoryManager()
        manager.save_prediction(99999, AlgorithmType.STATS, [1, 2, 3, 4, 5, 6])

        before_exists = os.path.exists(WEIGHTS_CACHE_FILE)

        mock = pd.DataFrame([{
            "drwNo": 99999,
            "drwtNo1": 1, "drwtNo2": 2, "drwtNo3": 3,
            "drwtNo4": 4, "drwtNo5": 5, "drwtNo6": 6,
            "bnusNo": 7, "drwNoDate": "2099-01-01"
        }])
        manager.update_hit_counts(mock)

        after_exists = os.path.exists(WEIGHTS_CACHE_FILE)
        print(f"  업데이트 전 캐시 존재: {before_exists}, 후: {after_exists}")
        assert after_exists, "update_hit_counts 후 캐시가 생성되어야 함"

        with open(WEIGHTS_CACHE_FILE) as f:
            cache = json.load(f)
        print(f"  캐시 키 수: {len(cache)}, Stats Based 가중치: {cache.get('Stats Based')}")
        assert "Stats Based" in cache, "Stats Based가 캐시에 있어야 함"
        print("  PASS")

    finally:
        # 복원
        if os.path.exists(history_backup):
            shutil.move(history_backup, HISTORY_FILE)
        elif os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)

        if os.path.exists(cache_backup):
            shutil.move(cache_backup, WEIGHTS_CACHE_FILE)
        elif os.path.exists(WEIGHTS_CACHE_FILE):
            os.remove(WEIGHTS_CACHE_FILE)


def test_ultimate_priority_order():
    """시나리오 3: Ultimate _get_model_weight 우선순위"""
    print("\n[Test 3] Ultimate _get_model_weight 우선순위")

    # 1순위: self.weights
    m = UltimateEnsembleModel(weights={"Stats Based": 7.7})
    val = m._get_model_weight("Stats Based")
    print(f"  self.weights 우선순위: {val}")
    assert val == 7.7, f"self.weights가 1순위여야 함 (got {val})"

    # 3순위: default (캐시에도 없는 알고리즘)
    val_default = m._get_model_weight("Unknown Algorithm Xyz")
    print(f"  default fallback (Unknown): {val_default}")
    assert val_default == 1.0, f"미지의 알고리즘은 1.0 반환해야 함 (got {val_default})"

    # default_weights에 정의된 키 (self.weights에도 캐시에도 없는 케이스)
    m_empty = UltimateEnsembleModel(weights={})
    # Note: 캐시가 있으면 캐시 값이 반환됨. 캐시 부재 시 default 값.
    # 캐시 존재 여부와 무관하게 모델 키는 반환되어야 함.
    val_lstm = m_empty._get_model_weight("LSTM")
    print(f"  LSTM 가중치 (캐시 또는 default): {val_lstm}")
    assert val_lstm > 0, "LSTM 가중치는 양수여야 함"

    print("  PASS")


def test_stacking_falls_back_to_cache():
    """시나리오 4: weights=None이면 캐시에서 fallback 로드"""
    print("\n[Test 4] StackingEnsembleModel weights=None → 캐시 fallback")

    from models.stacking_ensemble_model import StackingEnsembleModel

    cache_backup = WEIGHTS_CACHE_FILE + ".bak_test4"
    if os.path.exists(WEIGHTS_CACHE_FILE):
        shutil.copy(WEIGHTS_CACHE_FILE, cache_backup)

    try:
        # 임시 캐시 생성
        os.makedirs(os.path.dirname(WEIGHTS_CACHE_FILE), exist_ok=True)
        test_cache = {"Stats Based": 3.3, "PageRank": 2.2}
        with open(WEIGHTS_CACHE_FILE, 'w') as f:
            json.dump(test_cache, f)

        m = StackingEnsembleModel(weights=None)
        print(f"  로드된 weights: {m.weights}")
        assert m.weights.get("Stats Based") == 3.3
        assert m.weights.get("PageRank") == 2.2
        print("  PASS")

    finally:
        if os.path.exists(cache_backup):
            shutil.move(cache_backup, WEIGHTS_CACHE_FILE)
        elif os.path.exists(WEIGHTS_CACHE_FILE):
            os.remove(WEIGHTS_CACHE_FILE)


if __name__ == "__main__":
    print("=" * 60)
    print("예측 피드백 루프 구멍 메우기 검증")
    print("=" * 60)

    test_ultimate_priority_order()
    test_stacking_falls_back_to_cache()
    test_cache_auto_refresh()
    test_stacking_weights_affect_distribution()

    print("\n" + "=" * 60)
    print("모든 테스트 통과")
    print("=" * 60)
