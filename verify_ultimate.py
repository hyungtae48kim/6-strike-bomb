#!/usr/bin/env python3
"""
Ultimate Ensemble 모델 테스트 스크립트
모든 신규 모델의 기본 작동을 검증합니다.
"""

import sys
import pandas as pd
import numpy as np

# 경로 설정
sys.path.insert(0, '/home/hyungtae48kim/project/6-strike-bomb')

from utils.fetcher import load_data
from models.enums import AlgorithmType


def test_model(model_class, model_name, df, **kwargs):
    """개별 모델 테스트"""
    print(f"\n{'='*50}")
    print(f"테스트: {model_name}")
    print('='*50)

    try:
        # 모델 생성
        model = model_class(**kwargs) if kwargs else model_class()
        print(f"✓ 모델 생성 완료")

        # 학습
        model.train(df)
        print(f"✓ 학습 완료")

        # 예측
        prediction = model.predict()
        print(f"✓ 예측 완료: {prediction}")

        # 확률 분포
        probs = model.get_probability_distribution()
        print(f"✓ 확률 분포: shape={probs.shape}, sum={probs.sum():.4f}")

        # 검증
        assert len(prediction) == 6, "예측은 6개 번호여야 합니다"
        assert len(set(prediction)) == 6, "중복 번호가 없어야 합니다"
        assert all(1 <= n <= 45 for n in prediction), "번호는 1-45 범위여야 합니다"
        assert probs.shape == (45,), "확률 분포는 45차원이어야 합니다"
        assert abs(probs.sum() - 1.0) < 0.01, "확률 합은 1이어야 합니다"

        print(f"✓ 모든 검증 통과!")
        return True

    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Ultimate Ensemble 모델 테스트 시작")
    print("="*50)

    # 데이터 로드
    df = load_data()
    if df.empty:
        print("데이터가 없습니다. 먼저 데이터를 업데이트하세요.")
        return

    print(f"데이터 로드 완료: {len(df)}개 회차")

    # 테스트할 모델들
    results = {}

    # 1. 기존 모델 테스트
    from models.stats_model import StatsModel
    results['StatsModel'] = test_model(StatsModel, "StatsModel", df)

    from models.bayes_model import BayesModel
    results['BayesModel'] = test_model(BayesModel, "BayesModel", df)

    from models.gnn_model import GNNModel
    results['GNNModel'] = test_model(GNNModel, "GNNModel", df)

    # 2. 신규 모델 테스트
    from models.lstm_model import LSTMModel
    results['LSTMModel'] = test_model(LSTMModel, "LSTMModel", df, epochs=10)

    from models.transformer_model import TransformerModel
    results['TransformerModel'] = test_model(TransformerModel, "TransformerModel", df, epochs=10)

    from models.pagerank_model import PageRankModel
    results['PageRankModel'] = test_model(PageRankModel, "PageRankModel", df)

    from models.community_model import CommunityModel
    results['CommunityModel'] = test_model(CommunityModel, "CommunityModel", df)

    from models.markov_model import MarkovModel
    results['MarkovModel'] = test_model(MarkovModel, "MarkovModel", df)

    from models.pattern_model import PatternModel
    results['PatternModel'] = test_model(PatternModel, "PatternModel", df)

    from models.montecarlo_model import MonteCarloModel
    results['MonteCarloModel'] = test_model(MonteCarloModel, "MonteCarloModel", df, n_simulations=1000)

    # 3. Ultimate Ensemble 테스트
    from models.ultimate_ensemble_model import UltimateEnsembleModel
    print(f"\n{'='*50}")
    print("테스트: UltimateEnsembleModel (모든 모델 통합)")
    print('='*50)

    try:
        model = UltimateEnsembleModel()
        print("✓ 모델 생성 완료")

        model.train(df)
        print("✓ 학습 완료")

        prediction = model.predict()
        print(f"✓ 예측 완료: {prediction}")

        probs = model.get_probability_distribution()
        print(f"✓ 확률 분포: shape={probs.shape}, sum={probs.sum():.4f}")

        # 상위 번호
        top_nums = model.get_top_numbers(10)
        print(f"✓ 상위 10개 번호: {[(n, f'{p:.3f}') for n, p in top_nums]}")

        # 다중 예측
        multi_preds = model.predict_multiple(3)
        print(f"✓ 다중 예측 (3세트): {multi_preds}")

        results['UltimateEnsembleModel'] = True
        print("✓ 모든 검증 통과!")

    except Exception as e:
        print(f"✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        results['UltimateEnsembleModel'] = False

    # 결과 요약
    print("\n" + "="*50)
    print("테스트 결과 요약")
    print("="*50)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "✓ 통과" if success else "✗ 실패"
        print(f"  {name}: {status}")

    print(f"\n총 {passed}/{total} 모델 테스트 통과")

    if passed == total:
        print("\n🎉 모든 모델이 정상 작동합니다!")
    else:
        print("\n⚠️ 일부 모델에서 오류가 발생했습니다.")


if __name__ == "__main__":
    main()
