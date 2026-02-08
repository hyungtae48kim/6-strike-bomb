"""
6-Strike-Bomb 보완 사항 전체 검증 스크립트.
7가지 보완 사항을 모두 테스트합니다.
"""

import sys
import traceback
import numpy as np
import pandas as pd

# 데이터 로드
from utils.fetcher import load_data

df = load_data()
if df.empty:
    print("ERROR: 데이터가 없습니다. 먼저 데이터를 다운로드하세요.")
    sys.exit(1)

print(f"데이터 로드 완료: {len(df)} 회차")
print("=" * 60)

results = {}


def test(name, func):
    """테스트 실행 헬퍼"""
    print(f"\n[테스트] {name}")
    print("-" * 40)
    try:
        func()
        results[name] = "PASS"
        print(f"  ✅ {name}: PASS")
    except Exception as e:
        results[name] = f"FAIL: {e}"
        print(f"  ❌ {name}: FAIL")
        traceback.print_exc()


# ============================================================
# 1. LottoAnalyzer 테스트
# ============================================================
def test_analyzer():
    from utils.analysis import LottoAnalyzer

    analyzer = LottoAnalyzer()

    # AC값 테스트: [1,2,3,4,5,6] → 15쌍, 차이값 {1,2,3,4,5} → 5 - 5 = 0... 아니다
    # 차이값: 1-2=1, 1-3=2, 1-4=3, 1-5=4, 1-6=5, 2-3=1, 2-4=2, 2-5=3, 2-6=4,
    #         3-4=1, 3-5=2, 3-6=3, 4-5=1, 4-6=2, 5-6=1
    # 고유 차이: {1,2,3,4,5} → 5개 → AC = 5-5 = 0
    ac = analyzer.ac_value([1, 2, 3, 4, 5, 6])
    assert ac == 0, f"AC([1,2,3,4,5,6]) = {ac}, expected 0"

    # [1,7,13,19,25,31]: 등차수열(간격 6)
    # 차이: 6,12,18,24,30, 6,12,18,24, 6,12,18, 6,12, 6
    # 고유: {6,12,18,24,30} → 5개 → AC = 5-5 = 0
    ac2 = analyzer.ac_value([1, 7, 13, 19, 25, 31])
    assert ac2 == 0, f"AC([1,7,13,19,25,31]) = {ac2}, expected 0"

    # [1,5,15,22,33,44]: 다양한 간격
    ac3 = analyzer.ac_value([1, 5, 15, 22, 33, 44])
    assert ac3 >= 5, f"AC should be high for diverse combo, got {ac3}"

    # 합계 테스트
    assert analyzer.sum_value([1, 2, 3, 4, 5, 6]) == 21
    assert analyzer.sum_range_check([20, 22, 25, 28, 30, 35])  # sum=160, 범위 내

    # 홀짝 테스트
    odd, even = analyzer.odd_even_ratio([1, 3, 5, 7, 9, 11])
    assert odd == 6 and even == 0

    # 연번 테스트
    assert analyzer.consecutive_count([1, 2, 3, 10, 20, 30]) == 2  # 1-2, 2-3

    # 번호대 분포 테스트
    decades = analyzer.decade_distribution([3, 15, 22, 31, 40, 44])
    assert decades["1-9"] == 1
    assert decades["40-45"] == 2

    # 종합 점수 테스트
    score = analyzer.comprehensive_score([3, 15, 22, 31, 38, 44])
    assert isinstance(score, float)

    print(f"  AC값 테스트 통과 (0, 0, {ac3})")
    print(f"  종합 점수 예시: {score:.1f}")


test("1. LottoAnalyzer", test_analyzer)


# ============================================================
# 2. CombinationFilter 테스트
# ============================================================
def test_combination_filter():
    from utils.analysis import CombinationFilter

    cf = CombinationFilter(df)

    # 역대 당첨번호는 대부분 필터를 통과해야 함
    pass_count = 0
    total = min(100, len(df))
    for _, row in df.tail(total).iterrows():
        combo = sorted([int(row[f'drwtNo{i}']) for i in range(1, 7)])
        if cf.filter(combo):
            pass_count += 1

    pass_rate = pass_count / total * 100
    assert pass_rate >= 70, f"역대 당첨번호 필터 통과율 {pass_rate:.1f}% < 70%"

    # 비정상 조합은 필터에 걸려야 함
    assert not cf.filter([1, 2, 3, 4, 5, 6])  # 연번 과다 + 합계 너무 낮음
    assert not cf.filter([40, 41, 42, 43, 44, 45])  # 합계 너무 높음

    # 필터링된 샘플링 테스트
    probs = np.ones(45) / 45
    combo = cf.filtered_sampling(probs)
    assert len(combo) == 6
    assert len(set(combo)) == 6
    assert all(1 <= n <= 45 for n in combo)

    print(f"  역대 당첨번호 필터 통과율: {pass_rate:.1f}%")
    print(f"  필터링 샘플링 결과: {combo}")


test("2. CombinationFilter", test_combination_filter)


# ============================================================
# 3. CombinationScorer 테스트
# ============================================================
def test_combination_scorer():
    from utils.combination_scorer import CombinationScorer

    cs = CombinationScorer(df)

    # 상관관계 행렬 차원 검증
    assert cs.correlation.shape == (45, 45), f"상관관계 행렬 크기: {cs.correlation.shape}"

    # 조건부 확률 차원 검증
    assert cs.cond_probs.shape == (45, 45), f"조건부 확률 크기: {cs.cond_probs.shape}"

    # 대각선은 0 (자기 자신 제외)
    for i in range(45):
        assert cs.cond_probs[i][i] == 0, f"cond_probs[{i}][{i}] != 0"

    # 조합 점수 테스트
    probs = np.ones(45) / 45
    score = cs.score_combination([1, 10, 20, 30, 40, 45], probs)
    assert isinstance(score, float) and np.isfinite(score)

    # 조건부 샘플링 테스트
    combo = cs.adjusted_sampling(probs)
    assert len(combo) == 6
    assert len(set(combo)) == 6
    assert all(1 <= n <= 45 for n in combo)
    assert combo == sorted(combo)

    print(f"  상관관계 행렬: {cs.correlation.shape}")
    print(f"  조건부 샘플링: {combo}")
    print(f"  조합 점수: {score:.4f}")


test("3. CombinationScorer", test_combination_scorer)


# ============================================================
# 4. Walk-Forward Validator 테스트
# ============================================================
def test_validation():
    from utils.validation import WalkForwardValidator
    from models.stats_model import StatsModel

    validator = WalkForwardValidator(
        initial_train_size=800,
        test_size=3,
        step_size=200
    )

    result = validator.validate(StatsModel, df)

    assert result.n_folds > 0, "검증 폴드 없음"
    assert 0 <= result.avg_hits <= 6, f"평균 적중 범위 오류: {result.avg_hits}"
    assert result.hit_distribution, "적중 분포 비어있음"

    # 과적합 탐지 테스트
    overfit = WalkForwardValidator.detect_overfit(result)
    assert "severity" in overfit
    assert "is_overfit" in overfit

    print(f"  폴드 수: {result.n_folds}")
    print(f"  평균 적중: {result.avg_hits:.3f}")
    print(f"  과적합 갭: {result.overfit_gap:.3f}")
    print(f"  과적합 진단: {overfit['severity']}")


test("4. Walk-Forward Validation", test_validation)


# ============================================================
# 5. LSTM 수정 테스트 (BCEWithLogitsLoss)
# ============================================================
def test_lstm_fix():
    from models.lstm_model import LSTMModel

    model = LSTMModel(epochs=10, seq_length=10)
    model.train(df)

    pred = model.predict()
    assert len(pred) == 6, f"예측 길이: {len(pred)}"
    assert len(set(pred)) == 6, "중복 번호"
    assert all(1 <= n <= 45 for n in pred), "범위 오류"

    probs = model.get_probability_distribution()
    assert probs.shape == (45,), f"확률 벡터 크기: {probs.shape}"
    assert abs(probs.sum() - 1.0) < 0.01, f"확률 합: {probs.sum()}"
    assert all(p >= 0 for p in probs), "음수 확률 존재"

    print(f"  예측: {pred}")
    print(f"  확률 합: {probs.sum():.6f}")


test("5. LSTM 수정 (BCEWithLogitsLoss)", test_lstm_fix)


# ============================================================
# 6. Transformer 수정 테스트
# ============================================================
def test_transformer_fix():
    from models.transformer_model import TransformerModel

    model = TransformerModel(epochs=10, seq_length=10)
    model.train(df)

    pred = model.predict()
    assert len(pred) == 6
    assert len(set(pred)) == 6
    assert all(1 <= n <= 45 for n in pred)

    probs = model.get_probability_distribution()
    assert probs.shape == (45,)
    assert abs(probs.sum() - 1.0) < 0.01
    assert all(p >= 0 for p in probs)

    print(f"  예측: {pred}")
    print(f"  확률 합: {probs.sum():.6f}")


test("6. Transformer 수정 (BCEWithLogitsLoss)", test_transformer_fix)


# ============================================================
# 7. DeepSets 모델 테스트
# ============================================================
def test_deepsets():
    from models.deepsets_model import DeepSetsModel

    model = DeepSetsModel(epochs=10, seq_length=10)
    model.train(df)

    pred = model.predict()
    assert len(pred) == 6, f"예측 길이: {len(pred)}"
    assert len(set(pred)) == 6, "중복 번호"
    assert all(1 <= n <= 45 for n in pred), "범위 오류"
    assert pred == sorted(pred), "정렬 안됨"

    probs = model.get_probability_distribution()
    assert probs.shape == (45,), f"확률 벡터 크기: {probs.shape}"
    assert abs(probs.sum() - 1.0) < 0.01, f"확률 합: {probs.sum()}"

    print(f"  예측: {pred}")
    print(f"  확률 합: {probs.sum():.6f}")


test("7. DeepSets 모델", test_deepsets)


# ============================================================
# 8. Wheeling System 테스트
# ============================================================
def test_wheeling():
    from utils.wheeling import WheelingSystem
    from itertools import combinations

    numbers = [3, 7, 12, 18, 25, 31, 37, 42, 44, 45]
    ws = WheelingSystem(numbers, guarantee_match=3)
    wheel = ws.generate_abbreviated_wheel()

    assert len(wheel) > 0, "휠이 비어있음"

    # 모든 3-부분집합이 커버되는지 검증
    all_3_subsets = list(combinations(numbers, 3))
    covered = set()
    for ticket in wheel:
        ticket_set = set(ticket)
        for j, subset in enumerate(all_3_subsets):
            if set(subset).issubset(ticket_set):
                covered.add(j)

    coverage = len(covered) / len(all_3_subsets) * 100
    assert coverage == 100, f"커버리지 {coverage:.1f}% < 100%"

    # 리포트 테스트
    report = ws.get_coverage_report(wheel)
    assert "총_티켓_수" in report
    assert "커버리지" in report

    full_wheel = ws.generate_full_wheel()
    saving = (1 - len(wheel) / len(full_wheel)) * 100

    print(f"  후보 번호: {numbers}")
    print(f"  축약 휠 티켓: {len(wheel)}장 (완전 휠 {len(full_wheel)}장 대비 {saving:.1f}% 절감)")
    print(f"  3-매치 커버리지: {coverage:.1f}%")


test("8. Wheeling System", test_wheeling)


# ============================================================
# 9. MetaLearner 테스트
# ============================================================
def test_meta_learner():
    from utils.meta_learner import MetaLearner
    from models.stats_model import StatsModel
    from models.bayes_model import BayesModel

    ml = MetaLearner()

    # softmax 가중치 테스트
    scores = {"A": 1.0, "B": 0.5, "C": 0.8}
    weights = ml._softmax_weights(scores)
    assert len(weights) == 3
    assert all(w > 0 for w in weights.values())

    # BMA 테스트
    probs_a = np.random.dirichlet(np.ones(45))
    probs_b = np.random.dirichlet(np.ones(45))
    model_probs = {"A": probs_a, "B": probs_b}
    model_scores = {"A": 1.5, "B": 0.8}

    bma = ml.bayesian_model_averaging(model_probs, model_scores)
    assert bma.shape == (45,), f"BMA 크기: {bma.shape}"
    assert abs(bma.sum() - 1.0) < 0.01, f"BMA 합: {bma.sum()}"

    # 캐시 저장/로드 테스트
    test_weights = {"Stats": 1.2, "Bayes": 0.8}
    ml.save_cached_weights(test_weights, "data/test_weights_cache.json")
    loaded = ml.load_cached_weights("data/test_weights_cache.json")
    assert loaded == test_weights

    # 정리
    import os
    os.remove("data/test_weights_cache.json")

    print(f"  softmax 가중치: {weights}")
    print(f"  BMA 합: {bma.sum():.6f}")


test("9. MetaLearner", test_meta_learner)


# ============================================================
# 10. Stacking Ensemble 테스트
# ============================================================
def test_stacking():
    from models.stacking_ensemble_model import StackingEnsembleModel

    model = StackingEnsembleModel(meta_model_type='ridge')
    model.train(df)

    pred = model.predict()
    assert len(pred) == 6, f"예측 길이: {len(pred)}"
    assert len(set(pred)) == 6, "중복 번호"
    assert all(1 <= n <= 45 for n in pred), "범위 오류"

    probs = model.get_probability_distribution()
    assert probs.shape == (45,), f"확률 벡터 크기: {probs.shape}"
    assert abs(probs.sum() - 1.0) < 0.01, f"확률 합: {probs.sum()}"

    # predict_multiple 테스트
    multi = model.predict_multiple(3)
    assert len(multi) == 3
    for m in multi:
        assert len(m) == 6

    # get_top_numbers 테스트
    top = model.get_top_numbers(10)
    assert len(top) == 10

    print(f"  예측: {pred}")
    print(f"  다중 예측: {len(multi)}세트")


test("10. Stacking Ensemble", test_stacking)


# ============================================================
# 11. Enum 업데이트 검증
# ============================================================
def test_enums():
    from models.enums import AlgorithmType

    assert hasattr(AlgorithmType, 'DEEPSETS'), "DEEPSETS enum 없음"
    assert hasattr(AlgorithmType, 'STACKING'), "STACKING enum 없음"
    assert AlgorithmType.DEEPSETS.value == "DeepSets"
    assert AlgorithmType.STACKING.value == "Stacking Ensemble"

    # 전체 enum 수 검증 (기존 12 + 신규 2 = 14)
    all_types = list(AlgorithmType)
    assert len(all_types) == 14, f"AlgorithmType 수: {len(all_types)}, expected 14"

    print(f"  총 AlgorithmType: {len(all_types)}")
    print(f"  DEEPSETS: {AlgorithmType.DEEPSETS.value}")
    print(f"  STACKING: {AlgorithmType.STACKING.value}")


test("11. Enum 업데이트", test_enums)


# ============================================================
# 결과 요약
# ============================================================
print("\n" + "=" * 60)
print("검증 결과 요약")
print("=" * 60)

pass_count = sum(1 for v in results.values() if v == "PASS")
total = len(results)

for name, result in results.items():
    icon = "✅" if result == "PASS" else "❌"
    print(f"  {icon} {name}: {result}")

print(f"\n총 {total}개 테스트 중 {pass_count}개 통과 ({pass_count/total*100:.0f}%)")

if pass_count == total:
    print("\n🎉 모든 보완 사항이 정상적으로 구현되었습니다!")
else:
    print(f"\n⚠️ {total - pass_count}개 테스트 실패. 위 오류를 확인하세요.")
    sys.exit(1)
