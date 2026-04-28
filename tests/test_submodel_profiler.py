"""SubmodelProfiler 단위 테스트."""
import numpy as np
import pytest

from validator.submodel_profiler import (
    SubmodelProfiler,
    score_top6_hits,
    top6_from_probs,
)


def test_top6_from_probs_picks_highest_six_oneindexed():
    """45-dim probs에서 상위 6개를 1-indexed로 반환."""
    probs = np.zeros(45)
    probs[0] = 0.20  # 번호 1
    probs[10] = 0.15  # 번호 11
    probs[20] = 0.10  # 번호 21
    probs[30] = 0.08  # 번호 31
    probs[40] = 0.06  # 번호 41
    probs[5] = 0.05  # 번호 6
    probs[15] = 0.01  # 번호 16 (탈락 예상)
    top6 = top6_from_probs(probs)
    assert sorted(top6) == sorted([1, 11, 21, 31, 41, 6])


def test_score_top6_hits_returns_intersection_size():
    """top-6와 actual의 교집합 크기를 반환."""
    top6 = [1, 5, 10, 15, 20, 25]
    actual = [1, 5, 30, 35, 40, 45]
    assert score_top6_hits(top6, actual) == 2


def test_score_top6_hits_zero_when_disjoint():
    """겹치지 않으면 0."""
    assert score_top6_hits([1, 2, 3, 4, 5, 6], [40, 41, 42, 43, 44, 45]) == 0


def test_profiler_starts_empty():
    """초기 상태에서 aggregate는 빈 dict."""
    p = SubmodelProfiler()
    assert p.aggregate() == {}


def test_profiler_records_per_chunk_per_draw_and_averages():
    """각 chunk의 submodel probs를 그 chunk 내 actuals와 매칭, 전체 평균 hit 반환."""
    probs_a = np.zeros(45); probs_a[0:6] = 0.1  # 번호 1-6 강조
    probs_b = np.zeros(45); probs_b[39:45] = 0.1  # 번호 40-45 강조

    p = SubmodelProfiler()
    # chunk 1: 두 회차 — 한 번은 A 쪽 적중, 한 번은 B 쪽 적중
    p.record_chunk(
        submodel_probs={"A": probs_a, "B": probs_b},
        actuals=[[1, 2, 3, 7, 8, 9], [40, 41, 42, 7, 8, 9]],
    )
    avg = p.aggregate()
    # A는 첫 회차 3개 적중, 둘째 회차 0개 → 평균 1.5
    # B는 첫 회차 0개, 둘째 회차 3개 → 평균 1.5
    assert avg["A"] == pytest.approx(1.5)
    assert avg["B"] == pytest.approx(1.5)


def test_profiler_handles_inconsistent_submodel_keys_across_chunks():
    """chunk마다 submodel 셋이 달라도 각자 평균 계산."""
    probs_a = np.zeros(45); probs_a[0:6] = 0.1
    probs_c = np.zeros(45); probs_c[20:26] = 0.1
    p = SubmodelProfiler()
    p.record_chunk({"A": probs_a}, actuals=[[1, 2, 3, 4, 5, 6]])  # A: 6 hit
    p.record_chunk({"C": probs_c}, actuals=[[21, 22, 23, 24, 25, 26]])  # C: 6 hit
    avg = p.aggregate()
    assert avg["A"] == pytest.approx(6.0)
    assert avg["C"] == pytest.approx(6.0)


def test_profiler_to_json_dict_serializable():
    """aggregate 결과는 JSON 직렬화 가능 (float)."""
    import json
    p = SubmodelProfiler()
    probs = np.zeros(45); probs[0:6] = 0.1
    p.record_chunk({"X": probs}, actuals=[[1, 2, 3, 4, 5, 6]])
    avg = p.aggregate()
    s = json.dumps(avg)
    assert "X" in json.loads(s)
