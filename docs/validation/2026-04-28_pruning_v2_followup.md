# Pruning v2 (Aggressive) — Over-pruning, v1이 Sweet Spot

> 비교 baseline: `2026-04-25_18-30` (no tuning)
> v1 (best): `2026-04-28_23-04/` (3 모델 floor)
> v2 (this): `2026-04-28_23-30/` (5 모델 floor)

## 변경 요약

v1이 Stats/Bayes/Transformer 3개를 floor(0.1)로 내렸다. v2는 더 aggressive하게
**GNN(0.74)과 Markov(0.76)도 추가 floor**. 이론적 근거: random 0.8보다 낮으므로 noise.

| 모델 | hit | v1 weight | **v2 weight** |
|---|---|---|---|
| Pattern Analysis | 0.856 | 2.0 | 2.0 |
| Community | 0.827 | 1.7 | 1.7 |
| DeepSets | 0.798 | 1.4 | 1.4 |
| LSTM | 0.788 | 1.2 | 1.2 |
| PageRank | 0.779 | 1.2 | 1.2 |
| Monte Carlo | 0.769 | 1.1 | 1.1 |
| Markov | 0.760 | 1.0 | **0.1** ← |
| GNN | 0.740 | 0.7 | **0.1** ← |
| Stats | 0.712 | 0.1 | 0.1 |
| Bayes | 0.712 | 0.1 | 0.1 |
| Transformer | 0.712 | 0.1 | 0.1 |

## Before vs After

| 지표 | Baseline | **v1 (3 floored)** | v2 (5 floored) |
|---|---|---|---|
| mean_hits | 0.823 | **0.852** | 0.804 |
| baseline_imp | +2.9% | **+6.5%** | +0.5% |
| top10_hits | 1.36 | 1.385 | **1.442** |
| top6_prob_sum | 0.1338 | 0.1342 | n/a |
| 4-적중 | 0 | 0 | 0 |
| 3-적중 | 11 | 9 | 8 |
| 2-적중 | 85 | **94** | 80 |
| 1-적중 | 225 | 228 | 234 |
| 0-적중 | 199 | **189** | 198 |
| hit_distribution (0/1/2/3/4) | 199/225/85/11/0 | **189/228/94/9/0** | 198/234/80/8/0 |
| zone 41-45 dev | 0.030 | n/a | 0.031 |

## 판정: v1 = Sweet Spot, v2 = Over-pruning

핵심 발견: **GNN(0.74)과 Markov(0.76)는 below-random이지만 ensemble의 diversification 가치 보유**.
둘을 floor로 내리면:
- ✅ top10_hits 1.385 → 1.442 (+0.06): 상위 10번 정확도는 개선
- ❌ mean_hits 0.852 → 0.804 (-5.6%): 회차당 적중 개수는 감소
- ❌ 0-적중 189 → 198 (+9): 빵점 회차 증가
- ❌ 2-적중 94 → 80 (-14): mid-tier 적중 후퇴

**해석**: 가지치기로 인한 분포 sharper화는 top-N rank에서는 도움되지만, 6개 샘플링이라는 **stochastic** 단계에서는 다양한 모델의 약한 신호도 평균화에 기여한다. GNN/Markov가 random보다 약간 못해도 (-5~-8%), 정점이 다른 곳에 위치하면 ensemble noise reduction 역할.

이론적 추정 ≠ ensemble 효과. **Sweet spot은 v1**: 정말로 약한 3개만 floor, 그 외는 약한 weight으로라도 유지.

## 누적 결과 (Ultimate Ensemble) — 9회 시도

| # | 시도 | mean_hits | baseline_imp | 4-적중 | top10 | 판정 |
|---|---|---|---|---|---|---|
| 1 | Baseline | 0.823 | +2.9% | 0 | 1.36 | — |
| 2 | Dynamic Weights | 0.787 | -1.7% | 0 | 1.36 | 회귀 |
| 3 | τ=0.7 | 0.815 | +1.9% | 1 | 1.36 | 회귀 |
| 4 | τ=0.5 | 0.823 | +2.9% | 2 | 1.36 | 부분 (꼬리) |
| 5 | Zone 단독 | 0.787 | -1.7% | 2 | 1.36 | 회귀 |
| 6 | Zone+τ=0.5 | 0.800 | 0.0% | 0 | 1.385 | 회귀 |
| 7 | **Pruning v1** | **0.852** | **+6.5%** | 0 | 1.385 | **★ 베스트** |
| 8 | Pruning+τ=0.5 | 0.779 | -2.6% | 0 | 1.404 | 회귀 |
| 9 | Pruning v2 | 0.804 | +0.5% | 0 | **1.442** | 회귀 |

**9회 중 단 1회만 baseline 위 향상.** v1의 +6.5%가 최종.

## 다음 결정 옵션

| 옵션 | 내용 | 비고 |
|---|---|---|
| **C-G** | v1 ship — main 머지, 사이클 종료 | 안전. 추가 시도는 위험 대비 향상 폭 불확실 |
| C-K | Pruning v3: GNN만 floor (Markov 유지) — 점진적 가지치기 | 한 번 더, 단 v1보다 더 향상은 어려울 듯 |
| C-I | K=20 운영 파라미터 — 다른 차원 향상 | 모델 그대로, 운영성 직접 |
| C-J | 시드 sensitivity (--seed 다양화로 v1 신뢰도 검증) | v1의 +3.5% gain이 우연일 가능성 검증 |

권장: **C-J → C-G**. v1의 향상이 시드 의존적인지 먼저 검증 (5분, 시드 1-2개 추가). 강건하면 ship.

## 코드/테스트 영향

- 신규 가중치: `docs/validation/pruned_weights_v2.json` (실험 흔적)
- 변경 없음 (새 코드 추가 X, 단순 재실행)

## 재현 명령

```bash
venv/bin/python3 -m validator.run_validation \
  --submodel-weights docs/validation/pruned_weights_v2.json
```
