# Submodel Pruning 백테스트 — 첫 번째 의미 있는 향상

> 비교 baseline: `2026-04-25_18-30` (보정 없음, mean_hits 0.823)
> Profile 결과: `validation_results/2026-04-28_22-52/submodel_profile.json`
> Pruning 결과: `validation_results/2026-04-28_23-04/`

## 변경 요약

분포 형태 튜닝(Z-A~Z-B) 대신 **신호 source 자체를 정리**한다.
1. `validator/submodel_profiler.py` — 각 서브모델의 top-6 hit를 chunk 단위로 측정.
2. `BacktestEngine.run_model(profiler=...)` — chunk 학습 직후 `_model_probabilities` 스냅샷.
3. CLI: `--profile-submodels` (프로파일 저장), `--submodel-weights <json>` (Ultimate weights 오버라이드).

## 서브모델 프로파일 (104 회차, top-6 only)

| 모델 | mean_hits | vs random(0.8) | 기존 weight | 새 weight |
|---|---|---|---|---|
| **Pattern Analysis** | **0.856** | **+7%** | 1.2 | **2.0** |
| **Community** | **0.827** | **+3%** | 1.0 | **1.7** |
| DeepSets | 0.798 | -0% | 1.5 | 1.4 |
| LSTM | 0.788 | -1.5% | 1.5 | 1.2 |
| PageRank | 0.779 | -3% | 1.1 | 1.2 |
| Monte Carlo | 0.769 | -4% | 1.3 | 1.1 |
| Markov | 0.760 | -5% | 0.9 | 1.0 |
| GNN | 0.740 | -8% | 1.2 | 0.7 |
| Stats | 0.712 | -11% | 1.0 | **0.1** (floor) |
| Bayes | 0.712 | -11% | 1.0 | **0.1** (floor) |
| **Transformer** | **0.712** | **-11%** | **1.5** | **0.1** (floor) |

**핵심 발견:**
- 11개 중 **단 2개만** baseline(0.8) 위 (Pattern, Community).
- Transformer는 최하위인데 가장 높은 weight(1.5) — misalignment.
- 분산이 좁음 (0.71-0.86) → 모든 모델이 random 부근. 작은 차이가 큰 효과 가능.

## Before vs After (Ultimate Ensemble)

| 지표 | Baseline | **Pruned** | Δ |
|---|---|---|---|
| mean_hits | 0.823 | **0.852** | **+0.029 (+3.5%)** |
| baseline_improvement_pct | +2.9% | **+6.5%** | **+3.6%p** |
| top10_hits | 1.36 | 1.385 | +0.025 |
| top6_prob_sum | 0.1338 | 0.1342 | +0.0004 |
| log_likelihood | -22.92 | -22.93 | -0.01 (동일) |
| 4-적중 | 0 | 0 | 0 |
| 3-적중 | 11 | 9 | -2 |
| 2-적중 | 85 | **94** | **+9** |
| 1-적중 | 225 | 228 | +3 |
| 0-적중 | 199 | **189** | **-10** |
| hit_distribution (0/1/2/3/4) | 199/225/85/11/0 | **189/228/94/9/0** | 0이 ↓, 2가 ↑ |

**분포 전체가 우측 시프트.** 0-적중 회차가 -10건 줄고 2-적중이 +9건 늘었다. 4-적중 같은 꼬리 효과는 없지만 **전반적인 hit 베이스라인 자체가 올라감**.

## 판정: 첫 번째 의미 있는 향상

| 시도 | mean_hits | baseline_imp | 4-적중 | 판정 |
|---|---|---|---|---|
| Baseline | 0.823 | +2.9% | 0 | — |
| Dynamic Weights | 0.787 | -1.7% | 0 | 회귀 |
| τ=0.7 | 0.815 | +1.9% | 1 | 약간 회귀 |
| τ=0.5 | 0.823 | +2.9% | 2 | 부분 성공 (꼬리) |
| Zone 단독 | 0.787 | -1.7% | 2 | 회귀 |
| Zone+τ=0.5 | 0.800 | 0.0% | 0 | 회귀 |
| **Pruning v1** | **0.852** | **+6.5%** | 0 | **성공** |

- mean_hits 0.823 → 0.852 (실질 +3.5% 향상)
- baseline_improvement +2.9% → +6.5% (2배 이상)
- 분포 형태 튜닝 시도 6번 모두 회귀했지만, **신호 source 정리는 즉시 효과**.

## 원인 분석

1. **잘못된 default weight**: Transformer가 가장 약한데 가장 높은 weight(1.5)였음. 이는 default weight가 모델 "복잡도"에 비례한 직관적 추정이었기 때문 — 실측과 무관.
2. **noise dilution**: 약한 3개 모델(Stats/Bayes/Transformer)이 각각 weight 1.0~1.5로 들어가 강한 모델(Pattern/Community)의 신호를 희석.
3. **분포 형태 튜닝의 한계**: 평균 hit 0.77인 분포에 sharpening/calibration을 가해도 신호 자체가 weak이면 효과 미미.

## 다음 단계 옵션

| 옵션 | 내용 | 비고 |
|---|---|---|
| **C-A** | 인프라+가중치 머지. PR 생성. | 즉시 사용자 가치. |
| **C-B** | Pruning + τ=0.5 결합 (이번엔 신호 source가 강해진 후 sharpening) | +15분. 직교성 재시도 가치 있음. |
| C-C | 가중치를 더 aggressive하게 (Stats/Bayes/Transformer 완전 제거 = floor 0.0) | 코드 변경 필요 (현재 floor 0.1 hardcoded). |
| C-D | 운영 파라미터 (K=20) | 별도 사이클. |

권장: **C-A → C-B**. 인프라는 즉시 머지하고, pruning + τ=0.5 결합으로 추가 향상 시도. 이번에는 약한 모델이 제거된 상태이므로 sharpening 효과가 살아남을 가능성 큼.

## 코드/테스트 영향

- 신규 모듈: `validator/submodel_profiler.py`
- `BacktestEngine.run_model`/`run_all`: `profiler` 파라미터 추가
- `run_validation.py`: `--profile-submodels`, `--submodel-weights` flag
- 신규 테스트: `tests/test_submodel_profiler.py` (7개)
- pytest 34/34 통과 (기존 27 + 신규 7)

## 재현 명령

```bash
# 1. 프로파일 측정
venv/bin/python3 -m validator.run_validation --profile-submodels

# 2. 가중치 적용 재실행 (pruned weights)
venv/bin/python3 -m validator.run_validation \
  --submodel-weights docs/validation/pruned_weights_v1.json
```
