# Pruning v1 시드 강건성 검증 — Ship 결정

> 비교: 6 백테스트 (default 3시드 × pruned v1 3시드)
> 결과 디렉토리:
> - default seed 42: `2026-04-25_18-30/`
> - default seed 7: `2026-04-28_baseline_seed7/`
> - default seed 123: `2026-04-28_baseline_seed123/`
> - pruned seed 42: `2026-04-28_23-04/`
> - pruned seed 7: `2026-04-28_seed7/`
> - pruned seed 123: `2026-04-28_seed123/`

## 2×3 매트릭스 (mean_hits)

| | seed 42 | seed 7 | seed 123 | **평균 ± std** |
|---|---|---|---|---|
| Default weights | 0.823 | 0.804 | 0.819 | **0.815 ± 0.008** |
| **Pruned v1** | **0.852** | **0.831** | **0.827** | **0.837 ± 0.011** |
| Δ | +0.029 | +0.027 | +0.008 | **+0.022 (+2.6%)** |

## 판정: 강건한 향상, Ship

**3/3 시드에서 pruning v1이 default 위.** 평균 향상 +2.6%, 최소 +1.0%, 최대 +3.5%.

원본 시드 42 보고의 +6.5%는 random baseline(0.8) 대비. **default-weight baseline 대비는 평균 +2.6%** — 실질 운영 기대 효과.

## 추가 발견

| 메트릭 | Default 평균 | Pruned 평균 |
|---|---|---|
| top10_hits | 1.36 | 1.38 |
| 0-적중 회차 | 205.0 | 196.3 (-8.7) |
| 2-적중 회차 | 81.7 | 80.3 (≈) |
| 3-적중 회차 | 12.7 | 14.0 (+1.3) |
| 4-적중 회차 | 0.7 | 1.0 (≈, noise) |

**일관된 패턴**: 0-적중 회차 감소(-8.7건/104회차), 1-적중 약간 증가, 2-적중 동등, 3-적중 약간 증가. **분포 전체가 우측 시프트**.

4-적중은 시드 의존적 (0~2건). 결정적 이득 아님.

## 누적 결과 — 최종

| # | 시도 | mean_hits (seed 42) | seed 7/123 검증 | 판정 |
|---|---|---|---|---|
| 1 | Baseline | 0.823 | 0.804/0.819 | — (기준) |
| 2 | Dynamic Weights | 0.787 | n/a | 회귀 |
| 3 | τ=0.7 | 0.815 | n/a | 회귀 |
| 4 | τ=0.5 | 0.823 | n/a | 부분 (꼬리만, 시드 의존) |
| 5 | Zone 단독 | 0.787 | n/a | 회귀 |
| 6 | Zone+τ=0.5 | 0.800 | n/a | 회귀 |
| 7 | **Pruning v1** | **0.852** | **0.831/0.827 (3/3 양수)** | **★ Ship** |
| 8 | Pruning+τ=0.5 | 0.779 | n/a | 회귀 |
| 9 | Pruning v2 | 0.804 | n/a | 회귀 |

## 결정: Pruning v1을 표준 운영 가중치로 채택

권장 PR: `feature/submodel-pruning` → `main`. 머지 후 `pruned_weights_v1.json`을 default 가중치로 활용.

`MetaLearner.load_cached_weights` 캐시에 v1 가중치 저장하면 코드 변경 없이 표준이 됨. (`utils/meta_learner.py` 캐시 위치 확인 후 별도 단계.)

## 코드/테스트 영향

- 변경 없음 (이미 push 완료, 추가 시드는 측정만)
- 새 메트릭 데이터: `validation_results/2026-04-28_baseline_seed{7,123}/`, `2026-04-28_seed{7,123}/`

## 재현 명령

```bash
for s in 42 7 123; do
  venv/bin/python3 -m validator.run_validation --seed $s
  venv/bin/python3 -m validator.run_validation --seed $s \
    --submodel-weights docs/validation/pruned_weights_v1.json
done
```
