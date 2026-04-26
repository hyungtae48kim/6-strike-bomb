# Temperature Scaling Follow-up — 부분 성공 (꼬리 확장)

> 비교 baseline: `docs/validation/2026-04-25_ensemble_diagnostic.md` (`2026-04-25_18-30/`, sharpening 없음)
> τ=0.7 결과: `validation_results/2026-04-26_17-02/`
> τ=0.5 결과: `validation_results/2026-04-26_17-15/`

## 변경 요약

`UltimateEnsembleModel.__init__`에 `sharpening_temperature` 옵션 추가. `_compute_probability_distribution` 마지막 정규화 직후 `softmax(log(p)/τ)` 적용 (모듈 레벨 `_apply_temperature` 헬퍼). CLI는 `--temperature τ` 플래그로 활성화. 기본 None → 기존 동작 그대로.

Stacking은 미적용 (이미 0.140 mass로 sharper, Ridge가 자체 학습).

## Grid Search 결과 (Ultimate Ensemble)

| 지표 | Baseline (τ=None) | τ=0.7 | τ=0.5 |
|---|---|---|---|
| mean_hits | **0.823** | 0.815 | **0.823** |
| baseline_improvement_pct | **+2.9%** | +1.9% | **+2.9%** |
| top10_hits | 1.36 | 1.30 | 1.35 |
| top6_prob_sum | 0.1338 | 0.1337 | 0.1334 |
| log_likelihood | -22.92 | -23.05 | -23.29 |
| **4-적중 회차 수** | **0** | **1** | **2** |
| 5+ 적중 | 0 | 0 | 0 |
| hit_distribution (0/1/2/3/4) | 199/225/85/11/0 | 206/220/79/14/1 | 202/221/86/9/2 |
| zone 최대 편차 | 0.035 | 0.046 | 0.060 |

## 판정: 부분 성공

- **mean_hits 보존**: τ=0.5에서 baseline과 정확히 동일 (0.823) → **회귀 없음**.
- **꼬리 확장**: 4-적중이 baseline의 0건에서 τ=0.5의 2건으로 증가 → 상위 회차 분별력이 개선됨. 분석 보고서에서 강조한 "장기 꼬리 부재" 문제가 부분적으로 해소.
- **top10_hits 미세 감소**: 1.36 → 1.35는 noise 수준. log_likelihood -22.92 → -23.29는 sharpening이 일부 회차에서 자신 있게 틀린 것.
- **zone 편차 확대**: ±0.035 → ±0.060. 큰 모델 의견에 비중이 커지면서 학습 데이터의 zone 편향도 함께 증폭.
- **τ=0.7은 효과 미미**: 분포 변화가 너무 작아 회귀(-1%p)만 발생.

τ=0.5가 가장 균형 잡힌 결과. 평균은 그대로 두면서 상위 회차에서 진짜 hit를 늘림.

## 코드/테스트 영향

- 신규 모듈 함수: `models/ultimate_ensemble_model.py::_apply_temperature`
- 신규 인자: `UltimateEnsembleModel(sharpening_temperature=None)` (하위 호환)
- 신규 CLI 옵션: `--temperature τ` (Ultimate에만 주입)
- 신규 테스트: `tests/test_ultimate_sharpening.py` (8개)
- pytest 35/35 통과 (기존 27 + 신규 8)

## 결정 옵션

| 옵션 | 내용 | 비고 |
|---|---|---|
| **T-A. 인프라 머지 + 기본 off** | `feature/temperature-scaling`을 main에 PR. CLI는 `--temperature 0.5` 명시 시에만 활성. 기본 동작 영향 없음. | 가장 안전. 다음 분석은 4-적중 패턴 회차 분석. |
| **T-B. τ=0.5를 default로** | `ENSEMBLE_SPECS`의 Ultimate kwargs에 `sharpening_temperature: 0.5` 추가. baseline 변경. | mean_hits 동일·꼬리 개선이지만 zone bias 확대 → 다음 사이클(#4 zone 보정)과 결합 권장. |
| **T-C. τ=0.3 추가 시도** | 더 강한 sharpening으로 5+ 적중 가능성 탐색. zone bias 더 커질 위험. | 비용 ~10분. 효과 불확실. |
| **T-D. #4 zone 보정으로 진행** | 현재 sharpening은 동결, 다음 개선안 #4(zone scaling factor)을 단독 시도. | sharpening의 부작용(zone 편차)을 직접 해결. |

권장: **T-A → T-D**. 인프라는 안전하게 머지하고 (옵션 활성화 시 효과 확인 가능), 다음 사이클은 zone 보정으로 sharpening의 부작용을 잡으면서 기본 분포 자체를 개선.

## 재현 명령

```bash
# Baseline
venv/bin/python3 -m validator.run_validation

# τ=0.7
venv/bin/python3 -m validator.run_validation --temperature 0.7

# τ=0.5 (권장)
venv/bin/python3 -m validator.run_validation --temperature 0.5
```
