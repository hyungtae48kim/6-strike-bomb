# Zone Calibration Follow-up — 회귀 (zone 편차 미세 개선, mean_hits 악화)

> 비교 baseline: `docs/validation/2026-04-25_ensemble_diagnostic.md` (`2026-04-25_18-30/`, 보정 없음)
> Zone-only 결과: `validation_results/2026-04-26_18-36/`

## 변경 요약

`models/ultimate_ensemble_model.py`에 zone-aware 사후 보정 추가:
- 모듈 레벨 `ZONE_RANGES`, `_compute_zone_targets`, `_apply_zone_calibration` 헬퍼.
- `UltimateEnsembleModel.__init__`에 `enable_zone_calibration: bool = False` 인자.
- `train()`에서 활성 시 학습 데이터의 zone별 실제 당첨 비율을 `_zone_targets`로 저장.
- `_compute_probability_distribution` 마지막에 `combined_probs[zone] *= target / zone_mass` 비례 scaling.
- 같은 zone 내 상대 비율은 보존, zone 질량 0이면 균등 분배.

CLI: `--enable-zone-calibration`. 기본 off → 기존 동작 유지.
Stacking은 미적용.

## Before vs After (Ultimate Ensemble)

| 지표 | Baseline | Zone-only | Δ |
|---|---|---|---|
| mean_hits | **0.823** | 0.787 | -0.036 (악화) |
| baseline_improvement_pct | **+2.9%** | -1.7% | **-4.6%p (회귀)** |
| top10_hits | 1.36 | 1.36 | 0.00 |
| top6_prob_sum | 0.1338 | 0.1338 | 0.0000 |
| log_likelihood | -22.92 | -22.91 | +0.01 |
| **4-적중 회차** | **0** | **2** | +2 |
| 5+ 적중 | 0 | 0 | 0 |
| hit_distribution (0/1/2/3/4) | 199/225/85/11/0 | 211/223/74/10/2 | 0이 12 ↑, 2가 11 ↓ |
| zone 최대 편차 | 0.035 | **0.030** | -0.005 (소폭 개선) |
| zone 평균 절대 편차 | 0.025 | **0.017** | -0.008 |

분석 보고서의 기대 효과(`zone 편차 ±0.015 이내`, `mean_hits +1~2%p`)는 **달성 못함**. zone 편차는 0.035 → 0.030으로 미세 개선만, mean_hits는 정반대 방향(-4.6%p).

특이 관찰: 41-45 zone 편차 +0.030은 그대로 유지. train 데이터의 zone target(0.110)이 evaluation 기간의 실제 비율(0.080)과 다르기 때문 — calibration이 train 분포에 정렬됐을 뿐 evaluation 분포를 추적한 건 아님.

## 판정: 회귀

- mean_hits·baseline_improvement는 dynamic-weights 회귀(-1.7%)와 동일 수준 — **순효과 음수**.
- zone 편차는 약간 줄었지만 모델의 number-level 신호가 zone 평탄화로 깎임.
- 4-적중 0 → 2건은 τ=0.5 sharpening(2건)과 동일. 분포 형태가 약간 바뀐 부수효과로 보임.

원인 가설:
1. **train과 evaluation의 zone 분포 불일치**: 104회차 evaluation 기간에 1-10 zone이 평균보다 약간 자주 나옴. train 전체 평균으로 calibrate하면 evaluation 시점의 분포와 어긋남.
2. **모델의 신호가 zone-level이 아니라 number-level**: 특정 번호에 대한 강한 confidence(상위 6번 강조)가 zone 분배를 거치면서 약화.
3. **lotto는 본질적으로 zone-uniform**: long-run에서 zone 비율이 거의 균등(10/45=0.222). 모델이 우연히 발견한 zone 신호를 calibration이 제거.

## 코드/테스트 영향

- 신규 함수: `models/ultimate_ensemble_model.py::_apply_zone_calibration`, `_compute_zone_targets`, `ZONE_RANGES`
- 신규 인자: `UltimateEnsembleModel(enable_zone_calibration=False)` (하위 호환)
- 신규 CLI 옵션: `--enable-zone-calibration`
- 신규 테스트: `tests/test_ultimate_zone_calibration.py` (8개)
- pytest 35/35 통과 (기존 27 + 신규 8)

## 누적 결과 비교 (Ultimate Ensemble)

| 시도 | mean_hits | baseline_imp | 4-적중 | zone max dev | 판정 |
|---|---|---|---|---|---|
| Baseline | 0.823 | +2.9% | 0 | 0.035 | — |
| Dynamic Weights | 0.787 | -1.7% | 0 | n/a | 회귀 |
| τ=0.7 | 0.815 | +1.9% | 1 | 0.046 | 약간 회귀 |
| τ=0.5 | 0.823 | +2.9% | 2 | 0.060 | 부분 성공 (꼬리 확장) |
| Zone-only | 0.787 | -1.7% | 2 | 0.030 | 회귀 |

지금까지 **mean_hits를 의미 있게 끌어올린 시도는 없음**. τ=0.5만이 회귀 없이 분포 꼬리만 약간 확장.

## 결정 옵션

| 옵션 | 내용 | 비고 |
|---|---|---|
| **Z-A. 인프라만 머지 + 기본 off** | `feature/zone-calibration` PR. CLI `--enable-zone-calibration` 명시 시에만 활성. 기본 동작 영향 없음. | dynamic-weights와 동일 패턴. 분석 인프라 자체는 활용 가능. |
| **Z-B. zone+τ=0.5 결합 시도** | 두 옵션 합친 브랜치 만들어 `--temperature 0.5 --enable-zone-calibration` 실행. 효과가 직교적이면 둘 다 살릴 가능성. | +10분. 효과 불확실 (zone이 sharpening 효과를 상쇄할 수도). |
| **Z-C. Top-5 #5 (서브 모델 가지치기)로 진행** | 약한 서브모델을 식별해 가중치 조정. 분포 형태가 아니라 **소스** 자체를 정리. | 비용 Mid (per-submodel 백테스트 모드 추가 필요). |
| **Z-D. 다른 방향: K=20 등 운영 파라미터 튜닝** | 모델은 그대로, 회차당 K=5를 K=20+로 늘려 적중 1+ 확률을 끌어올림. 운영성과 직접 연결. | "예측력 향상"이 아니라 "구매 전략" 변경. 분석 보고서 #5 외 영역. |

권장: **Z-A → Z-B**. 인프라는 머지하고, zone+τ 결합으로 직교성 확인. 결합도 회귀면 #5 또는 운영 측면으로 전환.

## 재현 명령

```bash
# Baseline
venv/bin/python3 -m validator.run_validation

# Zone calibration 단독
venv/bin/python3 -m validator.run_validation --enable-zone-calibration
```
