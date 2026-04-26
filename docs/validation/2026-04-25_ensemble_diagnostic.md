# Lotto Validator — 검증 리포트

> 원본 결과: `validation_results/2026-04-25_18-30/` (gitignored, 재현은 `venv/bin/python3 -m validator.run_validation`)

## 실행 환경
- 결과 디렉토리: `validation_results/2026-04-25_18-30/`
- 평가 회차: 104회 (1117~1220회차)
- K (회차당 예측 반복): 5
- 재학습 전략: chunk_10 (10회차마다 재학습)
- 난수 시드: 42
- 백테스트 소요 시간: 616초 (약 10분)

## 요약

Ultimate/Stacking 두 메타 앙상블 모두 **무작위(0.8) 대비 +2.9~+3.8%만 개선**에 그쳐 사실상 신호 추출에 실패. 104회차 × 5예측 = 520건 중 **5+ 적중 0건**, 4 적중도 Stacking 1건뿐. top10_hits ≈ 1.36~1.39 (균등 baseline 1.33)로 **상위권 분별력이 거의 없음**. 핵심 병목은 (1) 동적 가중치 시스템 미작동, (2) 확률분포가 거의 균등(top6_prob_sum ≈ 0.134, uniform 0.133), (3) Stacking은 log-likelihood -63.77로 자신만만하게 틀리는 패턴.

## 체크리스트 결과

| # | 항목 | Ultimate | Stacking | 판정 |
|---|---|---|---|---|
| 1 | baseline 대비 | +2.9% | +3.8% | ⚠️ 우위 미확보 (목표 +10%) |
| 2 | 캘리브레이션 | top6=0.134 ≈ 균등 | top6=0.140 약간 자신감 | ⚠️ 거의 무신호 |
| 3 | 상위권 분별력 (top10) | 1.36 | 1.39 | ❌ 약함 (목표 ≥2.5; baseline 1.33) |
| 4 | 번호대 편향 | 최대 편차 ±0.035 | 21-30 +0.045, 11-20 -0.045 | ⚠️ Stacking이 더 편향 |
| 5 | 다양성 (히트분포) | 0/1/2/3만, 최대 3 | 0/1/2/3/4, 4는 1건 | ❌ 장기 꼬리 부재 (5+ 0건) |
| 6 | Ultimate vs Stacking | mean_hits 차이 0.008 (통계적 무의미) | log-like Ultimate가 ~3배 안정 | △ 거의 동률, 안정성은 Ultimate |
| 7 | 가중치 시스템 | `_get_model_weight`이 정적 상수 + 캐시만 사용 | (해당 없음) | ❌ 동적 가중치 미작동 |

## Top-5 적중률 향상 대책

### #1. 백테스트 기반 즉시 가중치 갱신 (영향도: High / 비용: Low)
- **근거**: `models/ultimate_ensemble_model.py:105-136` `_get_model_weight`이 정적 상수(LSTM=1.5, Stats=1.0 등) + `MetaLearner.load_cached_weights()` 캐시 의존. 백테스트 중 직전 회차 hit feedback이 가중치에 반영되지 않음. 그 결과 Ultimate의 베이스라인 대비 우위가 +2.9%에 그침.
- **실행 방법**: `_get_model_weight` 내부에 직전 N(=20)회차 슬라이딩 윈도우 hit count를 받아 softmax 가중치를 즉시 계산하는 분기를 추가. `BacktestEngine.run_model` 루프에서 chunk마다 누적 hit를 모델에 주입(`current_model.update_recent_hits(hits_window)`). 또는 `utils/history_manager.py:get_advanced_weights()`를 매 chunk 호출하도록 연결.
- **기대 효과**: 강한 서브 모델에 가중치 집중 → top10_hits 1.36 → 1.7~2.0, mean_hits +5~10%p 추가 개선.

### #2. 확률분포 Sharpening (Temperature Scaling) (영향도: High / 비용: Low)
- **근거**: Ultimate의 top6_prob_sum=0.1338이 균등분포 0.1333과 거의 동일. 11개 서브 모델의 확률을 평균낼 때 신호가 상쇄되는 현상. 각 모델은 신호를 갖고 있어도 합산되며 평탄해짐.
- **실행 방법**: `models/ultimate_ensemble_model.py:160-` `_compute_probability_distribution`에서 통합된 `self._probability_dist`에 temperature τ 적용:  `probs = softmax(log(probs)/τ)`. τ=0.5~0.7로 sharpening 후 top-6 mass가 0.20+ 도달하는지 grid search. 단 sharpening 과해서 zone 편향이 커지지 않도록 검증 필요.
- **기대 효과**: top10_hits 분별력 회복 (1.36 → 1.6+), 5+ 적중 사례 발생 가능.

### #3. Stacking Ridge Alpha 강화 (영향도: Mid / 비용: Low)
- **근거**: Stacking의 log_likelihood -63.77 (Ultimate -22.92 대비 ~3배 음수). 일부 회차에서 0에 가까운 확률을 매겼는데 적중하면 큰 음수가 누적 → Ridge 메타가 과적합되고 있음. mean_hits는 +0.008 우위지만 변동성이 큼 (std 0.826 vs Ultimate 0.774).
- **실행 방법**: `models/stacking_ensemble_model.py` `__init__`의 Ridge `alpha` 기본값을 (현재 0.1?) 1.0~5.0으로 상향. 추가로 `np.clip(probs, 1e-3, None)`로 분포 floor 적용해 0 근처 확률 방지.
- **기대 효과**: log-likelihood -63 → -25 부근, std 0.83 → 0.78로 안정화. 자신만만하게 틀리는 패턴 감소.

### #4. 번호대 편향 사후 보정 (영향도: Mid / 비용: Low)
- **근거**: Stacking 21-30 구간 예측 0.277 vs 실제 0.232 (+0.045 편향), 11-20 -0.045. 1년치 학습 데이터의 zone 분포가 학습 시점과 백테스트 시점에서 약간 어긋남.
- **실행 방법**: `validator/report_generator.py` 옵션이 아닌 모델 inference 단계에 zone scaling factor를 도입. `models/ultimate_ensemble_model.py` `_compute_probability_distribution` 마지막에 `probs[zone_idx] *= zone_actual_rate / zone_predicted_mass` 적용 후 정규화. 또는 별도 `utils/zone_calibrator.py` 작성.
- **기대 효과**: zone 편차 ±0.045 → ±0.015 이내, top6_prob_sum 0.140 → 0.147, mean_hits +1~2%p.

### #5. 서브 모델 백테스트 기반 가지치기 (영향도: Mid / 비용: Mid)
- **근거**: 11개 서브 모델 중 일부(Markov 정적 가중치 0.9)가 평균 적중에 거의 기여 못할 가능성. 현재 평균하면서 약한 모델이 신호를 희석시킴.
- **실행 방법**: 본 validator를 확장해 **각 서브 모델 단독**의 mean_hits/top10을 측정하는 `--per-submodel` 모드 추가 (`validator/run_validation.py` + `model_registry.py`에 11개 서브 모델 spec 등록). 결과로 mean_hits < 0.85인 서브 모델은 Ultimate에서 가중치 0.5 이하 또는 제외. Stacking 입력에서도 동일.
- **기대 효과**: 약한 모델 제거로 노이즈 감소. mean_hits +3~5%p, top10 +0.3.

## 부수적 관찰

- **Ultimate의 hit_distribution**이 Stacking보다 좁음 (최대 3 vs 최대 4). 분산은 더 작지만 상한 돌파 능력도 낮음 → diversity_weight=0.1을 0.2~0.3으로 늘려 탐색을 키울 여지.
- **Stacking 1-10 구간 0.265 예측**은 Ridge가 저번호 편향 경향을 학습했음을 시사. 학습 데이터 편향 점검 필요.
- **5+ 적중 0건**은 본질적으로 어렵지만 (확률 ≈ 1/2900), 104회차 × 5예측 = 520건 중 0건은 4+ 적중 가능성도 거의 없다는 의미 — 모델의 예측 다양성 확장 필요.

## 운영 메모

- 본 결과는 chunk_10 전략 사용 (10회차마다 재학습). 더 정밀한 평가는 chunk_1 (매 회차 재학습)이지만 ~10배 시간 소요.
- 데이터: 1220회차 (4월 18일까지). 4월 25일 회차(1221) 추가 후 재실행하면 한 회차 더 평가 가능.
