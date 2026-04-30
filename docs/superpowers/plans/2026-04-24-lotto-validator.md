# Lotto Validator Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ultimate Ensemble와 Stacking Ensemble 두 메타 앙상블을 최근 104회차에서 Walk-Forward 백테스트하여 B+C 메트릭으로 평가하고, 서브에이전트가 해석해 적중률 향상 대책을 제시하는 검증 시스템을 구축한다.

**Architecture:** `validator/` 파이썬 패키지가 백테스트 루프·메트릭·리포트를 담당하고, `.claude/agents/lotto-validator.md` 서브에이전트가 결과를 읽어 체크리스트 기반 진단 + 대책을 이 창에 돌려준다. 모든 결정적 실험은 `validation_results/<timestamp>/`에 저장되어 재현 가능하다.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, matplotlib (차트), pytest (테스트), PyTorch (앙상블 내부). 모든 직렬화는 JSON (pickle 미사용 — 보안 고려).

**Spec:** `docs/superpowers/specs/2026-04-24-lotto-validator-design.md`

---

## File Structure

**Create:**
- `validator/__init__.py` — 패키지 마커
- `validator/config.py` — `ValidatorConfig`, `set_seeds()`, `ENSEMBLE_SPECS`
- `validator/metrics.py` — `BMetrics`, `CMetrics`, 계산 함수
- `validator/model_registry.py` — `ModelSpec`, `get_ensemble_specs()`
- `validator/backtest_engine.py` — `DrawResult`, `ModelBacktestResult`, `BacktestEngine`, `save_checkpoint`/`load_checkpoint` (JSON 기반)
- `validator/report_generator.py` — CSV/JSON/MD/PNG 출력
- `validator/run_validation.py` — CLI 엔트리포인트
- `tests/__init__.py`, `tests/conftest.py`
- `tests/test_config.py`, `tests/test_metrics.py`, `tests/test_model_registry.py`, `tests/test_backtest_engine.py`, `tests/test_report_generator.py`, `tests/test_smoke.py`
- `.claude/agents/lotto-validator.md` — 서브에이전트 정의

**Modify:**
- `requirements.txt` — `pytest`, `matplotlib` 추가
- `.gitignore` — `validation_results/` 추가

---

## Task 0: 준비 — pytest 및 matplotlib 설치

**Files:**
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: `requirements.txt`에 테스트/차트 의존성 추가**

기존 내용 마지막에 추가:
```
matplotlib
pytest
```

- [ ] **Step 2: `.gitignore`에 검증 결과 디렉토리 추가**

기존 내용 마지막에 한 줄 추가:
```
validation_results/
```

- [ ] **Step 3: 의존성 설치 및 버전 확인**

Run: `pip install -r requirements.txt && python -c "import pytest, matplotlib; print(pytest.__version__, matplotlib.__version__)"`
Expected: 두 버전 숫자가 출력됨. 실패 시 pip 권한/네트워크 확인.

- [ ] **Step 4: Commit**

```bash
git add requirements.txt .gitignore
git commit -m "chore: add pytest and matplotlib for validator agent"
```

---

## Task 1: validator 패키지 스켈레톤 + config 모듈

**Files:**
- Create: `validator/__init__.py`
- Create: `validator/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: 테스트 파일부터 작성 (TDD)**

`tests/__init__.py`:
```python
```

`tests/conftest.py`:
```python
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_lotto_df() -> pd.DataFrame:
    """50회차 가상의 로또 이력 DataFrame."""
    rng = np.random.default_rng(0)
    rows = []
    for draw_no in range(1, 51):
        nums = sorted(rng.choice(range(1, 46), size=6, replace=False).tolist())
        bonus = int(rng.integers(1, 46))
        while bonus in nums:
            bonus = int(rng.integers(1, 46))
        rows.append({
            "drwNo": draw_no,
            "drwNoDate": f"2020-01-{draw_no:02d}",
            "drwtNo1": nums[0], "drwtNo2": nums[1], "drwtNo3": nums[2],
            "drwtNo4": nums[3], "drwtNo5": nums[4], "drwtNo6": nums[5],
            "bnusNo": bonus,
        })
    return pd.DataFrame(rows)
```

`tests/test_config.py`:
```python
import random
import numpy as np

from validator.config import ValidatorConfig, set_seeds, ENSEMBLE_SPECS


def test_config_has_sensible_defaults():
    cfg = ValidatorConfig()
    assert cfg.eval_window_draws == 104
    assert cfg.random_seed == 42
    assert cfg.predictions_per_draw == 5
    assert cfg.retrain_chunk_size == 10
    assert cfg.report_ttl_hours == 24
    assert cfg.results_base_dir == "validation_results"


def test_set_seeds_is_deterministic():
    set_seeds(123)
    a = (random.random(), np.random.rand())
    set_seeds(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_ensemble_specs_lists_only_ultimate_and_stacking():
    names = sorted(spec["name"] for spec in ENSEMBLE_SPECS)
    assert names == ["Stacking Ensemble", "Ultimate Ensemble"]
    for spec in ENSEMBLE_SPECS:
        assert spec["strategy"] == "chunk_10"
```

- [ ] **Step 2: 테스트 실행으로 실패 확인**

Run: `pytest tests/test_config.py -v`
Expected: `ModuleNotFoundError: No module named 'validator'`.

- [ ] **Step 3: 패키지 및 config 구현**

`validator/__init__.py`:
```python
```

`validator/config.py`:
```python
"""Validator 설정 및 시드 고정 유틸리티."""
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValidatorConfig:
    """Validator 실행 설정."""
    eval_window_draws: int = 104
    random_seed: int = 42
    predictions_per_draw: int = 5
    retrain_chunk_size: int = 10
    report_ttl_hours: int = 24
    results_base_dir: str = "validation_results"


ENSEMBLE_SPECS = [
    {
        "name": "Ultimate Ensemble",
        "module": "models.ultimate_ensemble_model",
        "class": "UltimateEnsembleModel",
        "kwargs": {},
        "strategy": "chunk_10",
    },
    {
        "name": "Stacking Ensemble",
        "module": "models.stacking_ensemble_model",
        "class": "StackingEnsembleModel",
        "kwargs": {"meta_model_type": "ridge"},
        "strategy": "chunk_10",
    },
]


def set_seeds(seed: int) -> None:
    """모든 주요 난수 라이브러리의 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
```

- [ ] **Step 4: 테스트 재실행으로 통과 확인**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add validator/__init__.py validator/config.py tests/__init__.py tests/conftest.py tests/test_config.py
git commit -m "feat(validator): add config module with seed fixing and ensemble specs"
```

---

## Task 2: B 메트릭 (예측 기반)

**Files:**
- Create: `validator/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_metrics.py`:
```python
import numpy as np
import pytest

from validator.metrics import BMetrics, compute_b_metrics


def test_b_metrics_counts_hits_correctly():
    predictions = [
        [[1, 2, 3, 4, 5, 6]],         # 6 hits
        [[10, 20, 30, 40, 41, 42]],   # 0 hits
        [[1, 2, 3, 40, 41, 42]],      # 3 hits
    ]
    actuals = [
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6],
    ]
    m = compute_b_metrics(predictions, actuals)
    assert m.mean_hits == (6 + 0 + 3) / 3
    assert m.hit_distribution[6] == 1
    assert m.hit_distribution[0] == 1
    assert m.hit_distribution[3] == 1
    assert m.high_tier_rate == 1 / 3


def test_b_metrics_averages_across_k_predictions():
    predictions = [
        [[1, 2, 3, 4, 5, 6], [10, 20, 30, 40, 41, 42]],
    ]
    actuals = [[1, 2, 3, 4, 5, 6]]
    m = compute_b_metrics(predictions, actuals)
    assert m.mean_hits == 3.0


def test_b_metrics_baseline_comparison_is_percent():
    predictions = [[[1, 2, 3, 40, 41, 42]]]  # 3 hits
    actuals = [[1, 2, 3, 4, 5, 6]]
    m = compute_b_metrics(predictions, actuals)
    # baseline = 0.8, 3 / 0.8 - 1 = 2.75 -> 275%
    assert abs(m.baseline_improvement_pct - 275.0) < 0.1


def test_b_metrics_empty_raises():
    with pytest.raises(ValueError):
        compute_b_metrics([], [])
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_metrics.py -v`
Expected: ImportError.

- [ ] **Step 3: metrics.py B 메트릭 구현**

`validator/metrics.py`:
```python
"""B (예측 기반) 및 C (확률분포 기반) 메트릭 계산."""
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


BASELINE_EXPECTED_HITS = 6 * 6 / 45  # 0.8


@dataclass
class BMetrics:
    """예측 기반 적중 통계."""
    mean_hits: float
    std_hits: float
    hit_distribution: Dict[int, int]
    high_tier_rate: float
    baseline_improvement_pct: float


def compute_b_metrics(
    predictions: List[List[List[int]]],
    actuals: List[List[int]],
) -> BMetrics:
    """
    predictions: [회차][K][6] 구조의 예측. K는 회차별 예측 횟수.
    actuals: [회차][6] 구조의 실제 당첨 번호.
    """
    if not predictions or not actuals:
        raise ValueError("predictions and actuals must be non-empty")
    if len(predictions) != len(actuals):
        raise ValueError("predictions and actuals length must match")

    all_hits: List[int] = []
    for preds_k, actual in zip(predictions, actuals):
        actual_set = set(actual)
        for pred in preds_k:
            hits = len(set(pred) & actual_set)
            all_hits.append(hits)

    hits_arr = np.array(all_hits, dtype=float)
    mean_hits = float(hits_arr.mean())
    std_hits = float(hits_arr.std())
    distribution = dict(Counter(int(h) for h in hits_arr))
    high_tier_rate = float((hits_arr >= 5).sum() / len(hits_arr))
    improvement = (mean_hits / BASELINE_EXPECTED_HITS - 1.0) * 100.0

    return BMetrics(
        mean_hits=mean_hits,
        std_hits=std_hits,
        hit_distribution=distribution,
        high_tier_rate=high_tier_rate,
        baseline_improvement_pct=improvement,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_metrics.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add validator/metrics.py tests/test_metrics.py
git commit -m "feat(validator): add B metrics (prediction-based hit statistics)"
```

---

## Task 3: C 메트릭 (확률분포 기반)

**Files:**
- Modify: `validator/metrics.py`
- Modify: `tests/test_metrics.py`

- [ ] **Step 1: C 메트릭 테스트 추가**

`tests/test_metrics.py` 끝에 추가:
```python
from validator.metrics import CMetrics, compute_c_metrics


def _peaked_probs(top_numbers: list, base: float = 0.01) -> np.ndarray:
    """상위 번호에 높은 확률을 준 45-dim 분포 (sums to 1)."""
    probs = np.full(45, base)
    for rank, num in enumerate(top_numbers):
        probs[num - 1] = base + 0.2 - rank * 0.01
    probs = probs / probs.sum()
    return probs


def test_c_metrics_top6_prob_sum_high_when_winners_are_top():
    probs = _peaked_probs([1, 2, 3, 4, 5, 6])
    actual = [1, 2, 3, 4, 5, 6]
    m = compute_c_metrics([probs], [actual])
    assert m.top6_prob_sum > 0.5


def test_c_metrics_mean_rank_near_baseline_for_uniform():
    probs = np.ones(45) / 45
    actual = [1, 2, 3, 4, 5, 6]
    m = compute_c_metrics([probs], [actual])
    assert 20 <= m.mean_rank <= 26


def test_c_metrics_top10_hits_counts_correctly():
    probs = _peaked_probs([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    actual = [1, 2, 3, 40, 41, 42]
    m = compute_c_metrics([probs], [actual])
    assert m.top10_hits == 3.0


def test_c_metrics_zone_bias_detects_concentration():
    probs = np.full(45, 0.001)
    probs[0:10] = 0.099
    probs = probs / probs.sum()
    actual = [30, 31, 32, 33, 34, 35]
    m = compute_c_metrics([probs], [actual])
    assert m.zone_predicted_mass["1-10"] > 0.5


def test_c_metrics_log_likelihood_higher_for_better_predictor():
    actual = [1, 2, 3, 4, 5, 6]
    good = _peaked_probs([1, 2, 3, 4, 5, 6])
    uniform = np.ones(45) / 45
    m_good = compute_c_metrics([good], [actual])
    m_uniform = compute_c_metrics([uniform], [actual])
    assert m_good.log_likelihood > m_uniform.log_likelihood
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_metrics.py -v -k c_metrics`
Expected: 5 tests failed/errored with ImportError or AttributeError.

- [ ] **Step 3: CMetrics 및 compute_c_metrics 구현**

`validator/metrics.py` 하단에 추가 (모듈 상단 import에 `from scipy.stats import rankdata` 추가):
```python
@dataclass
class CMetrics:
    """확률분포 기반 메트릭."""
    top6_prob_sum: float
    top10_hits: float
    mean_rank: float
    log_likelihood: float
    zone_predicted_mass: Dict[str, float]
    zone_actual_rate: Dict[str, float]


ZONES = [("1-10", 1, 10), ("11-20", 11, 20), ("21-30", 21, 30),
         ("31-40", 31, 40), ("41-45", 41, 45)]


def compute_c_metrics(
    probabilities: List[np.ndarray],
    actuals: List[List[int]],
) -> CMetrics:
    """
    probabilities: [회차][45] 확률 분포.
    actuals: [회차][6] 당첨 번호.
    """
    if not probabilities or not actuals:
        raise ValueError("probabilities and actuals must be non-empty")
    if len(probabilities) != len(actuals):
        raise ValueError("probabilities and actuals length must match")

    top6_sums: List[float] = []
    top10_hit_counts: List[float] = []
    ranks_collected: List[float] = []
    log_likes: List[float] = []
    zone_pred_sums = {z: 0.0 for z, _, _ in ZONES}
    zone_actual_counts = {z: 0 for z, _, _ in ZONES}

    for probs, actual in zip(probabilities, actuals):
        probs = np.asarray(probs, dtype=float)
        if probs.shape != (45,):
            raise ValueError(f"probability must be 45-dim, got {probs.shape}")

        actual_idx = [n - 1 for n in actual]
        top6_sums.append(float(probs[actual_idx].sum()))

        # 동순위는 평균 랭크로 처리 (균등분포에서 mean_rank ≈ 23이 되도록)
        all_ranks = rankdata(-probs, method="average")
        ranks = [float(all_ranks[i]) for i in actual_idx]
        ranks_collected.extend(ranks)
        top10_hit_counts.append(float(sum(1 for r in ranks if r <= 10)))

        eps = 1e-12
        log_likes.append(float(np.log(np.clip(probs[actual_idx], eps, None)).sum()))

        for zone_name, lo, hi in ZONES:
            zone_pred_sums[zone_name] += float(probs[lo - 1:hi].sum())
            zone_actual_counts[zone_name] += int(sum(1 for n in actual if lo <= n <= hi))

    n_draws = len(probabilities)
    zone_pred_mass = {k: v / n_draws for k, v in zone_pred_sums.items()}
    zone_actual_rate = {k: v / (n_draws * 6) for k, v in zone_actual_counts.items()}

    return CMetrics(
        top6_prob_sum=float(np.mean(top6_sums)),
        top10_hits=float(np.mean(top10_hit_counts)),
        mean_rank=float(np.mean(ranks_collected)),
        log_likelihood=float(np.mean(log_likes)),
        zone_predicted_mass=zone_pred_mass,
        zone_actual_rate=zone_actual_rate,
    )
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_metrics.py -v`
Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add validator/metrics.py tests/test_metrics.py
git commit -m "feat(validator): add C metrics (probability distribution evaluation)"
```

---

## Task 4: Model Registry

**Files:**
- Create: `validator/model_registry.py`
- Create: `tests/test_model_registry.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_model_registry.py`:
```python
import numpy as np

from validator.model_registry import ModelSpec, get_ensemble_specs


def test_registry_returns_two_ensembles():
    specs = get_ensemble_specs()
    assert len(specs) == 2
    names = sorted(s.name for s in specs)
    assert names == ["Stacking Ensemble", "Ultimate Ensemble"]


def test_every_spec_can_instantiate_model():
    specs = get_ensemble_specs()
    for spec in specs:
        instance = spec.instantiate()
        assert instance is not None
        assert hasattr(instance, "get_probability_distribution")
        probs = instance.get_probability_distribution()
        assert isinstance(probs, np.ndarray)
        assert probs.shape == (45,)


def test_spec_has_chunk_10_strategy():
    for spec in get_ensemble_specs():
        assert spec.strategy == "chunk_10"


def test_spec_name_is_readable():
    for spec in get_ensemble_specs():
        assert " " in spec.name
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_model_registry.py -v`
Expected: ImportError.

- [ ] **Step 3: 구현**

`validator/model_registry.py`:
```python
"""평가 대상 앙상블 모델의 메타데이터 레지스트리."""
import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List

from validator.config import ENSEMBLE_SPECS


@dataclass
class ModelSpec:
    """백테스트 대상 모델의 스펙."""
    name: str
    module: str
    class_name: str
    kwargs: Dict[str, Any] = field(default_factory=dict)
    strategy: str = "chunk_10"

    def instantiate(self):
        """모델 인스턴스를 동적으로 생성한다."""
        mod = importlib.import_module(self.module)
        cls = getattr(mod, self.class_name)
        return cls(**self.kwargs)


def get_ensemble_specs() -> List[ModelSpec]:
    """평가 대상 앙상블 스펙 리스트를 반환한다."""
    return [
        ModelSpec(
            name=spec["name"],
            module=spec["module"],
            class_name=spec["class"],
            kwargs=dict(spec.get("kwargs", {})),
            strategy=spec["strategy"],
        )
        for spec in ENSEMBLE_SPECS
    ]
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_model_registry.py -v`
Expected: 4 passed. (Ultimate Ensemble 인스턴스 생성 시 서브모델 초기화로 약간의 시간 소요 가능.)

- [ ] **Step 5: Commit**

```bash
git add validator/model_registry.py tests/test_model_registry.py
git commit -m "feat(validator): add model registry for ensemble specs"
```

---

## Task 5: Backtest Engine — Walk-Forward 루프

**Files:**
- Create: `validator/backtest_engine.py`
- Create: `tests/test_backtest_engine.py`

- [ ] **Step 1: 테스트 작성 (스텁 모델로 엔진 동작 검증)**

`tests/test_backtest_engine.py`:
```python
import numpy as np
import pandas as pd
import pytest

from validator.backtest_engine import (
    BacktestEngine,
    DrawResult,
    ModelBacktestResult,
)
from validator.config import ValidatorConfig
from validator.model_registry import ModelSpec


class _StubModel:
    """결정적 스텁 모델: 항상 1-6번 예측."""
    def __init__(self):
        self._probs = None

    def train(self, df: pd.DataFrame):
        probs = np.full(45, 0.01)
        probs[0:6] = 0.15
        self._probs = probs / probs.sum()

    def predict(self):
        return [1, 2, 3, 4, 5, 6]

    def predict_multiple(self, n_sets=5):
        return [[1, 2, 3, 4, 5, 6] for _ in range(n_sets)]

    def get_probability_distribution(self):
        return self._probs.copy() if self._probs is not None else np.ones(45) / 45


class _StubSpec(ModelSpec):
    def instantiate(self):
        return _StubModel()


@pytest.fixture
def stub_spec():
    return _StubSpec(
        name="Stub",
        module="__main__",
        class_name="Stub",
        kwargs={},
        strategy="chunk_10",
    )


def test_engine_runs_on_window_and_produces_draw_results(sample_lotto_df, stub_spec):
    cfg = ValidatorConfig(
        eval_window_draws=20,
        predictions_per_draw=2,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    result = engine.run_model(stub_spec)

    assert isinstance(result, ModelBacktestResult)
    assert result.model_name == "Stub"
    assert len(result.draw_results) == 20
    for dr in result.draw_results:
        assert isinstance(dr, DrawResult)
        assert len(dr.predictions) == 2  # K=2
        assert len(dr.actual) == 6
        assert dr.probability.shape == (45,)


def test_engine_respects_chunk_size_by_not_retraining_every_draw(sample_lotto_df):
    instance_count = {"n": 0}

    class CountingSpec(ModelSpec):
        def instantiate(self):
            instance_count["n"] += 1
            return _StubModel()

    cfg = ValidatorConfig(
        eval_window_draws=20,
        predictions_per_draw=1,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    spec = CountingSpec(
        name="Counting", module="x", class_name="x", kwargs={}, strategy="chunk_10",
    )
    engine.run_model(spec)
    assert instance_count["n"] == 4  # 20회차 / 5 청크


def test_engine_auto_shrinks_window_when_insufficient_data(sample_lotto_df, stub_spec):
    # 50회차 데이터에 100회차 요구 → 자동 축소
    cfg = ValidatorConfig(
        eval_window_draws=100,
        predictions_per_draw=1,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    result = engine.run_model(stub_spec)
    assert 0 < len(result.draw_results) < 50


def test_run_all_returns_dict_keyed_by_model_name(sample_lotto_df, stub_spec):
    cfg = ValidatorConfig(
        eval_window_draws=10,
        predictions_per_draw=1,
        retrain_chunk_size=5,
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    results = engine.run_all([stub_spec])
    assert set(results.keys()) == {"Stub"}
    assert isinstance(results["Stub"], ModelBacktestResult)
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_backtest_engine.py -v`
Expected: ImportError.

- [ ] **Step 3: 구현**

`validator/backtest_engine.py`:
```python
"""Walk-Forward 백테스트 엔진."""
import contextlib
import io
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd

from validator.config import ValidatorConfig, set_seeds
from validator.model_registry import ModelSpec


@dataclass
class DrawResult:
    """단일 회차의 백테스트 결과."""
    draw_no: int
    predictions: List[List[int]]   # 회차당 K개 예측
    probability: np.ndarray        # 45-dim 확률 분포
    actual: List[int]


@dataclass
class ModelBacktestResult:
    """단일 모델의 전체 백테스트 결과."""
    model_name: str
    draw_results: List[DrawResult] = field(default_factory=list)


class BacktestEngine:
    """Walk-Forward 백테스트 엔진."""

    MIN_TRAIN_DRAWS = 100  # 앙상블이 안정적으로 학습되려면 최소 필요

    def __init__(self, df: pd.DataFrame, config: ValidatorConfig):
        self.df = df.sort_values("drwNo").reset_index(drop=True)
        self.config = config

    def _eval_range(self) -> range:
        """평가할 회차의 인덱스 범위. 학습 데이터가 부족하면 자동으로 축소된다."""
        total = len(self.df)
        min_train = min(self.MIN_TRAIN_DRAWS, max(1, total // 2))
        window = min(self.config.eval_window_draws, total - min_train)
        window = max(0, window)
        start = total - window
        return range(start, total)

    def _extract_actual(self, row: pd.Series) -> List[int]:
        return sorted(int(row[f"drwtNo{i}"]) for i in range(1, 7))

    def _predict_k(self, model, k: int) -> List[List[int]]:
        """K개 예측을 얻는다. predict_multiple 사용 가능하면 활용."""
        if hasattr(model, "predict_multiple"):
            try:
                return [sorted(list(p)) for p in model.predict_multiple(n_sets=k)]
            except Exception:
                pass
        return [sorted(list(model.predict())) for _ in range(k)]

    def run_model(self, spec: ModelSpec) -> ModelBacktestResult:
        """단일 모델에 대해 Walk-Forward 백테스트를 수행한다."""
        set_seeds(self.config.random_seed)
        eval_range = self._eval_range()
        result = ModelBacktestResult(model_name=spec.name)

        chunk = self.config.retrain_chunk_size
        eval_indices = list(eval_range)
        current_model = None

        for i, idx in enumerate(eval_indices):
            if i % chunk == 0:
                train_df = self.df.iloc[:idx]
                current_model = spec.instantiate()
                with contextlib.redirect_stdout(io.StringIO()):
                    current_model.train(train_df)

            row = self.df.iloc[idx]
            predictions = self._predict_k(current_model, self.config.predictions_per_draw)
            try:
                probability = np.asarray(
                    current_model.get_probability_distribution(), dtype=float
                )
            except Exception:
                probability = np.ones(45) / 45
            if probability.shape != (45,):
                probability = np.ones(45) / 45

            result.draw_results.append(DrawResult(
                draw_no=int(row["drwNo"]),
                predictions=predictions,
                probability=probability,
                actual=self._extract_actual(row),
            ))

        return result

    def run_all(self, specs: List[ModelSpec]) -> Dict[str, ModelBacktestResult]:
        """여러 모델을 순차적으로 백테스트한다."""
        return {spec.name: self.run_model(spec) for spec in specs}
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_backtest_engine.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add validator/backtest_engine.py tests/test_backtest_engine.py
git commit -m "feat(validator): add Walk-Forward backtest engine with chunk_10 strategy"
```

---

## Task 6: Backtest Engine — JSON 체크포인트 저장/복원

**Files:**
- Modify: `validator/backtest_engine.py`
- Modify: `tests/test_backtest_engine.py`

- [ ] **Step 1: 체크포인트 테스트 추가**

`tests/test_backtest_engine.py`에 추가:
```python
from pathlib import Path

from validator.backtest_engine import save_checkpoint, load_checkpoint


def test_save_and_load_checkpoint_roundtrip(tmp_path, sample_lotto_df, stub_spec):
    cfg = ValidatorConfig(
        eval_window_draws=10, predictions_per_draw=1, retrain_chunk_size=5
    )
    engine = BacktestEngine(sample_lotto_df, cfg)
    results = engine.run_all([stub_spec])

    ckpt = tmp_path / "checkpoint.json"
    save_checkpoint(results, ckpt)
    assert ckpt.exists()

    loaded = load_checkpoint(ckpt)
    assert set(loaded.keys()) == set(results.keys())
    assert loaded["Stub"].model_name == "Stub"
    assert len(loaded["Stub"].draw_results) == len(results["Stub"].draw_results)
    dr_a = results["Stub"].draw_results[0]
    dr_b = loaded["Stub"].draw_results[0]
    assert dr_a.draw_no == dr_b.draw_no
    assert dr_a.actual == dr_b.actual
    assert np.allclose(dr_a.probability, dr_b.probability)


def test_load_checkpoint_missing_returns_none(tmp_path):
    assert load_checkpoint(tmp_path / "does-not-exist.json") is None
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_backtest_engine.py -v -k checkpoint`
Expected: ImportError (`save_checkpoint`, `load_checkpoint` undefined).

- [ ] **Step 3: JSON 체크포인트 함수 추가 (pickle 미사용)**

`validator/backtest_engine.py` 끝에 추가:
```python
import json
from pathlib import Path
from typing import Optional


def save_checkpoint(
    results: Dict[str, ModelBacktestResult],
    path,
) -> None:
    """백테스트 결과를 JSON으로 저장한다. (pickle 미사용)"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {}
    for model_name, result in results.items():
        serializable[model_name] = {
            "model_name": result.model_name,
            "draw_results": [
                {
                    "draw_no": dr.draw_no,
                    "predictions": [list(p) for p in dr.predictions],
                    "probability": dr.probability.tolist(),
                    "actual": list(dr.actual),
                }
                for dr in result.draw_results
            ],
        }
    with path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False)


def load_checkpoint(path) -> Optional[Dict[str, ModelBacktestResult]]:
    """체크포인트가 있으면 JSON에서 로드, 없으면 None."""
    path = Path(path)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    results: Dict[str, ModelBacktestResult] = {}
    for model_name, payload in data.items():
        draw_results = [
            DrawResult(
                draw_no=int(dr["draw_no"]),
                predictions=[list(p) for p in dr["predictions"]],
                probability=np.array(dr["probability"], dtype=float),
                actual=list(dr["actual"]),
            )
            for dr in payload["draw_results"]
        ]
        results[model_name] = ModelBacktestResult(
            model_name=payload["model_name"],
            draw_results=draw_results,
        )
    return results
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_backtest_engine.py -v`
Expected: 6 passed (기존 4 + 새 2).

- [ ] **Step 5: Commit**

```bash
git add validator/backtest_engine.py tests/test_backtest_engine.py
git commit -m "feat(validator): add JSON checkpoint save/load for backtest results"
```

---

## Task 7: Report Generator — CSV + JSON

**Files:**
- Create: `validator/report_generator.py`
- Create: `tests/test_report_generator.py`

- [ ] **Step 1: 테스트 작성**

`tests/test_report_generator.py`:
```python
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validator.backtest_engine import DrawResult, ModelBacktestResult
from validator.config import ValidatorConfig
from validator.metrics import compute_b_metrics, compute_c_metrics
from validator.report_generator import (
    write_raw_predictions_csv,
    write_metrics_summary_json,
)


def _fake_result(name: str) -> ModelBacktestResult:
    probs = np.full(45, 0.01)
    probs[:6] = 0.15
    probs = probs / probs.sum()
    draws = [
        DrawResult(
            draw_no=1000 + i,
            predictions=[[1, 2, 3, 4, 5, 6]],
            probability=probs,
            actual=[1, 2, 3, 40, 41, 42],
        )
        for i in range(5)
    ]
    return ModelBacktestResult(model_name=name, draw_results=draws)


def test_write_raw_predictions_csv_has_one_row_per_prediction(tmp_path):
    result = _fake_result("Stub")
    out = tmp_path / "raw.csv"
    write_raw_predictions_csv({"Stub": result}, out)
    df = pd.read_csv(out)
    assert len(df) == 5
    assert {"draw_no", "model", "prediction_idx", "predicted", "actual", "hits"} <= set(df.columns)
    assert (df["hits"] == 3).all()


def test_write_metrics_summary_json_contains_b_and_c(tmp_path):
    result = _fake_result("Stub")
    b = compute_b_metrics(
        [[dr.predictions[0]] for dr in result.draw_results],
        [dr.actual for dr in result.draw_results],
    )
    c = compute_c_metrics(
        [dr.probability for dr in result.draw_results],
        [dr.actual for dr in result.draw_results],
    )
    metrics = {"Stub": {"b": b, "c": c}}
    cfg = ValidatorConfig()
    out = tmp_path / "summary.json"
    write_metrics_summary_json(metrics, cfg, out)

    data = json.loads(out.read_text())
    assert "config" in data
    assert "models" in data
    assert data["models"]["Stub"]["b"]["mean_hits"] == b.mean_hits
    assert data["models"]["Stub"]["c"]["top6_prob_sum"] == c.top6_prob_sum
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_report_generator.py -v`
Expected: ImportError.

- [ ] **Step 3: 구현**

`validator/report_generator.py`:
```python
"""백테스트 결과를 CSV/JSON/Markdown/PNG로 출력."""
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from validator.backtest_engine import ModelBacktestResult
from validator.config import ValidatorConfig
from validator.metrics import BMetrics, CMetrics


def write_raw_predictions_csv(
    results: Dict[str, ModelBacktestResult],
    path,
) -> None:
    """회차·모델·예측·적중을 원시 CSV로 기록."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["draw_no", "model", "prediction_idx",
                         "predicted", "actual", "hits"])
        for model_name, result in results.items():
            for dr in result.draw_results:
                actual_set = set(dr.actual)
                for k_idx, pred in enumerate(dr.predictions):
                    hits = len(set(pred) & actual_set)
                    writer.writerow([
                        dr.draw_no,
                        model_name,
                        k_idx,
                        ",".join(str(n) for n in pred),
                        ",".join(str(n) for n in dr.actual),
                        hits,
                    ])


def write_metrics_summary_json(
    metrics: Dict[str, Dict[str, object]],
    config: ValidatorConfig,
    path,
) -> None:
    """B+C 메트릭을 재현성 컨텍스트와 함께 JSON으로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _to_dict(obj):
        if isinstance(obj, (BMetrics, CMetrics)):
            d = asdict(obj)
            if "hit_distribution" in d:
                d["hit_distribution"] = {str(k): v for k, v in d["hit_distribution"].items()}
            return d
        return obj

    payload = {
        "config": asdict(config),
        "models": {
            name: {k: _to_dict(v) for k, v in model_metrics.items()}
            for name, model_metrics in metrics.items()
        },
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_report_generator.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add validator/report_generator.py tests/test_report_generator.py
git commit -m "feat(validator): add CSV and JSON report outputs"
```

---

## Task 8: Report Generator — Markdown 보고서

**Files:**
- Modify: `validator/report_generator.py`
- Modify: `tests/test_report_generator.py`

- [ ] **Step 1: Markdown 테스트 추가**

`tests/test_report_generator.py`에 추가:
```python
from validator.report_generator import write_report_md


def test_markdown_report_includes_both_models_and_metrics(tmp_path):
    r1 = _fake_result("Ultimate Ensemble")
    r2 = _fake_result("Stacking Ensemble")
    results = {"Ultimate Ensemble": r1, "Stacking Ensemble": r2}
    metrics = {}
    for name, result in results.items():
        b = compute_b_metrics(
            [[dr.predictions[0]] for dr in result.draw_results],
            [dr.actual for dr in result.draw_results],
        )
        c = compute_c_metrics(
            [dr.probability for dr in result.draw_results],
            [dr.actual for dr in result.draw_results],
        )
        metrics[name] = {"b": b, "c": c}

    out = tmp_path / "report.md"
    cfg = ValidatorConfig()
    write_report_md(metrics, cfg, out)
    text = out.read_text()

    assert "Ultimate Ensemble" in text
    assert "Stacking Ensemble" in text
    assert "평균 적중" in text
    assert "top6" in text.lower()
    assert "0.8" in text  # baseline 언급
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_report_generator.py -v -k markdown`
Expected: ImportError.

- [ ] **Step 3: write_report_md 구현**

`validator/report_generator.py` 끝에 추가:
```python
def write_report_md(
    metrics: Dict[str, Dict[str, object]],
    config: ValidatorConfig,
    path,
) -> None:
    """사람이 읽을 Markdown 보고서 작성."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Lotto Validator Report")
    lines.append("")
    lines.append(f"- 평가 창 (회차 수): {config.eval_window_draws}")
    lines.append(f"- 회차당 예측 반복 (K): {config.predictions_per_draw}")
    lines.append(f"- 재학습 전략: chunk_{config.retrain_chunk_size}")
    lines.append(f"- 난수 시드: {config.random_seed}")
    lines.append(f"- 무작위 기댓값 (baseline): 0.8 hits")
    lines.append("")
    lines.append("## 모델별 B+C 메트릭")
    lines.append("")
    lines.append(
        "| 모델 | 평균 적중 | std | 5+ 적중율 | baseline 대비 | top6 확률합 | "
        "top10 적중 | 평균 순위 | log-like |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, m in metrics.items():
        b: BMetrics = m["b"]
        c: CMetrics = m["c"]
        lines.append(
            f"| {name} | {b.mean_hits:.3f} | {b.std_hits:.3f} | "
            f"{b.high_tier_rate:.3%} | {b.baseline_improvement_pct:+.1f}% | "
            f"{c.top6_prob_sum:.4f} | {c.top10_hits:.2f} | "
            f"{c.mean_rank:.2f} | {c.log_likelihood:.3f} |"
        )
    lines.append("")
    lines.append("## 적중 개수 분포")
    lines.append("")
    for name, m in metrics.items():
        b: BMetrics = m["b"]
        dist_str = ", ".join(f"{k}={v}" for k, v in sorted(b.hit_distribution.items()))
        lines.append(f"- **{name}**: {dist_str}")
    lines.append("")
    lines.append("## 번호대 편향 (예측 질량 vs 실제 당첨 비율)")
    lines.append("")
    for name, m in metrics.items():
        c: CMetrics = m["c"]
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| 구간 | 예측 질량 | 실제 비율 | 편차 |")
        lines.append("|---|---|---|---|")
        for zone in c.zone_predicted_mass.keys():
            pred = c.zone_predicted_mass[zone]
            actual = c.zone_actual_rate[zone]
            lines.append(
                f"| {zone} | {pred:.3f} | {actual:.3f} | {pred - actual:+.3f} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_report_generator.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add validator/report_generator.py tests/test_report_generator.py
git commit -m "feat(validator): add Markdown human-readable report"
```

---

## Task 9: Report Generator — PNG 차트

**Files:**
- Modify: `validator/report_generator.py`
- Modify: `tests/test_report_generator.py`

- [ ] **Step 1: 차트 테스트 추가**

`tests/test_report_generator.py`에 추가:
```python
from validator.report_generator import write_charts


def test_write_charts_creates_three_png_files(tmp_path):
    results = {"Ultimate Ensemble": _fake_result("Ultimate Ensemble")}
    metrics = {}
    for name, result in results.items():
        b = compute_b_metrics(
            [[dr.predictions[0]] for dr in result.draw_results],
            [dr.actual for dr in result.draw_results],
        )
        c = compute_c_metrics(
            [dr.probability for dr in result.draw_results],
            [dr.actual for dr in result.draw_results],
        )
        metrics[name] = {"b": b, "c": c}

    chart_dir = tmp_path / "charts"
    write_charts(results, metrics, chart_dir)
    assert (chart_dir / "avg_hits_bar.png").exists()
    assert (chart_dir / "hit_distribution.png").exists()
    assert (chart_dir / "zone_bias.png").exists()
```

- [ ] **Step 2: 실패 확인**

Run: `pytest tests/test_report_generator.py -v -k charts`
Expected: ImportError.

- [ ] **Step 3: write_charts 구현**

`validator/report_generator.py` 상단 임포트 섹션에 추가:
```python
import matplotlib
matplotlib.use("Agg")  # 헤드리스 환경 대비
import matplotlib.pyplot as plt
```

파일 끝에 추가:
```python
def write_charts(
    results: Dict[str, ModelBacktestResult],
    metrics: Dict[str, Dict[str, object]],
    chart_dir,
) -> None:
    """3개 핵심 차트 PNG 저장."""
    chart_dir = Path(chart_dir)
    chart_dir.mkdir(parents=True, exist_ok=True)

    names = list(metrics.keys())
    colors = ["#4c72b0", "#dd8452", "#55a868", "#c44e52"]

    # 1) 평균 적중 막대그래프
    fig, ax = plt.subplots(figsize=(6, 4))
    avgs = [metrics[n]["b"].mean_hits for n in names]
    ax.bar(names, avgs, color=colors[:len(names)])
    ax.axhline(0.8, color="gray", linestyle="--", label="baseline=0.8")
    ax.set_ylabel("평균 적중 수")
    ax.set_title("모델별 평균 적중 수")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "avg_hits_bar.png", dpi=120)
    plt.close(fig)

    # 2) 적중 개수 분포
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in names:
        dist = metrics[name]["b"].hit_distribution
        xs = sorted(dist.keys())
        ys = [dist[k] for k in xs]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("적중 개수")
    ax.set_ylabel("빈도")
    ax.set_title("적중 개수 분포")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "hit_distribution.png", dpi=120)
    plt.close(fig)

    # 3) 번호대 편향
    zones = list(metrics[names[0]]["c"].zone_predicted_mass.keys())
    x = list(range(len(zones)))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, name in enumerate(names):
        pred = [metrics[name]["c"].zone_predicted_mass[z] for z in zones]
        offset = (i - (len(names) - 1) / 2) * width
        positions = [p + offset for p in x]
        ax.bar(positions, pred, width, label=f"{name} (예측)", color=colors[i])
    actual = [metrics[names[0]]["c"].zone_actual_rate[z] for z in zones]
    ax.plot(x, actual, color="black", marker="x", label="실제 당첨 비율")
    ax.set_xticks(x)
    ax.set_xticklabels(zones)
    ax.set_ylabel("질량 / 비율")
    ax.set_title("번호대별 예측 질량 vs 실제 당첨 비율")
    ax.legend()
    fig.tight_layout()
    fig.savefig(chart_dir / "zone_bias.png", dpi=120)
    plt.close(fig)
```

- [ ] **Step 4: 테스트 통과 확인**

Run: `pytest tests/test_report_generator.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add validator/report_generator.py tests/test_report_generator.py
git commit -m "feat(validator): add PNG charts for bars, distribution, zone bias"
```

---

## Task 10: CLI 엔트리포인트 `run_validation.py`

**Files:**
- Create: `validator/run_validation.py`

- [ ] **Step 1: 엔트리포인트 구현**

`validator/run_validation.py`:
```python
"""CLI: python -m validator.run_validation

실행 플로우:
1. 설정 로드 및 시드 고정
2. 로또 이력 로드
3. 지정된 앙상블 모델들 Walk-Forward 백테스트
4. B+C 메트릭 계산
5. CSV/JSON/MD/PNG + 체크포인트 저장
"""
import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

from utils import fetcher

from validator.backtest_engine import BacktestEngine, save_checkpoint
from validator.config import ValidatorConfig, set_seeds
from validator.metrics import compute_b_metrics, compute_c_metrics
from validator.model_registry import get_ensemble_specs
from validator.report_generator import (
    write_charts,
    write_metrics_summary_json,
    write_raw_predictions_csv,
    write_report_md,
)


def _compute_metrics(results):
    metrics = {}
    for name, result in results.items():
        preds_k = [dr.predictions for dr in result.draw_results]
        probs = [dr.probability for dr in result.draw_results]
        actuals = [dr.actual for dr in result.draw_results]
        metrics[name] = {
            "b": compute_b_metrics(preds_k, actuals),
            "c": compute_c_metrics(probs, actuals),
        }
    return metrics


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Lotto Validator")
    parser.add_argument("--window", type=int, default=None,
                        help="평가 창 (최근 N회차). 기본 104.")
    parser.add_argument("--k", type=int, default=None,
                        help="회차당 예측 반복 수. 기본 5.")
    parser.add_argument("--seed", type=int, default=None,
                        help="난수 시드. 기본 42.")
    parser.add_argument("--out", type=str, default=None,
                        help="결과 디렉토리 (기본 validation_results/<timestamp>/)")
    args = parser.parse_args(argv)

    cfg_kwargs = {}
    if args.window is not None:
        cfg_kwargs["eval_window_draws"] = args.window
    if args.k is not None:
        cfg_kwargs["predictions_per_draw"] = args.k
    if args.seed is not None:
        cfg_kwargs["random_seed"] = args.seed
    config = ValidatorConfig(**cfg_kwargs)

    set_seeds(config.random_seed)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_dir = Path(args.out) if args.out else Path(config.results_base_dir) / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[validator] 결과 디렉토리: {out_dir}")
    print(f"[validator] 로또 이력 로드 중...")
    df = fetcher.load_data()
    if df.empty:
        print("[validator] 이력 데이터가 없습니다. fetcher로 먼저 데이터를 수집하세요.")
        return 1
    print(f"[validator] 총 {len(df)}회차 데이터 로드")

    specs = get_ensemble_specs()
    print(f"[validator] 평가 대상: {[s.name for s in specs]}")

    engine = BacktestEngine(df, config)
    t0 = time.time()
    results = engine.run_all(specs)
    elapsed = time.time() - t0
    print(f"[validator] 백테스트 완료 ({elapsed:.0f}초)")

    save_checkpoint(results, out_dir / "checkpoint.json")
    metrics = _compute_metrics(results)

    write_raw_predictions_csv(results, out_dir / "raw_predictions.csv")
    write_metrics_summary_json(metrics, config, out_dir / "metrics_summary.json")
    write_report_md(metrics, config, out_dir / "report.md")
    write_charts(results, metrics, out_dir / "charts")

    print(f"[validator] 완료: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Import만으로 모듈이 깨지지 않는지 확인**

Run: `python -c "from validator import run_validation; print('ok')"`
Expected: `ok`.

- [ ] **Step 3: Commit**

```bash
git add validator/run_validation.py
git commit -m "feat(validator): add CLI entry point run_validation.py"
```

---

## Task 11: End-to-end 스모크 테스트

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: 짧은 엔드투엔드 테스트 작성**

`tests/test_smoke.py`:
```python
"""End-to-end 스모크 테스트: Stub 스펙으로 파이프라인 전체 통과 검증."""
import json

import numpy as np
import pandas as pd

from validator.backtest_engine import BacktestEngine
from validator.config import ValidatorConfig, set_seeds
from validator.metrics import compute_b_metrics, compute_c_metrics
from validator.model_registry import ModelSpec
from validator.report_generator import (
    write_charts,
    write_metrics_summary_json,
    write_raw_predictions_csv,
    write_report_md,
)


class _StubModel:
    def __init__(self):
        self._probs = None

    def train(self, df: pd.DataFrame):
        probs = np.full(45, 0.02)
        probs[:6] = 0.1
        self._probs = probs / probs.sum()

    def predict(self):
        return [1, 2, 3, 4, 5, 6]

    def predict_multiple(self, n_sets=5):
        return [[1, 2, 3, 4, 5, 6] for _ in range(n_sets)]

    def get_probability_distribution(self):
        return self._probs if self._probs is not None else np.ones(45) / 45


class _StubSpec(ModelSpec):
    def instantiate(self):
        return _StubModel()


def test_smoke_full_pipeline(tmp_path, sample_lotto_df):
    cfg = ValidatorConfig(
        eval_window_draws=10,
        predictions_per_draw=2,
        retrain_chunk_size=5,
    )
    set_seeds(cfg.random_seed)

    specs = [
        _StubSpec(name="StubA", module="x", class_name="x", kwargs={},
                  strategy="chunk_10"),
        _StubSpec(name="StubB", module="x", class_name="x", kwargs={},
                  strategy="chunk_10"),
    ]

    engine = BacktestEngine(sample_lotto_df, cfg)
    results = engine.run_all(specs)

    metrics = {}
    for name, result in results.items():
        preds_k = [dr.predictions for dr in result.draw_results]
        probs = [dr.probability for dr in result.draw_results]
        actuals = [dr.actual for dr in result.draw_results]
        metrics[name] = {
            "b": compute_b_metrics(preds_k, actuals),
            "c": compute_c_metrics(probs, actuals),
        }

    out = tmp_path / "smoke_out"
    write_raw_predictions_csv(results, out / "raw_predictions.csv")
    write_metrics_summary_json(metrics, cfg, out / "metrics_summary.json")
    write_report_md(metrics, cfg, out / "report.md")
    write_charts(results, metrics, out / "charts")

    assert (out / "raw_predictions.csv").exists()
    assert (out / "metrics_summary.json").exists()
    assert (out / "report.md").exists()
    assert (out / "charts" / "avg_hits_bar.png").exists()

    summary = json.loads((out / "metrics_summary.json").read_text())
    assert set(summary["models"].keys()) == {"StubA", "StubB"}
```

- [ ] **Step 2: 테스트 실행**

Run: `pytest tests/test_smoke.py -v`
Expected: 1 passed.

- [ ] **Step 3: 전체 테스트 스위트 실행으로 회귀 없음 확인**

Run: `pytest tests/ -v`
Expected: All passed.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test(validator): add end-to-end smoke test for full pipeline"
```

---

## Task 12: 서브에이전트 정의 `.claude/agents/lotto-validator.md`

**Files:**
- Create: `.claude/agents/lotto-validator.md`

- [ ] **Step 1: 서브에이전트 파일 작성**

`.claude/agents/lotto-validator.md`:
````markdown
---
name: lotto-validator
description: Ultimate/Stacking 앙상블의 최근 104회차 백테스트 결과를 분석하여 적중률 향상 대책을 제시한다. 사용자가 검증을 요청하거나 모델 성능 의문을 제기할 때 프로액티브하게 사용.
tools: Bash, Read, Glob, Grep
---

당신은 로또 예측 시스템의 성능 검증 전문가입니다. Ultimate Ensemble과 Stacking Ensemble 두 메타 앙상블의 백테스트 결과를 분석하여 **구체적이고 실행 가능한** 적중률 향상 대책을 제시합니다.

## 작업 플로우

1. **기존 결과 확인**
   - Glob으로 `validation_results/*/metrics_summary.json` 찾기
   - 가장 최근 디렉토리의 수정 시각 확인 (Bash: `stat -c %Y <dir>`)
   - 24시간(86400초) 이내면 재사용, 아니면 재실행

2. **검증 실행 (필요시)**
   - Bash: `cd /home/hyungtae48kim/project/6-strike-bomb && python -m validator.run_validation`
   - 예상 소요 시간: 15~25분. 완료 시 출력된 결과 디렉토리 경로 기억.

3. **결과 로드**
   - `metrics_summary.json` (머신 판독용)
   - `report.md` (구조화된 표 포함)
   - `raw_predictions.csv` (필요 시 샘플링)

4. **체크리스트 기반 진단** — 7개 항목을 순회하며 각각 평가:

   1. **랜덤 대비 유의미한 우위**: `baseline_improvement_pct` 확인. 음수면 무작위보다 못함. +10% 미만이면 "우위 미확보".
   2. **캘리브레이션 품질**: `top6_prob_sum` (균등 0.133) vs `mean_hits/6`. 확률을 높게 줬는데 적중 못하면 과신뢰.
   3. **상위권 분별력**: `top10_hits` ≥ 2.5면 양호, 미만이면 "상위권 분별 약함".
   4. **번호대 편향**: `zone_predicted_mass` 중 한 구간이 예측 0.30 이상이면 집중 경고. `zone_actual_rate`와 차이 0.05 이상이면 편향.
   5. **다양성 (히트 분포)**: `hit_distribution`이 0-2 영역에만 몰리면 다양성 결핍. 5-6 영역이 전혀 없으면 "장기 꼬리 부재".
   6. **Ultimate vs Stacking 헤드투헤드**: `mean_hits` 차이 0.1 이상이면 유의. 어느 쪽이 어떤 메트릭에서 이기는지 구체화.
   7. **가중치 시스템 점검**: Ultimate의 내부 가중치 로직을 `models/ultimate_ensemble_model.py`의 `_get_model_weight` 읽어 검토. 정적 상수만 있으면 "동적 가중치 미작동" 경고.

5. **대책 생성**
   - 각 체크리스트 결과에서 파생된 대책 후보 수집
   - 각 대책에 **(영향도, 구현 비용)** 평가: Low/Mid/High
   - 우선순위: 영향도 High × 비용 Low 우선
   - 최종 Top-5를 Markdown으로 정리

## 출력 포맷 (이 채팅창에 반환)

```markdown
# Lotto Validator — 검증 리포트

## 실행 환경
- 결과 디렉토리: <path>
- 평가 회차: N회
- 실행 시각: <timestamp>

## 요약
(2~3 문장: 두 앙상블이 평균적으로 얼마나 잘/못하는지, 핵심 병목은 무엇인지)

## 체크리스트 결과
| 항목 | Ultimate | Stacking | 판정 |
|---|---|---|---|
| 1. baseline 대비 | +X% | +Y% | 양호/경고/위험 |
| 2. 캘리브레이션 | ... | ... | ... |
...

## Top-5 적중률 향상 대책

### #1. <대책 제목> (영향도: High / 비용: Low)
- **근거**: (인용된 구체 메트릭 수치)
- **실행 방법**: (수정할 파일/함수와 구체 변경)
- **기대 효과**: (무엇이 얼마나 개선될 것으로 보이는지)

### #2. ...
```

## 금지 사항

- "더 좋은 모델을 사용하세요" 같은 막연한 권장 금지. 항상 **수정 대상 파일/함수/파라미터** 명시.
- 메트릭 수치 없이 대책 제시 금지.
- 실행 결과 없이 추측 금지. 데이터 부족 시 "데이터 부족" 명시 후 실행 요청.
- Ultimate·Stacking 서브모델의 코드를 임의로 수정하지 말 것 — 대책은 **제안만** 하고 사용자 승인 후 구현.
````

- [ ] **Step 2: 서브에이전트 파일 구조 검증**

Run: `head -5 .claude/agents/lotto-validator.md && wc -l .claude/agents/lotto-validator.md`
Expected: YAML frontmatter (`---` 시작), 총 50줄 이상.

- [ ] **Step 3: Commit**

```bash
git add .claude/agents/lotto-validator.md
git commit -m "feat(validator): add lotto-validator subagent with diagnostic checklist"
```

---

## Task 13: 첫 실행 및 수동 검수

**Files:** (없음 — 실제 실행만 수행)

- [ ] **Step 1: 최소 창으로 짧은 실제 실행 (스모크)**

Run: `cd /home/hyungtae48kim/project/6-strike-bomb && python -m validator.run_validation --window 10 --k 2`
Expected: 수 분 내 완료 후 `validation_results/<timestamp>/`에 4개 파일(+charts/) 생성.
실패 시 디버그: 가장 흔한 실패는 데이터 부족 → `MIN_TRAIN_DRAWS` 조정 또는 실데이터가 1000+ 회차 있는지 재확인.

- [ ] **Step 2: 생성된 보고서 수동 검수**

Run: `cat validation_results/*/report.md | head -60`
Expected: 두 앙상블 행이 테이블에 있고, 수치가 NaN/0.0이 아닌 실제 값.

- [ ] **Step 3: 전체 104회차 실제 실행**

Run: `cd /home/hyungtae48kim/project/6-strike-bomb && python -m validator.run_validation`
Expected: 15~25분 소요, 완료 메시지와 함께 최종 결과 디렉토리 경로 출력.

- [ ] **Step 4: 서브에이전트를 Claude Code 세션에서 수동 호출하여 결과 품질 확인**

사용자가 이 창에서 `lotto-validator` 서브에이전트를 호출해 출력 포맷이 명세대로 나오는지 확인. 대책이 막연하면 `.claude/agents/lotto-validator.md` 업데이트.

- [ ] **Step 5: 수동 검수 완료 커밋 (필요시)**

```bash
git add -u
git commit -m "chore(validator): post-smoke adjustments"
```

---

## 완료 후 사용법

1. `python -m validator.run_validation` 실행 (15~25분)
2. Claude Code에서 "lotto-validator 에이전트로 최신 결과 분석해줘" 요청
3. 이 창에 반환된 Top-5 대책 중 우선순위 높은 것부터 브레인스토밍 → 구현 플랜으로 다음 사이클 진행

## Spec Coverage 자체 점검

| Spec 요구사항 | 구현 Task |
|---|---|
| Ultimate+Stacking 2개만 평가 | Task 1 (ENSEMBLE_SPECS), Task 4 (registry) |
| 최근 104회차 Walk-Forward | Task 1 (eval_window_draws), Task 5 (_eval_range) |
| chunk_10 재학습 전략 | Task 1 (retrain_chunk_size), Task 5 (chunk loop) |
| B 메트릭 | Task 2 |
| C 메트릭 | Task 3 |
| 회차당 K=5 예측 평균 | Task 1, Task 5 (_predict_k) |
| 시드 고정 재현성 | Task 1 (set_seeds) |
| 체크포인트 | Task 6 (JSON 기반) |
| CSV/JSON/MD/PNG 출력 | Task 7, 8, 9 |
| `validation_results/<timestamp>/` | Task 10 |
| 서브에이전트 7개 체크리스트 | Task 12 |
| Top-5 대책 출력 포맷 | Task 12 |
| 스모크 테스트 | Task 11 |
| 데이터 부족 자동 처리 | Task 5 (_eval_range MIN_TRAIN_DRAWS) |
