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
