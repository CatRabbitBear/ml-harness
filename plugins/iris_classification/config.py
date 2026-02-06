from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class SplitConfig:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15


@dataclass(frozen=True, slots=True)
class ModelConfig:
    type: str = "logreg"
    C: float = 1.0
    max_iter: int = 200


@dataclass(frozen=True, slots=True)
class IrisConfig:
    split: SplitConfig
    model: ModelConfig


def parse_config(data_spec: Mapping[str, Any] | None) -> IrisConfig:
    data_spec = data_spec or {}
    split_spec = data_spec.get("split", {}) or {}
    model_spec = data_spec.get("model", {}) or {}

    split = SplitConfig(
        train=float(split_spec.get("train", 0.7)),
        val=float(split_spec.get("val", 0.15)),
        test=float(split_spec.get("test", 0.15)),
    )
    _validate_split(split)

    model_type = str(model_spec.get("type", "logreg"))
    if model_type != "logreg":
        raise ValueError(f"Unsupported model.type: {model_type}")

    model = ModelConfig(
        type=model_type,
        C=float(model_spec.get("C", 1.0)),
        max_iter=int(model_spec.get("max_iter", 200)),
    )

    return IrisConfig(split=split, model=model)


def _validate_split(split: SplitConfig) -> None:
    if split.train <= 0 or split.val <= 0 or split.test <= 0:
        raise ValueError("split ratios must be positive values")
    total = split.train + split.val + split.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total}")
