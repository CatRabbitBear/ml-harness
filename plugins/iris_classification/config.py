from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    @model_validator(mode="after")
    def _validate_ratios(self) -> SplitConfig:
        if self.train <= 0 or self.val <= 0 or self.test <= 0:
            raise ValueError("split ratios must be positive values")
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = "logreg"
    C: float = 1.0
    max_iter: int = 200

    @model_validator(mode="after")
    def _validate_type(self) -> ModelConfig:
        if self.type != "logreg":
            raise ValueError(f"Unsupported model.type: {self.type}")
        return self


class IrisParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)


def default_params() -> dict[str, Any]:
    return IrisParams().model_dump(mode="python")


def validate_params(params: Mapping[str, Any] | None, *, strict: bool = True) -> dict[str, Any]:
    raw_params = dict(params or {})
    if strict:
        validated = IrisParams.model_validate(raw_params)
        return validated.model_dump(mode="python")

    known_subset = _extract_known_fields(raw_params, IrisParams)
    validated = IrisParams.model_validate(known_subset)
    return _deep_merge(raw_params, validated.model_dump(mode="python"))


def parse_config(params: Mapping[str, Any] | None, *, strict: bool = True) -> IrisParams:
    raw_params = dict(params or {})
    if strict:
        return IrisParams.model_validate(raw_params)
    known_subset = _extract_known_fields(raw_params, IrisParams)
    return IrisParams.model_validate(known_subset)


def _extract_known_fields(data: Mapping[str, Any], model_type: type[BaseModel]) -> dict[str, Any]:
    known: dict[str, Any] = {}
    for field_name, field in model_type.model_fields.items():
        if field_name not in data:
            continue
        value = data[field_name]
        nested_type = _as_model_type(field.annotation)
        if nested_type is not None and isinstance(value, Mapping):
            known[field_name] = _extract_known_fields(value, nested_type)
            continue
        known[field_name] = value
    return known


def _as_model_type(annotation: Any) -> type[BaseModel] | None:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation
    return None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged
