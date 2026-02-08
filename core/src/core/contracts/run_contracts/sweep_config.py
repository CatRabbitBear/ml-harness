from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SweepDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["grid"] = "grid"
    overrides: dict[str, list[Any]] = Field(default_factory=dict)

    @field_validator("overrides")
    @classmethod
    def _validate_overrides(cls, value: dict[str, list[Any]]) -> dict[str, list[Any]]:
        if not value:
            raise ValueError("sweep.overrides must not be empty")
        for path, choices in value.items():
            if not path or "." not in path:
                raise ValueError(
                    "sweep.overrides keys must be dot-paths like 'params.model.n_components'"
                )
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError(f"sweep.overrides.{path} must be a non-empty list")
        return value


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sweep: SweepDefinition
