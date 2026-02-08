from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunMode = Literal["once", "backfill", "scheduled"]
Pipeline = Literal["train", "score"]


class PluginConfigRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str = Field(min_length=1)


class RunConfigMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)
    seed: int | None = Field(default=None, ge=0)
    strict: bool = True
    pipeline: Pipeline = "train"
    run_mode: RunMode = "once"
    request_id: str | None = None
    notes: str | None = None


class RunDataConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_id: str | None = None
    dataset_path: str | None = None
    split_name: str | None = None


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plugin: PluginConfigRef
    run: RunConfigMeta = Field(default_factory=RunConfigMeta)
    data: RunDataConfig = Field(default_factory=RunDataConfig)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("params", mode="before")
    @classmethod
    def _coerce_params_dict(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if isinstance(value, dict):
            return value
        raise ValueError("params must be a mapping")
