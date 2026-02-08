from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_path: str
    split_name: str = "default"
    load_dataframe: bool = True

    @model_validator(mode="after")
    def _validate_dataset_path(self) -> DataConfig:
        if not self.dataset_path:
            raise ValueError("data.dataset_path is required")
        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_components: int = 4
    covariance_type: str = "full"
    transmat_prior_strength: float = 20.0
    transmat_prior_mode: str = "sticky_diag"
    transmat_prior_custom: list[list[float]] | None = None

    @model_validator(mode="after")
    def _validate_model(self) -> ModelConfig:
        if self.n_components <= 0:
            raise ValueError("model.n_components must be positive")
        if self.transmat_prior_strength < 0:
            raise ValueError("model.transmat_prior_strength must be >= 0")
        if self.transmat_prior_mode not in {"sticky_diag", "uniform", "custom_matrix"}:
            raise ValueError(
                "model.transmat_prior_mode must be one of sticky_diag|uniform|custom_matrix"
            )
        return self


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_init: int = 5
    n_iter: int = 300
    tol: float = 1e-3
    init_strategy: str = "kmeans"
    random_seed: int | None = None

    @model_validator(mode="after")
    def _validate_train(self) -> TrainConfig:
        if self.n_init <= 0:
            raise ValueError("train.n_init must be >= 1")
        if self.n_iter <= 0:
            raise ValueError("train.n_iter must be >= 1")
        if self.tol <= 0:
            raise ValueError("train.tol must be > 0")
        if self.init_strategy not in {"kmeans", "random"}:
            raise ValueError("train.init_strategy must be kmeans|random")
        return self


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scaler: str = "standard"
    winsorize_vol: bool = False

    @model_validator(mode="after")
    def _validate_preprocess(self) -> PreprocessConfig:
        if self.scaler not in {"standard", "robust", "none"}:
            raise ValueError("preprocess.scaler must be standard|robust|none")
        return self


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    eval_scheme: str = "last_n_days"
    eval_last_n_days: int = 252

    @model_validator(mode="after")
    def _validate_eval(self) -> EvalConfig:
        if self.eval_scheme not in {"none", "last_n_days", "walkforward"}:
            raise ValueError("eval.eval_scheme must be none|last_n_days|walkforward")
        if self.eval_last_n_days <= 0:
            raise ValueError("eval.eval_last_n_days must be positive")
        return self


class HmmParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)


class HmmConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    preprocess: PreprocessConfig
    eval: EvalConfig


def default_params() -> dict[str, Any]:
    return HmmParams().model_dump(mode="python")


def validate_params(params: Mapping[str, Any] | None, *, strict: bool = True) -> dict[str, Any]:
    raw_params = dict(params or {})
    if strict:
        validated = HmmParams.model_validate(raw_params)
        return validated.model_dump(mode="python")

    known_subset = _extract_known_fields(raw_params, HmmParams)
    validated = HmmParams.model_validate(known_subset)
    return _deep_merge(raw_params, validated.model_dump(mode="python"))


def parse_config(
    data_spec: Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
    *,
    strict: bool = True,
) -> HmmConfig:
    data_cfg = DataConfig.model_validate(dict(data_spec or {}))
    raw_params = dict(params or {})
    if strict:
        parsed_params = HmmParams.model_validate(raw_params)
    else:
        known_subset = _extract_known_fields(raw_params, HmmParams)
        parsed_params = HmmParams.model_validate(known_subset)
    return HmmConfig(
        data=data_cfg,
        model=parsed_params.model,
        train=parsed_params.train,
        preprocess=parsed_params.preprocess,
        eval=parsed_params.eval,
    )


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
