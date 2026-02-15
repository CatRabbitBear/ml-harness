from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
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


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "ignite_base_zero"
    target_cols: list[str] = Field(default_factory=lambda: ["ignite5", "ignite10", "ignite20"])

    @model_validator(mode="after")
    def _validate_experiment(self) -> ExperimentConfig:
        allowed = {
            "ignite_base_zero",
            "ignite_base_shift1",
            "ignite_rms5_stats_gbr",
            "ignite_pca6_gbr",
            "ignite_pca6_abs12_gbr",
            "ignite_combo15_gbr",
        }
        if self.name not in allowed:
            raise ValueError(f"Unsupported experiment.name: {self.name}")
        if not self.target_cols:
            raise ValueError("experiment.target_cols must not be empty")
        return self


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train_end_date: str = "2021-12-31"
    val_end_date: str = "2022-12-31"
    test_end_date: str = "2024-12-01"


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gbr_n_estimators: int = 300
    gbr_learning_rate: float = 0.05
    gbr_max_depth: int = 3
    gbr_min_samples_leaf: int = 5
    gbr_subsample: float = 0.9

    @model_validator(mode="after")
    def _validate_model(self) -> ModelConfig:
        if self.gbr_n_estimators <= 0:
            raise ValueError("model.gbr_n_estimators must be > 0")
        if self.gbr_learning_rate <= 0:
            raise ValueError("model.gbr_learning_rate must be > 0")
        if self.gbr_max_depth <= 0:
            raise ValueError("model.gbr_max_depth must be > 0")
        if self.gbr_min_samples_leaf <= 0:
            raise ValueError("model.gbr_min_samples_leaf must be > 0")
        if self.gbr_subsample <= 0 or self.gbr_subsample > 1:
            raise ValueError("model.gbr_subsample must be in (0, 1]")
        return self


class PreprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    log_target: bool = False
    target_epsilon: float = 1e-8

    @model_validator(mode="after")
    def _validate_preprocess(self) -> PreprocessConfig:
        if self.target_epsilon <= 0:
            raise ValueError("preprocess.target_epsilon must be > 0")
        return self


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_quantile: float = 0.2

    @model_validator(mode="after")
    def _validate_eval(self) -> EvalConfig:
        if self.top_quantile <= 0 or self.top_quantile >= 1:
            raise ValueError("eval.top_quantile must be in (0, 1)")
        return self


class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    overlay_years: list[int] = Field(default_factory=lambda: [2022, 2023])


class FxRvRegressionParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    split: SplitConfig = Field(default_factory=SplitConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    plots: PlotConfig = Field(default_factory=PlotConfig)


class FxRvRegressionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: DataConfig
    experiment: ExperimentConfig
    split: SplitConfig
    model: ModelConfig
    preprocess: PreprocessConfig
    eval: EvalConfig
    plots: PlotConfig


def default_params() -> dict[str, Any]:
    path = Path(__file__).with_name("default_config.yaml")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    validated = FxRvRegressionParams.model_validate(payload)
    return validated.model_dump(mode="python")


def validate_params(params: Mapping[str, Any] | None, *, strict: bool = True) -> dict[str, Any]:
    raw_params = dict(params or {})
    if strict:
        validated = FxRvRegressionParams.model_validate(raw_params)
        return validated.model_dump(mode="python")

    known_subset = _extract_known_fields(raw_params, FxRvRegressionParams)
    validated = FxRvRegressionParams.model_validate(known_subset)
    return _deep_merge(raw_params, validated.model_dump(mode="python"))


def parse_config(
    data_spec: Mapping[str, Any] | None,
    params: Mapping[str, Any] | None,
    *,
    strict: bool = True,
) -> FxRvRegressionConfig:
    data_cfg = DataConfig.model_validate(dict(data_spec or {}))
    raw_params = dict(params or {})
    if strict:
        parsed_params = FxRvRegressionParams.model_validate(raw_params)
    else:
        known_subset = _extract_known_fields(raw_params, FxRvRegressionParams)
        parsed_params = FxRvRegressionParams.model_validate(known_subset)

    return FxRvRegressionConfig(
        data=data_cfg,
        experiment=parsed_params.experiment,
        split=parsed_params.split,
        model=parsed_params.model,
        preprocess=parsed_params.preprocess,
        eval=parsed_params.eval,
        plots=parsed_params.plots,
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
