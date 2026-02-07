from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DataConfig:
    dataset_path: str
    split_name: str = "default"
    load_dataframe: bool = True


@dataclass(frozen=True, slots=True)
class ModelConfig:
    n_components: int = 4
    covariance_type: str = "full"
    transmat_prior_strength: float = 20.0
    transmat_prior_mode: str = "sticky_diag"
    transmat_prior_custom: list[list[float]] | None = None


@dataclass(frozen=True, slots=True)
class TrainConfig:
    n_init: int = 5
    n_iter: int = 300
    tol: float = 1e-3
    init_strategy: str = "kmeans"
    random_seed: int | None = None


@dataclass(frozen=True, slots=True)
class PreprocessConfig:
    scaler: str = "standard"
    winsorize_vol: bool = False


@dataclass(frozen=True, slots=True)
class EvalConfig:
    eval_scheme: str = "last_n_days"
    eval_last_n_days: int = 252


@dataclass(frozen=True, slots=True)
class HmmConfig:
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    preprocess: PreprocessConfig
    eval: EvalConfig


def parse_config(data_spec: Mapping[str, Any] | None) -> HmmConfig:
    data_spec = data_spec or {}
    data_block = data_spec.get("data", {}) or {}
    model_block = data_spec.get("model", {}) or {}
    train_block = data_spec.get("train", {}) or {}
    preprocess_block = data_spec.get("preprocess", {}) or {}
    eval_block = data_spec.get("eval", {}) or {}

    dataset_path = (
        data_block.get("dataset_path")
        or data_spec.get("dataset_path")
        or os.getenv("HMM_DATASET_PATH")
        or ""
    )
    dataset_path = str(dataset_path)
    if not dataset_path:
        raise ValueError(
            "Dataset path not provided. Set data.dataset_path in RunSpec.data_spec "
            "or define HMM_DATASET_PATH in the environment."
        )

    split_name = str(data_block.get("split_name") or data_spec.get("split_name") or "default")
    load_dataframe = bool(data_block.get("load_dataframe", True))

    model = ModelConfig(
        n_components=int(model_block.get("n_components", 4)),
        covariance_type=str(model_block.get("covariance_type", "full")),
        transmat_prior_strength=float(model_block.get("transmat_prior_strength", 20.0)),
        transmat_prior_mode=str(model_block.get("transmat_prior_mode", "sticky_diag")),
        transmat_prior_custom=model_block.get("transmat_prior_custom"),
    )
    train = TrainConfig(
        n_init=int(train_block.get("n_init", 5)),
        n_iter=int(train_block.get("n_iter", 300)),
        tol=float(train_block.get("tol", 1e-3)),
        init_strategy=str(train_block.get("init_strategy", "kmeans")),
        random_seed=_parse_optional_int(train_block.get("random_seed", None)),
    )
    preprocess = PreprocessConfig(
        scaler=str(preprocess_block.get("scaler", "standard")),
        winsorize_vol=bool(preprocess_block.get("winsorize_vol", False)),
    )
    eval_cfg = EvalConfig(
        eval_scheme=str(eval_block.get("eval_scheme", "last_n_days")),
        eval_last_n_days=int(eval_block.get("eval_last_n_days", 252)),
    )

    _validate_model(model)
    _validate_train(train)
    _validate_preprocess(preprocess)
    _validate_eval(eval_cfg)

    return HmmConfig(
        data=DataConfig(
            dataset_path=dataset_path, split_name=split_name, load_dataframe=load_dataframe
        ),
        model=model,
        train=train,
        preprocess=preprocess,
        eval=eval_cfg,
    )


def _parse_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _validate_model(model: ModelConfig) -> None:
    if model.n_components <= 0:
        raise ValueError("model.n_components must be positive")
    # if model.covariance_type != "full":
    #     raise ValueError("model.covariance_type must be 'full' in v1")
    if model.transmat_prior_strength < 0:
        raise ValueError("model.transmat_prior_strength must be >= 0")
    if model.transmat_prior_mode not in {"sticky_diag", "uniform", "custom_matrix"}:
        raise ValueError(
            "model.transmat_prior_mode must be one of sticky_diag|uniform|custom_matrix"
        )


def _validate_train(train: TrainConfig) -> None:
    if train.n_init <= 0:
        raise ValueError("train.n_init must be >= 1")
    if train.n_iter <= 0:
        raise ValueError("train.n_iter must be >= 1")
    if train.tol <= 0:
        raise ValueError("train.tol must be > 0")
    if train.init_strategy not in {"kmeans", "random"}:
        raise ValueError("train.init_strategy must be kmeans|random")


def _validate_preprocess(preprocess: PreprocessConfig) -> None:
    if preprocess.scaler not in {"standard", "robust", "none"}:
        raise ValueError("preprocess.scaler must be standard|robust|none")


def _validate_eval(eval_cfg: EvalConfig) -> None:
    if eval_cfg.eval_scheme not in {"none", "last_n_days", "walkforward"}:
        raise ValueError("eval.eval_scheme must be none|last_n_days|walkforward")
    if eval_cfg.eval_last_n_days <= 0:
        raise ValueError("eval.eval_last_n_days must be positive")
