from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FxRvRegressionConfig
from .data import DataFrameSplits

PCA_COLS = [f"PC{i}" for i in range(1, 7)]
PCA_ABS_COLS = [f"{c}_abs" for c in PCA_COLS]
RMS_STATS_COLS = ["rms5__mean", "rms5__std", "rms5__max"]
RMS_VEC8_COLS = [
    "rms5__GBP",
    "rms5__USD",
    "rms5__EUR",
    "rms5__JPY",
    "rms5__NZD",
    "rms5__CAD",
    "rms5__CHF",
    "rms5__AUD",
]
STRESS_COLS = ["stress__pca_full_norm"]


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    name: str
    model_kind: str
    feature_cols: list[str]
    include_abs_pca: bool


@dataclass(frozen=True, slots=True)
class HorizonResult:
    target_col: str
    feature_cols: list[str]
    model_kind: str
    model: Any | None
    predictions_raw: dict[str, pd.DataFrame]


def normalize_experiment_name(name: str) -> str:
    aliases = {
        "rv_regress_v1_persist": "rv_base_persist_shift1",
        "rv_regress_v1_pca6abs_ridge": "rv_pca6_abs12_ridge",
        "rv_regress_v1_pca6abs_gbr": "rv_pca6_abs12_gbr",
    }
    return aliases.get(name, name)


def resolve_experiment_plan(name: str) -> ExperimentPlan:
    normalized = normalize_experiment_name(name)

    if normalized == "rv_base_persist_shift1":
        return ExperimentPlan(
            name=normalized,
            model_kind="persistence",
            feature_cols=[],
            include_abs_pca=False,
        )

    if normalized == "rv_rms5_stats_ridge":
        return ExperimentPlan(normalized, "ridge", RMS_STATS_COLS, include_abs_pca=False)
    if normalized == "rv_rms5_stats_gbr":
        return ExperimentPlan(normalized, "gbr", RMS_STATS_COLS, include_abs_pca=False)

    if normalized == "rv_rms5_vec8_ridge":
        return ExperimentPlan(normalized, "ridge", RMS_VEC8_COLS, include_abs_pca=False)
    if normalized == "rv_rms5_vec8_gbr":
        return ExperimentPlan(normalized, "gbr", RMS_VEC8_COLS, include_abs_pca=False)

    if normalized == "rv_pca6_ridge":
        return ExperimentPlan(normalized, "ridge", PCA_COLS, include_abs_pca=False)
    if normalized == "rv_pca6_gbr":
        return ExperimentPlan(normalized, "gbr", PCA_COLS, include_abs_pca=False)

    if normalized == "rv_pca6_abs12_ridge":
        return ExperimentPlan(normalized, "ridge", PCA_COLS, include_abs_pca=True)
    if normalized == "rv_pca6_abs12_gbr":
        return ExperimentPlan(normalized, "gbr", PCA_COLS, include_abs_pca=True)

    if normalized == "rv_combo15_ridge":
        return ExperimentPlan(normalized, "ridge", RMS_STATS_COLS + PCA_COLS, include_abs_pca=True)
    if normalized == "rv_combo15_gbr":
        return ExperimentPlan(normalized, "gbr", RMS_STATS_COLS + PCA_COLS, include_abs_pca=True)

    if normalized == "rv_stress1_ridge":
        return ExperimentPlan(normalized, "ridge", STRESS_COLS, include_abs_pca=False)
    if normalized == "rv_stress1_gbr":
        return ExperimentPlan(normalized, "gbr", STRESS_COLS, include_abs_pca=False)

    raise ValueError(f"Unsupported experiment name: {name}")


def run_experiment(
    config: FxRvRegressionConfig,
    splits: DataFrameSplits,
    *,
    seed: int | None,
) -> list[HorizonResult]:
    plan = resolve_experiment_plan(config.experiment.name)

    results: list[HorizonResult] = []
    for target_col in config.experiment.target_cols:
        if plan.model_kind == "persistence":
            predictions = run_persistence(target_col, splits)
            results.append(
                HorizonResult(
                    target_col=target_col,
                    feature_cols=[f"{target_col}_lag1"],
                    model_kind="persistence",
                    model=None,
                    predictions_raw=predictions,
                )
            )
            continue

        feature_cols, frames = _build_feature_frames(
            plan=plan,
            splits=splits,
            target_col=target_col,
            use_log_target=config.preprocess.log_target,
            target_epsilon=config.preprocess.target_epsilon,
        )
        model = _fit_model(
            config=config,
            model_kind=plan.model_kind,
            X_train=frames["train"][feature_cols].to_numpy(),
            y_train=frames["train"]["__target_model__"].to_numpy(),
            seed=seed,
        )
        predictions = {
            split_name: pd.DataFrame(
                {
                    "date": pd.to_datetime(frame["__date__"], utc=True),
                    "y_true_raw": frame["__target_raw__"].to_numpy(),
                    "y_pred_raw": _inverse_target(
                        model.predict(frame[feature_cols].to_numpy()),
                        use_log=config.preprocess.log_target,
                        eps=config.preprocess.target_epsilon,
                    ),
                    "rms5__mean": frame["rms5__mean"].to_numpy()
                    if "rms5__mean" in frame.columns
                    else np.nan,
                },
                index=frame.index,
            )
            for split_name, frame in frames.items()
        }
        results.append(
            HorizonResult(
                target_col=target_col,
                feature_cols=feature_cols,
                model_kind=plan.model_kind,
                model=model,
                predictions_raw=predictions,
            )
        )
    return results


def run_persistence(target_col: str, splits: DataFrameSplits) -> dict[str, pd.DataFrame]:
    all_df = pd.concat(
        [splits.train, splits.val, splits.test],
        axis=0,
    ).copy()
    lag_col = f"{target_col}_lag1"
    all_df[lag_col] = all_df[target_col].shift(1)

    pred_frames: dict[str, pd.DataFrame] = {}
    for split_name, frame in {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }.items():
        joined = all_df.loc[frame.index]
        pred = pd.DataFrame(
            {
                "date": pd.to_datetime(joined["__date__"], utc=True),
                "y_true_raw": joined[target_col].to_numpy(),
                "y_pred_raw": joined[lag_col].to_numpy(),
                "rms5__mean": joined["rms5__mean"].to_numpy()
                if "rms5__mean" in joined.columns
                else np.nan,
            },
            index=joined.index,
        )
        pred = pred.dropna(axis=0, how="any")
        pred_frames[split_name] = pred
    return pred_frames


def required_columns_for_experiment(name: str, targets: list[str]) -> set[str]:
    plan = resolve_experiment_plan(name)
    required = set(targets)
    required.update(plan.feature_cols)
    if plan.model_kind == "persistence":
        required.update(targets)
    if plan.include_abs_pca:
        required.update(PCA_COLS)
    return required


def _build_feature_frames(
    *,
    plan: ExperimentPlan,
    splits: DataFrameSplits,
    target_col: str,
    use_log_target: bool,
    target_epsilon: float,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    feature_cols = list(plan.feature_cols)
    if plan.include_abs_pca:
        feature_cols = feature_cols + PCA_ABS_COLS

    all_needed = list(feature_cols) + [target_col, "__date__"]
    for col in all_needed:
        if col in splits.train.columns:
            continue
        if col in PCA_ABS_COLS:
            continue
        raise ValueError(f"Required column missing for experiment '{plan.name}': {col}")

    frames: dict[str, pd.DataFrame] = {}
    for split_name, source in {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }.items():
        frame = source.copy()
        if plan.include_abs_pca:
            for col in PCA_COLS:
                if col not in frame.columns:
                    raise ValueError(f"Missing PCA column in {split_name} split: {col}")
                frame[f"{col}_abs"] = frame[col].abs()

        frame["__target_raw__"] = frame[target_col].astype("float64")
        frame["__target_model__"] = _transform_target(
            frame["__target_raw__"].to_numpy(),
            use_log=use_log_target,
            eps=target_epsilon,
        )
        frames[split_name] = frame

    return feature_cols, frames


def _fit_model(
    config: FxRvRegressionConfig,
    *,
    model_kind: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int | None,
):
    if model_kind == "ridge":
        from sklearn.linear_model import Ridge

        return Ridge(alpha=config.model.ridge_alpha).fit(X_train, y_train)

    if model_kind == "gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(
            n_estimators=config.model.gbr_n_estimators,
            learning_rate=config.model.gbr_learning_rate,
            max_depth=config.model.gbr_max_depth,
            min_samples_leaf=config.model.gbr_min_samples_leaf,
            subsample=config.model.gbr_subsample,
            random_state=seed,
        ).fit(X_train, y_train)

    raise ValueError(f"Unsupported model kind: {model_kind}")


def _transform_target(values: np.ndarray, *, use_log: bool, eps: float) -> np.ndarray:
    typed = values.astype("float64")
    if not use_log:
        return typed
    return np.log(np.clip(typed + eps, a_min=eps, a_max=None))


def _inverse_target(values: np.ndarray, *, use_log: bool, eps: float) -> np.ndarray:
    if not use_log:
        return values.astype("float64")
    return np.exp(values.astype("float64")) - eps
