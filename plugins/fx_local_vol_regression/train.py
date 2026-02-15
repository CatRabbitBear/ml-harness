from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import FxLocalVolConfig
from .data import DataFrameSplits

GLOBAL_RMS_CANDIDATES = [
    "rms5__mean",
    "rms5__std",
    "rms5__max",
    "rms5__global_mean",
    "rms5__global_std",
    "rms5__global_max",
]
IMPULSE_AGG_CANDIDATES = [
    "abs_latent_mean",
    "max_abs_latent",
    "abs_lat_ret_mean",
    "max_abs_lat_ret",
]
LAT_HIST_ABS_TEMPLATE = [
    "abs_lat_ret__JPY",
    "abs_lat_ret__JPY_lag1",
    "abs_lat_ret__JPY_lag2",
    "abs_lat_ret__JPY_lag3",
    "abs_lat_ret__JPY_lag4",
    "abs_lat_ret__JPY_lag5",
    "abs_latent_mean",
]
LAT_HIST_SIGNED_TEMPLATE = [
    "lat_ret__JPY",
    "lat_ret__JPY_lag1",
    "lat_ret__JPY_lag2",
    "lat_ret__JPY_lag3",
    "lat_ret__JPY_lag4",
    "lat_ret__JPY_lag5",
    "max_abs_latent",
]
LAT_HIST_PLUS_GLOBAL_TEMPLATE = [
    "abs_lat_ret__JPY",
    "abs_lat_ret__JPY_lag1",
    "abs_lat_ret__JPY_lag2",
    "abs_lat_ret__JPY_lag3",
    "abs_lat_ret__JPY_lag4",
    "abs_lat_ret__JPY_lag5",
    "abs_latent_mean",
    "lat_ret__JPY",
    "lat_ret__JPY_lag1",
    "lat_ret__JPY_lag2",
    "lat_ret__JPY_lag3",
    "lat_ret__JPY_lag4",
    "lat_ret__JPY_lag5",
    "max_abs_latent",
    "rms5__global_max",
    "rms5__global_mean",
    "rms5__global_std",
]
LAT_RMS_NUANCED_TEMPLATE = [
    "abs_lat_ret__JPY",
    "abs_latent_mean",
    "rms10__JPY",
    "rms20__JPY",
    "rms3__JPY",
    "rms3_over_rms20__JPY",
    "rms5__JPY",
    "rms5__global_max",
    "rms5__global_mean",
    "rms5__global_std",
    "rms5_minus_rms20__JPY",
]


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    out_col: str
    source_col: str
    abs_transform: bool


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    name: str
    model_kind: str
    feature_level: str | None


@dataclass(frozen=True, slots=True)
class HorizonResult:
    target_col: str
    feature_cols: list[str]
    model_kind: str
    model: Any | None
    predictions_raw: dict[str, pd.DataFrame]


def normalize_experiment_name(name: str) -> str:
    return name


def resolve_experiment_plan(name: str) -> ExperimentPlan:
    if name == "local_base_zero":
        return ExperimentPlan(name=name, model_kind="zero", feature_level=None)
    if name == "local_base_shift1":
        return ExperimentPlan(name=name, model_kind="shift1", feature_level=None)
    if name in {"local_rms_stats_gbr", "local_features_a_gbr"}:
        return ExperimentPlan(name=name, model_kind="gbr", feature_level="A")
    if name == "local_features_b_gbr":
        return ExperimentPlan(name=name, model_kind="gbr", feature_level="B")
    if name == "local_features_c_gbr":
        return ExperimentPlan(name=name, model_kind="gbr", feature_level="C")
    if name == "local_lat_hist_abs_gbr":
        return ExperimentPlan(name=name, model_kind="gbr", feature_level=None)
    if name == "local_lat_hist_signed_gbr":
        return ExperimentPlan(name=name, model_kind="gbr", feature_level=None)
    if name == "local_lat_hist_plus_global_gbr":
        return ExperimentPlan(name=name, model_kind="gbr", feature_level=None)
    if name == "local_lat_rms_nuanced_gbr":
        return ExperimentPlan(name=name, model_kind="gbr", feature_level=None)
    raise ValueError(f"Unsupported experiment name: {name}")


def run_experiment(
    config: FxLocalVolConfig,
    splits: DataFrameSplits,
    *,
    seed: int | None,
) -> list[HorizonResult]:
    plan = resolve_experiment_plan(config.experiment.name)

    results: list[HorizonResult] = []
    for target_col in config.experiment.target_cols:
        if plan.model_kind == "zero":
            results.append(
                HorizonResult(
                    target_col=target_col,
                    feature_cols=["constant_zero"],
                    model_kind="zero",
                    model=None,
                    predictions_raw=run_zero_baseline(target_col, splits),
                )
            )
            continue

        if plan.model_kind == "shift1":
            results.append(
                HorizonResult(
                    target_col=target_col,
                    feature_cols=[f"{target_col}_lag1"],
                    model_kind="shift1",
                    model=None,
                    predictions_raw=run_shift1_baseline(target_col, splits),
                )
            )
            continue

        feature_specs = resolve_feature_specs(
            experiment_name=plan.name,
            target_col=target_col,
            feature_level=plan.feature_level or "A",
            available_cols=set(splits.train.columns),
        )
        feature_cols, frames = _build_feature_frames(
            feature_specs=feature_specs,
            splits=splits,
            target_col=target_col,
            use_log_target=config.preprocess.log_target,
            target_epsilon=config.preprocess.target_epsilon,
        )
        model = _fit_model(
            config=config,
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
                    "rms5__mean": _rms_ref_series(frame),
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


def required_columns_for_experiment(name: str, targets: list[str]) -> set[str]:
    _ = resolve_experiment_plan(name)
    return set(targets)


def resolve_feature_specs(
    *,
    experiment_name: str,
    target_col: str,
    feature_level: str,
    available_cols: set[str],
) -> list[FeatureSpec]:
    ccy = _infer_currency(target_col)
    explicit_templates = {
        "local_lat_hist_abs_gbr": LAT_HIST_ABS_TEMPLATE,
        "local_lat_hist_signed_gbr": LAT_HIST_SIGNED_TEMPLATE,
        "local_lat_hist_plus_global_gbr": LAT_HIST_PLUS_GLOBAL_TEMPLATE,
        "local_lat_rms_nuanced_gbr": LAT_RMS_NUANCED_TEMPLATE,
    }
    if experiment_name in explicit_templates:
        return _resolve_explicit_feature_specs(
            target_col=target_col,
            templates=explicit_templates[experiment_name],
            available_cols=available_cols,
        )

    specs: list[FeatureSpec] = []
    seen: set[str] = set()

    def add_raw(col: str) -> None:
        if col in available_cols and col not in seen:
            specs.append(FeatureSpec(out_col=col, source_col=col, abs_transform=False))
            seen.add(col)

    def add_first(candidates: list[str]) -> str | None:
        for col in candidates:
            if col in available_cols:
                add_raw(col)
                return col
        return None

    def add_abs_from(source_col: str, out_col: str | None = None) -> None:
        if source_col not in available_cols:
            return
        resolved_out = out_col or f"{source_col}_abs"
        if resolved_out in seen:
            return
        specs.append(FeatureSpec(out_col=resolved_out, source_col=source_col, abs_transform=True))
        seen.add(resolved_out)

    # A: local state + global context.
    if ccy:
        for col in (f"rms5__{ccy}", f"rms5_{ccy}", f"rms5__ccy_{ccy}"):
            add_raw(col)
    for col in GLOBAL_RMS_CANDIDATES:
        add_raw(col)

    if feature_level in {"B", "C"}:
        if ccy:
            add_raw(f"abs_lat_ret__{ccy}")
            add_raw(f"abs_lat_ret_{ccy}")
            for col in (f"lat_ret__{ccy}", f"lat_ret_{ccy}"):
                add_abs_from(col)
        for col in IMPULSE_AGG_CANDIDATES:
            add_raw(col)

        # PCA ladder: B uses top-2 components, C uses full set.
        max_components = 2 if feature_level == "B" else 6
        for idx in range(1, max_components + 1):
            raw = add_first([f"pca_vol_PC{idx}", f"PC{idx}"])
            abs_existing = add_first([f"pca_vol_abs_PC{idx}", f"PC{idx}_abs"])
            if raw is not None and abs_existing is None:
                add_abs_from(raw, out_col=f"{raw}_abs")

    if feature_level == "C":
        for col in sorted(available_cols):
            if re.fullmatch(r"lat_ret__([A-Z]{3})", col) or re.fullmatch(
                r"lat_ret_([A-Z]{3})", col
            ):
                add_raw(col)
                add_abs_from(col)

    if not specs:
        preview = ", ".join(sorted(list(available_cols))[:20])
        raise ValueError(
            f"No usable features resolved for target '{target_col}' at level '{feature_level}'. "
            f"Available columns (first 20): {preview}"
        )

    return specs


def run_zero_baseline(target_col: str, splits: DataFrameSplits) -> dict[str, pd.DataFrame]:
    pred_frames: dict[str, pd.DataFrame] = {}
    for split_name, frame in {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }.items():
        pred = pd.DataFrame(
            {
                "date": pd.to_datetime(frame["__date__"], utc=True),
                "y_true_raw": frame[target_col].to_numpy(dtype="float64"),
                "y_pred_raw": np.zeros(len(frame), dtype="float64"),
                "rms5__mean": _rms_ref_series(frame),
            },
            index=frame.index,
        )
        pred_frames[split_name] = pred
    return pred_frames


def run_shift1_baseline(target_col: str, splits: DataFrameSplits) -> dict[str, pd.DataFrame]:
    all_df = pd.concat([splits.train, splits.val, splits.test], axis=0).copy()
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
                "rms5__mean": _rms_ref_series(joined),
            },
            index=joined.index,
        )
        pred = pred.dropna(axis=0, how="any")
        pred_frames[split_name] = pred
    return pred_frames


def _build_feature_frames(
    *,
    feature_specs: list[FeatureSpec],
    splits: DataFrameSplits,
    target_col: str,
    use_log_target: bool,
    target_epsilon: float,
) -> tuple[list[str], dict[str, pd.DataFrame]]:
    feature_cols = [spec.out_col for spec in feature_specs]
    frames: dict[str, pd.DataFrame] = {}

    for split_name, source in {
        "train": splits.train,
        "val": splits.val,
        "test": splits.test,
    }.items():
        frame = source.copy()
        for spec in feature_specs:
            if spec.abs_transform:
                frame[spec.out_col] = frame[spec.source_col].abs()
            elif spec.out_col != spec.source_col:
                frame[spec.out_col] = frame[spec.source_col]

        frame["__target_raw__"] = frame[target_col].astype("float64")
        frame["__target_model__"] = _transform_target(
            frame["__target_raw__"].to_numpy(),
            use_log=use_log_target,
            eps=target_epsilon,
        )

        keep_cols = ["__date__", "__target_raw__", "__target_model__"] + feature_cols
        frame = frame.dropna(axis=0, subset=keep_cols)
        if frame.empty:
            raise ValueError(
                f"Split '{split_name}' has no rows after dropping NaNs for target '{target_col}'."
            )
        frames[split_name] = frame

    return feature_cols, frames


def _fit_model(
    config: FxLocalVolConfig,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int | None,
):
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=config.model.gbr_n_estimators,
        learning_rate=config.model.gbr_learning_rate,
        max_depth=config.model.gbr_max_depth,
        min_samples_leaf=config.model.gbr_min_samples_leaf,
        subsample=config.model.gbr_subsample,
        random_state=seed,
    ).fit(X_train, y_train)


def _transform_target(values: np.ndarray, *, use_log: bool, eps: float) -> np.ndarray:
    typed = values.astype("float64")
    if not use_log:
        return typed
    return np.log(np.clip(typed + eps, a_min=eps, a_max=None))


def _inverse_target(values: np.ndarray, *, use_log: bool, eps: float) -> np.ndarray:
    if not use_log:
        return values.astype("float64")
    return np.exp(values.astype("float64")) - eps


def _infer_currency(target_col: str) -> str | None:
    m = re.search(r"__([A-Z]{3})$", target_col)
    if m:
        return m.group(1)
    m = re.search(r"_([A-Z]{3})$", target_col)
    if m:
        return m.group(1)
    return None


def _rms_ref_series(frame: pd.DataFrame) -> np.ndarray:
    for col in ("rms5__mean", "rms5__global_mean", "rms5_mean"):
        if col in frame.columns:
            return frame[col].to_numpy(dtype="float64")
    return np.full(shape=len(frame), fill_value=np.nan, dtype="float64")


def _resolve_explicit_feature_specs(
    *,
    target_col: str,
    templates: list[str],
    available_cols: set[str],
) -> list[FeatureSpec]:
    ccy = _infer_currency(target_col)
    specs: list[FeatureSpec] = []
    missing: list[str] = []
    for template_col in templates:
        col = template_col
        if ccy:
            col = template_col.replace("__JPY", f"__{ccy}")
        if col not in available_cols:
            missing.append(col)
            continue
        specs.append(FeatureSpec(out_col=col, source_col=col, abs_transform=False))

    if missing:
        raise ValueError(
            f"Experiment is missing required feature columns for target '{target_col}': {missing}"
        )
    return specs
