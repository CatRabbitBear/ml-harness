from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from core.contracts import Plugin, PluginInfo, RunResult, RunSpec
from core.contracts.run_context import RunContext

from .artifacts import (
    write_dataset_summary,
    write_json_artifact,
    write_model_artifacts,
    write_state_summary,
    write_state_timeseries,
    write_transmat,
)
from .config import parse_config
from .data import get_feature_frame, get_time_index, load_dataset
from .eval import (
    build_state_summary,
    compute_duration_metrics,
    compute_state_metrics,
    compute_state_series,
    compute_switch_metrics,
    score_log_likelihood,
)
from .plots import (
    plot_feature_means_by_state,
    plot_regime_overlay,
    plot_state_duration_hist,
    plot_state_occupancy_rolling,
    plot_transmat_heatmap,
)
from .train import build_transmat_prior, fit_best_hmm


class HmmFxDailyPlugin(Plugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            key="hmm.fx_daily",
            name="HMM FX Daily",
            version="0.1.0",
            description="Load latent_returns_daily dataset artifacts and prepare splits for HMM modeling.",
        )

    def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
        start = datetime.now(UTC)

        _ensure_dependencies()

        config = parse_config(spec.data_spec)
        train_seed = spec.seed if spec.seed is not None else config.train.random_seed
        train_cfg = replace(config.train, random_seed=train_seed)

        dataset = load_dataset(config.data)
        feature_df = get_feature_frame(dataset)
        dates = get_time_index(dataset)

        feature_cols = list(feature_df.columns)
        dataset_id = spec.dataset_id or "fx:latent_returns_daily"
        summary_path = write_dataset_summary(
            context.artifact_dir,
            dataset,
            dataset_id=dataset_id,
            split_name=config.data.split_name,
        )

        warnings: list[str] = []
        X_full = _preprocess_features(
            feature_df,
            winsorize_vol=config.preprocess.winsorize_vol,
        )
        train_slice, eval_slice = _compute_train_eval_slices(
            len(X_full),
            eval_scheme=config.eval.eval_scheme,
            eval_last_n_days=config.eval.eval_last_n_days,
            warnings=warnings,
        )

        scaler, X_train, X_full_scaled = _scale_features(
            X_full,
            train_slice=train_slice,
            scaler_mode=config.preprocess.scaler,
        )

        transmat_prior = build_transmat_prior(
            n_components=config.model.n_components,
            mode=config.model.transmat_prior_mode,
            strength=config.model.transmat_prior_strength,
            custom_matrix=config.model.transmat_prior_custom,
        )

        fit_result = fit_best_hmm(
            X_train,
            model_cfg=config.model,
            train_cfg=train_cfg,
            transmat_prior=transmat_prior,
        )

        model = fit_result.model
        states, max_posterior, entropy = compute_state_series(model, X_full_scaled)
        switch_metrics = compute_switch_metrics(states)
        overall_duration_metrics, durations_per_state = compute_duration_metrics(
            states, config.model.n_components
        )
        state_metrics = compute_state_metrics(states, durations_per_state)

        state_summary = build_state_summary(
            feature_df,
            states,
            config.model.n_components,
            durations_per_state,
        )

        val_metrics: dict[str, float] = {}
        if eval_slice is not None:
            X_val = X_full_scaled[eval_slice]
            if X_val.shape[0] > 0:
                val_ll = _score_eval(
                    model,
                    X_val,
                    eval_scheme=config.eval.eval_scheme,
                )
                val_metrics = {
                    "val_log_likelihood": val_ll,
                    "val_avg_log_likelihood_per_step": val_ll / float(X_val.shape[0]),
                }

        train_ll = fit_result.train_log_likelihood
        train_metrics = {
            "train_log_likelihood": train_ll,
            "train_avg_log_likelihood_per_step": train_ll / float(X_train.shape[0]),
            "train_converged": 1.0 if fit_result.converged else 0.0,
            "train_iterations": float(fit_result.iterations),
        }

        posterior_metrics = {
            "mean_max_posterior": float(max_posterior.mean()) if max_posterior.size else 0.0,
            "mean_state_entropy": float(entropy.mean()) if entropy.size else 0.0,
        }

        transmat = model.transmat_
        transmat_df = _matrix_df(transmat, config.model.n_components)
        prior_df = _matrix_df(_normalize_rows(transmat_prior), config.model.n_components)

        state_timeseries = _build_state_timeseries(
            dates,
            states,
            max_posterior,
            entropy,
            feature_df,
            feature_cols,
        )

        summary_csv, summary_parquet = write_state_summary(context.artifact_dir, state_summary)
        transmat_csv, transmat_parquet = write_transmat(context.artifact_dir, transmat_df)
        prior_csv, prior_parquet = write_transmat(
            context.artifact_dir, prior_df, name="transmat_prior"
        )
        timeseries_csv, timeseries_parquet = write_state_timeseries(
            context.artifact_dir, state_timeseries
        )
        prior_json = write_json_artifact(
            context.artifact_dir, "transmat_prior.json", prior_df.to_dict()
        )

        model_path, scaler_path = write_model_artifacts(
            context.artifact_dir, model=model, scaler=scaler
        )

        plots_dir = Path(context.artifact_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        proxy_series = _choose_proxy_series(feature_df, feature_cols)
        regime_plot = plot_regime_overlay(
            plots_dir / "regime_overlay.png",
            dates=dates,
            proxy=proxy_series,
            states=states,
            max_posterior=max_posterior,
        )
        duration_plot = plot_state_duration_hist(
            plots_dir / "state_duration_hist.png",
            durations_per_state=durations_per_state,
        )
        occupancy_plot = plot_state_occupancy_rolling(
            plots_dir / "state_occupancy_rolling.png",
            dates=dates,
            states=states,
            n_components=config.model.n_components,
        )
        feature_means_plot = plot_feature_means_by_state(
            plots_dir / "feature_means_by_state.png",
            state_summary=state_summary,
            feature_cols=feature_cols,
        )
        transmat_plot = plot_transmat_heatmap(
            plots_dir / "transmat_heatmap.png",
            transmat=transmat,
        )

        manifest = dataset.loaded.manifest
        date_min = dates.min().isoformat()
        date_max = dates.max().isoformat()
        tags = {
            "model_family": "HMM",
            "model_impl": "hmmlearn.GaussianHMM",
            "dataset_kind": "fx_daily_latent",
            "regime_model": "true",
        }
        if spec.notes:
            tags["notes"] = spec.notes
        context.tracking.set_tags(tags)
        context.tracking.log_params(
            {
                "data.dataset_id": dataset_id,
                "data.dataset_path": config.data.dataset_path,
                "data.split_name": config.data.split_name,
                "data.index_col": dataset.index_col,
                "data.feature_count": len(feature_cols),
                "data.target_count": len(dataset.target_cols),
                "data.row_count": int(len(feature_df)),
                "data.dataset_name": manifest.dataset_name if manifest else None,
                "data.version": manifest.version if manifest else None,
                "data.date_min": date_min,
                "data.date_max": date_max,
                "data.features": ",".join(feature_cols),
                "model.n_components": config.model.n_components,
                "model.covariance_type": config.model.covariance_type,
                "model.transmat_prior_strength": config.model.transmat_prior_strength,
                "model.transmat_prior_mode": config.model.transmat_prior_mode,
                "train.n_init": train_cfg.n_init,
                "train.n_iter": train_cfg.n_iter,
                "train.tol": train_cfg.tol,
                "train.init_strategy": train_cfg.init_strategy,
                "train.random_seed": train_seed,
                "preprocess.scaler": config.preprocess.scaler,
                "preprocess.winsorize_vol": config.preprocess.winsorize_vol,
                "eval.eval_scheme": config.eval.eval_scheme,
                "eval.eval_last_n_days": config.eval.eval_last_n_days,
            }
        )
        warnings_path = None
        if warnings:
            warnings_path = write_json_artifact(context.artifact_dir, "warnings.json", warnings)

        metrics = {}
        metrics.update(train_metrics)
        metrics.update(val_metrics)
        metrics.update(switch_metrics)
        metrics.update(overall_duration_metrics)
        metrics.update(state_metrics)
        metrics.update(posterior_metrics)
        context.tracking.log_metrics(_sanitize_metrics(metrics))

        context.tracking.log_artifact(str(summary_path), artifact_path="data")
        context.tracking.log_artifact(str(summary_csv), artifact_path="data")
        context.tracking.log_artifact(str(summary_parquet), artifact_path="data")
        context.tracking.log_artifact(str(transmat_csv), artifact_path="data")
        context.tracking.log_artifact(str(transmat_parquet), artifact_path="data")
        context.tracking.log_artifact(str(prior_csv), artifact_path="data")
        context.tracking.log_artifact(str(prior_parquet), artifact_path="data")
        context.tracking.log_artifact(str(prior_json), artifact_path="data")
        context.tracking.log_artifact(str(timeseries_csv), artifact_path="data")
        context.tracking.log_artifact(str(timeseries_parquet), artifact_path="data")
        if warnings_path is not None:
            context.tracking.log_artifact(str(warnings_path), artifact_path="data")
        context.tracking.log_artifact(str(model_path), artifact_path="models")
        if scaler_path is not None:
            context.tracking.log_artifact(str(scaler_path), artifact_path="models")

        context.tracking.log_artifact(str(regime_plot), artifact_path="plots")
        context.tracking.log_artifact(str(duration_plot), artifact_path="plots")
        context.tracking.log_artifact(str(occupancy_plot), artifact_path="plots")
        context.tracking.log_artifact(str(feature_means_plot), artifact_path="plots")
        context.tracking.log_artifact(str(transmat_plot), artifact_path="plots")

        end = datetime.now(UTC)
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={
                "dataset_summary": "data/dataset_summary.json",
                "state_timeseries": "data/state_timeseries.parquet",
                "transmat": "data/transmat.csv",
                "transmat_prior": "data/transmat_prior.csv",
                "model_artifact": "models/hmm_model.joblib",
            },
            message="HMM trained and artifacts logged.",
        )


def _ensure_dependencies() -> None:
    import importlib.util

    missing: list[str] = []
    for module in (
        "mlh_data",
        "pandas",
        "numpy",
        "pyarrow",
        "hmmlearn",
        "sklearn",
        "matplotlib",
        "joblib",
    ):
        if importlib.util.find_spec(module) is None:
            missing.append(module)

    if missing:
        raise RuntimeError(
            "Missing dependencies for HmmFxDailyPlugin: "
            f"{', '.join(missing)}. "
            'Install with `pip install -e ".[mlh-data,data,hmm,sklearn,mlflow]"`.'
        )


def _compute_train_eval_slices(
    n_rows: int,
    *,
    eval_scheme: str,
    eval_last_n_days: int,
    warnings: list[str],
) -> tuple[slice, slice | None]:
    if eval_scheme == "none":
        return slice(0, n_rows), None

    if eval_last_n_days >= n_rows:
        warnings.append("eval_last_n_days >= dataset length; using full data for training.")
        return slice(0, n_rows), None

    train_end = n_rows - eval_last_n_days
    return slice(0, train_end), slice(train_end, n_rows)


def _preprocess_features(feature_df: pd.DataFrame, *, winsorize_vol: bool) -> pd.DataFrame:
    df = feature_df.copy()
    if winsorize_vol:
        vol_cols = [c for c in df.columns if c in {"g_rng_mean", "g_absret_rms"}]
        for col in vol_cols:
            lower = df[col].quantile(0.01)
            upper = df[col].quantile(0.99)
            df[col] = df[col].clip(lower, upper)
    return df


def _scale_features(
    feature_df: pd.DataFrame,
    *,
    train_slice: slice,
    scaler_mode: str,
) -> tuple[Any | None, np.ndarray, np.ndarray]:
    scaler = None
    train_df = feature_df.iloc[train_slice]
    if scaler_mode == "none":
        return None, train_df.to_numpy(), feature_df.to_numpy()

    if scaler_mode == "standard":
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
    elif scaler_mode == "robust":
        from sklearn.preprocessing import RobustScaler

        scaler = RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler mode: {scaler_mode}")

    X_train = scaler.fit_transform(train_df.to_numpy())
    X_full = scaler.transform(feature_df.to_numpy())
    return scaler, X_train, X_full


def _score_eval(model: Any, X_val: np.ndarray, *, eval_scheme: str) -> float:
    if eval_scheme == "walkforward":
        return _score_walkforward(model, X_val)
    return score_log_likelihood(model, X_val)


def _score_walkforward(model: Any, X_val: np.ndarray) -> float:
    n = X_val.shape[0]
    if n < 10:
        return score_log_likelihood(model, X_val)
    folds = 4
    size = max(1, n // folds)
    scores = []
    for i in range(folds):
        start = i * size
        end = n if i == folds - 1 else (i + 1) * size
        chunk = X_val[start:end]
        if chunk.shape[0] == 0:
            continue
        scores.append(score_log_likelihood(model, chunk))
    if not scores:
        return score_log_likelihood(model, X_val)
    return float(np.mean(scores))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def _matrix_df(matrix: np.ndarray, n_components: int) -> pd.DataFrame:
    idx = [f"s{i}" for i in range(n_components)]
    return pd.DataFrame(matrix, index=idx, columns=idx)


def _build_state_timeseries(
    dates: pd.Series,
    states: np.ndarray,
    max_posterior: np.ndarray,
    entropy: np.ndarray,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    if isinstance(dates, pd.DatetimeIndex):
        date_values = dates.strftime("%Y-%m-%d")
    else:
        date_values = pd.to_datetime(dates, utc=True).dt.strftime("%Y-%m-%d")

    payload = {
        "date": date_values,
        "state": states,
        "max_posterior": max_posterior,
        "entropy": entropy,
    }
    proxy_cols = _select_proxy_columns(feature_cols)
    for col in proxy_cols:
        payload[col] = feature_df[col].values
    return pd.DataFrame(payload)


def _select_proxy_columns(feature_cols: list[str]) -> list[str]:
    preferred = ["g_absret_rms", "g_rng_mean"]
    proxies = [c for c in preferred if c in feature_cols]
    if len(proxies) >= 2:
        return proxies[:2]
    for col in feature_cols:
        if col not in proxies:
            proxies.append(col)
        if len(proxies) >= 2:
            break
    return proxies


def _choose_proxy_series(feature_df: pd.DataFrame, feature_cols: list[str]) -> pd.Series:
    proxy_cols = _select_proxy_columns(feature_cols)
    proxy = feature_df[proxy_cols[0]].copy()
    return proxy


def _sanitize_metrics(metrics: dict[str, float]) -> dict[str, float]:
    import math

    cleaned: dict[str, float] = {}
    for key, value in metrics.items():
        if value is None:
            continue
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            cleaned[key] = 0.0
        else:
            cleaned[key] = float(value)
    return cleaned
