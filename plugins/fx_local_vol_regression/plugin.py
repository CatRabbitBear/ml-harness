from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from core.contracts import ConfigurablePlugin, Plugin, PluginInfo, RunResult, RunSpec
from core.contracts.run_contracts.run_context import RunContext

from .artifacts import (
    write_baseline_spec,
    write_data_summary,
    write_metrics,
    write_model,
    write_predictions,
)
from .config import FxLocalVolConfig, default_params, parse_config, validate_params
from .data import load_dataset, split_by_date
from .eval import compute_test_deltas, evaluate_predictions
from .plots import write_diagnostic_plots
from .train import (
    normalize_experiment_name,
    required_columns_for_experiment,
    run_experiment,
    run_zero_baseline,
)


class FxLocalVolRegressionPlugin(Plugin, ConfigurablePlugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            key="fx.local_vol_regression",
            name="FX Local Vol Regression",
            version="0.2.0",
            description="Local-vol regression baselines on FX geometry features.",
        )

    def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
        start = datetime.now(UTC)
        _ensure_dependencies()

        config = parse_config(spec.data_spec, spec.params, strict=spec.strict)
        canonical_experiment_name = normalize_experiment_name(config.experiment.name)

        dataset = load_dataset(config.data)
        target_cols = (
            config.experiment.target_cols if config.experiment.target_cols else dataset.target_cols
        )
        if not target_cols:
            raise ValueError(
                "No target columns resolved. Set params.experiment.target_cols or dataset roles.target_cols."
            )

        _validate_dataset_columns(dataset.full_df.columns, config, target_cols)
        splits = split_by_date(dataset.full_df, index_col=dataset.index_col, split_cfg=config.split)

        run_cfg = config.model_copy(
            update={"experiment": config.experiment.model_copy(update={"target_cols": target_cols})}
        )
        horizon_results = run_experiment(run_cfg, splits, seed=spec.seed)

        all_metrics: dict[str, float] = {}
        model_artifacts: dict[str, str] = {}

        for result in horizon_results:
            metrics = evaluate_predictions(
                predictions_by_split=result.predictions_raw,
                top_quantile=config.eval.top_quantile,
                target_epsilon=config.preprocess.target_epsilon,
                use_log_metrics=config.preprocess.log_target,
            )
            zero_preds = run_zero_baseline(result.target_col, splits)
            zero_metrics = evaluate_predictions(
                predictions_by_split=zero_preds,
                top_quantile=config.eval.top_quantile,
                target_epsilon=config.preprocess.target_epsilon,
                use_log_metrics=config.preprocess.log_target,
            )
            if result.model_kind != "zero":
                metrics.update(
                    compute_test_deltas(
                        model_metrics=metrics,
                        baseline_metrics=zero_metrics,
                    )
                )
            for key, value in metrics.items():
                all_metrics[f"{result.target_col}_{key}"] = value

            for split_name, pred_frame in result.predictions_raw.items():
                write_predictions(
                    context.artifact_dir,
                    target_col=result.target_col,
                    split_name=split_name,
                    frame=pred_frame,
                )

            if config.plots.enabled and "test" in result.predictions_raw:
                plot_paths = write_diagnostic_plots(
                    context.artifact_dir,
                    experiment_name=canonical_experiment_name,
                    target_col=result.target_col,
                    test_frame=result.predictions_raw["test"],
                    top_quantile=config.eval.top_quantile,
                    target_epsilon=config.preprocess.target_epsilon,
                    overlay_years=config.plots.overlay_years,
                    use_log_space=config.preprocess.log_target,
                )
                for plot_path in plot_paths:
                    context.tracking.log_artifact(str(plot_path), artifact_path="plots")

            if result.model_kind in {"zero", "shift1"}:
                strategy = "const_zero" if result.model_kind == "zero" else "lag1_target"
                model_path = write_baseline_spec(
                    context.artifact_dir,
                    target_col=result.target_col,
                    strategy=strategy,
                )
            else:
                model_path = write_model(
                    context.artifact_dir,
                    target_col=result.target_col,
                    model=result.model,
                )
            model_artifacts[result.target_col] = str(model_path)

        metrics_path = write_metrics(context.artifact_dir, all_metrics)
        summary_path = write_data_summary(
            context.artifact_dir,
            dataset_id=spec.dataset_id or "fx:local_vol_dataset:v1",
            dataset_path=config.data.dataset_path,
            index_col=dataset.index_col,
            row_counts={
                "train": int(len(splits.train)),
                "val": int(len(splits.val)),
                "test": int(len(splits.test)),
            },
            split_dates={
                "train_end_date": config.split.train_end_date,
                "val_end_date": config.split.val_end_date,
                "test_end_date": config.split.test_end_date,
            },
            experiment_name=canonical_experiment_name,
            target_cols=target_cols,
        )

        context.tracking.set_tags(
            {
                "purpose": "realdata",
                "plugin": self.info.key,
                "pipeline": spec.pipeline,
                "experiment_name": canonical_experiment_name,
                "target_type": "local_vol",
            }
        )
        context.tracking.log_params(
            {
                "data.dataset_id": spec.dataset_id or "",
                "data.dataset_path": config.data.dataset_path,
                "data.split_name": config.data.split_name,
                "experiment.name": canonical_experiment_name,
                "experiment.feature_set": config.experiment.feature_set,
                "experiment.model_kind": config.experiment.model_kind,
                "experiment.target_cols": ",".join(target_cols),
                "split.train_end_date": config.split.train_end_date,
                "split.val_end_date": config.split.val_end_date,
                "split.test_end_date": config.split.test_end_date,
                "model.gbr_n_estimators": config.model.gbr_n_estimators,
                "model.gbr_learning_rate": config.model.gbr_learning_rate,
                "model.gbr_max_depth": config.model.gbr_max_depth,
                "model.gbr_min_samples_leaf": config.model.gbr_min_samples_leaf,
                "model.gbr_subsample": config.model.gbr_subsample,
                "preprocess.log_target": config.preprocess.log_target,
                "preprocess.target_epsilon": config.preprocess.target_epsilon,
                "eval.top_quantile": config.eval.top_quantile,
                "plots.enabled": config.plots.enabled,
                "plots.overlay_years": ",".join(str(y) for y in config.plots.overlay_years),
            }
        )
        context.tracking.log_metrics(all_metrics)
        context.tracking.log_artifact(str(metrics_path), artifact_path="metrics")
        context.tracking.log_artifact(str(summary_path), artifact_path="data")

        end = datetime.now(UTC)
        primary_metric_key = f"{target_cols[0]}_test_rmse"
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={
                "primary_metric": primary_metric_key,
                "primary_metric_value": float(all_metrics.get(primary_metric_key, 0.0)),
                "metrics_artifact": "metrics/metrics.json",
                "model_artifacts": model_artifacts,
            },
        )

    def default_params(self) -> dict[str, Any]:
        return default_params()

    def validate_params(self, params: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
        return validate_params(params, strict=strict)


def _ensure_dependencies() -> None:
    import importlib.util

    missing: list[str] = []
    for module in (
        "mlh_data",
        "pandas",
        "numpy",
        "pyarrow",
        "sklearn",
        "joblib",
        "matplotlib",
    ):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing dependencies for FxLocalVolRegressionPlugin: "
            f"{', '.join(missing)}. "
            'Install with `pip install -e ".[mlh-data,data,sklearn]"`.'
        )


def _validate_dataset_columns(
    columns: Sequence[str],
    config: FxLocalVolConfig,
    target_cols: list[str],
) -> None:
    column_set = set(columns)
    required = required_columns_for_experiment(config.experiment.name, target_cols)

    missing = [c for c in sorted(required) if c not in column_set]
    if missing:
        available = sorted(column_set)
        preview = ", ".join(available[:20])
        raise ValueError(
            "Dataset is missing required columns for experiment "
            f"{config.experiment.name}: {missing}. "
            f"Available columns (first 20): {preview}"
        )
