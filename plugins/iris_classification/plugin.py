from __future__ import annotations

from datetime import UTC, datetime

from core.contracts import Plugin, PluginInfo, RunResult, RunSpec
from core.contracts.run_context import RunContext

from .artifacts import (
    write_confusion_matrix_plot,
    write_data_summary,
    write_metrics,
    write_model,
)
from .config import parse_config
from .data import load_iris_splits
from .eval import evaluate_model
from .train import build_model, fit_model


class IrisClassificationPlugin(Plugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            key="sklearn.iris_classification",
            name="Iris Classification",
            version="0.1.0",
            description="Train and evaluate a logistic regression classifier on sklearn iris.",
        )

    def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
        start = datetime.now(UTC)

        _ensure_dependencies()

        config = parse_config(spec.data_spec)
        seed = spec.seed

        splits = load_iris_splits(split=config.split, seed=seed)
        model = build_model(config=config.model, seed=seed)
        model = fit_model(model, splits.X_train, splits.y_train)

        metrics, cm = evaluate_model(model, splits)

        dataset_id = spec.dataset_id or "sklearn:iris"
        data_summary_path = write_data_summary(
            context.artifact_dir,
            dataset_id=dataset_id,
            feature_names=splits.feature_names,
            target_names=splits.target_names,
            splits=splits,
            seed=seed,
        )
        metrics_path = write_metrics(context.artifact_dir, metrics)
        model_path = write_model(context.artifact_dir, model)
        plot_path = write_confusion_matrix_plot(
            context.artifact_dir,
            confusion_matrix=cm,
            labels=splits.target_names,
        )

        context.tracking.log_params(
            {
                "data.source": "sklearn",
                "data.dataset": "iris",
                "split.train": config.split.train,
                "split.val": config.split.val,
                "split.test": config.split.test,
                "model.type": config.model.type,
                "model.C": config.model.C,
                "model.max_iter": config.model.max_iter,
            }
        )

        for name, value in metrics.items():
            context.tracking.log_metric(name, float(value))

        context.tracking.log_artifact(str(data_summary_path), artifact_path="data")
        context.tracking.log_artifact(str(metrics_path), artifact_path="metrics")
        context.tracking.log_artifact(str(model_path), artifact_path="models")
        context.tracking.log_artifact(str(plot_path), artifact_path="plots")

        end = datetime.now(UTC)
        primary_metric = "test_f1_macro"
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={
                "primary_metric": primary_metric,
                "primary_metric_value": float(metrics[primary_metric]),
                "model_artifact": "models/model.joblib",
            },
        )


def _ensure_dependencies() -> None:
    import importlib

    missing: list[str] = []
    for module in ("numpy", "sklearn", "joblib", "matplotlib"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)

    if missing:
        raise RuntimeError(
            "Missing dependencies for IrisClassificationPlugin: "
            f"{', '.join(missing)}. Install with `pip install -e \".[sklearn]\"`."
        )
