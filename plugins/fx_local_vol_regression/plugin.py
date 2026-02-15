from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from core.contracts import ConfigurablePlugin, Plugin, PluginInfo, RunResult, RunSpec
from core.contracts.run_contracts.run_context import RunContext

from .artifacts import write_bootstrap_summary
from .config import default_params, parse_config, validate_params
from .data import load_dataset


class FxLocalVolRegressionPlugin(Plugin, ConfigurablePlugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            key="fx.local_vol_regression",
            name="FX Local Vol Regression",
            version="0.1.0",
            description="Local-vol plugin scaffold for A/B/C feature-set experiments.",
        )

    def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
        start = datetime.now(UTC)
        _ensure_dependencies()

        config = parse_config(spec.data_spec, spec.params, strict=spec.strict)
        dataset = load_dataset(config.data)

        target_cols = (
            config.experiment.target_cols if config.experiment.target_cols else dataset.target_cols
        )
        payload = {
            "status": "scaffold_only",
            "note": "No modeling yet. Dataset and experiment contract validated.",
            "dataset_path": str(dataset.path),
            "index_col": dataset.index_col,
            "feature_cols": dataset.feature_cols,
            "target_cols_from_roles": dataset.target_cols,
            "target_cols_selected": target_cols,
            "experiment": config.experiment.model_dump(mode="python"),
            "rows": int(len(dataset.full_df)),
            "columns": list(dataset.full_df.columns),
        }
        summary_path = write_bootstrap_summary(context.artifact_dir, payload)

        context.tracking.set_tags(
            {
                "purpose": "bootstrap",
                "plugin": self.info.key,
                "pipeline": spec.pipeline,
            }
        )
        context.tracking.log_params(
            {
                "data.dataset_id": spec.dataset_id or "",
                "data.dataset_path": config.data.dataset_path,
                "data.split_name": config.data.split_name,
                "experiment.name": config.experiment.name,
                "experiment.feature_set": config.experiment.feature_set,
                "experiment.model_kind": config.experiment.model_kind,
                "split.train_end_date": config.split.train_end_date,
                "split.val_end_date": config.split.val_end_date,
                "split.test_end_date": config.split.test_end_date,
            }
        )
        context.tracking.log_metric("bootstrap_rows", float(len(dataset.full_df)))
        context.tracking.log_artifact(str(summary_path), artifact_path="reports")

        end = datetime.now(UTC)
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={
                "bootstrap_summary": "reports/bootstrap_summary.json",
                "target_cols_selected": target_cols,
            },
            message="Local-vol plugin scaffold run complete (no model training yet).",
        )

    def default_params(self) -> dict[str, Any]:
        return default_params()

    def validate_params(self, params: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
        return validate_params(params, strict=strict)


def _ensure_dependencies() -> None:
    import importlib.util

    missing: list[str] = []
    for module in ("mlh_data", "pandas", "pyarrow"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    if missing:
        raise RuntimeError(
            "Missing dependencies for FxLocalVolRegressionPlugin: "
            f"{', '.join(missing)}. "
            'Install with `pip install -e ".[mlh-data,data]"`.'
        )
