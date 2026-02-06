from __future__ import annotations

from datetime import UTC, datetime

from core.contracts import Plugin, PluginInfo, RunResult, RunSpec
from core.contracts.run_context import RunContext

from .artifacts import write_dataset_summary
from .config import parse_config
from .data import load_dataset


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
        dataset = load_dataset(config.data)

        dataset_id = spec.dataset_id or "fx:latent_returns_daily"
        summary_path = write_dataset_summary(
            context.artifact_dir,
            dataset,
            dataset_id=dataset_id,
            split_name=config.data.split_name,
        )

        manifest = dataset.loaded.manifest
        context.tracking.log_params(
            {
                "data.dataset_id": dataset_id,
                "data.dataset_path": config.data.dataset_path,
                "data.split_name": config.data.split_name,
                "data.index_col": dataset.index_col,
                "data.feature_count": len(dataset.feature_cols),
                "data.target_count": len(dataset.target_cols),
                "data.row_count": int(len(dataset.loaded.df)),
                "data.dataset_name": manifest.dataset_name if manifest else None,
                "data.version": manifest.version if manifest else None,
            }
        )

        context.tracking.log_artifact(str(summary_path), artifact_path="data")

        end = datetime.now(UTC)
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={
                "dataset_summary": "data/dataset_summary.json",
                "split_name": config.data.split_name,
                "dataset_path": config.data.dataset_path,
            },
            message="Dataset loaded and summarized. HMM training not implemented yet.",
        )


def _ensure_dependencies() -> None:
    import importlib.util

    missing: list[str] = []
    for module in ("mlh_data", "pandas", "pyarrow"):
        if importlib.util.find_spec(module) is None:
            missing.append(module)

    if missing:
        raise RuntimeError(
            "Missing dependencies for HmmFxDailyPlugin: "
            f"{', '.join(missing)}. Install with `pip install -e \".[mlh-data,data]\"`."
        )
