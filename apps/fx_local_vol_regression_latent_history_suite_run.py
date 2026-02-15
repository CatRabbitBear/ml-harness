from __future__ import annotations

import os

from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from plugins.fx_local_vol_regression import FxLocalVolRegressionPlugin
from tracking_clients import MlflowTrackingClient

LADDER_EXPERIMENTS = [
    "local_lat_hist_abs_gbr",
    "local_lat_hist_signed_gbr",
    "local_lat_hist_plus_global_gbr",
    "local_lat_rms_nuanced_gbr",
]


def _resolve_tracking_uri() -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return tracking_uri or "http://localhost:5000"


def _resolve_experiment_name() -> str:
    local_experiment = os.environ.get("FX_LOCAL_MLFLOW_EXPERIMENT")
    if local_experiment:
        return local_experiment
    return os.environ.get("MLFLOW_EXPERIMENT", "fx-local-vol")


def _resolve_dataset_path() -> str:
    explicit = os.environ.get("FX_LOCAL_DATASET_PATH")
    if explicit:
        return explicit
    dataset_path = os.environ.get("DATASET_PATH")
    if dataset_path:
        return dataset_path
    raise RuntimeError("FX_LOCAL_DATASET_PATH or DATASET_PATH must be set.")


def _resolve_target_cols() -> list[str]:
    raw = os.environ.get("FX_LOCAL_TARGET_COLS", "")
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def main() -> None:
    registry = DictPluginRegistry({"fx.local_vol_regression": FxLocalVolRegressionPlugin()})
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )

    target_cols = _resolve_target_cols()
    dataset_path = _resolve_dataset_path()

    for experiment_name in LADDER_EXPERIMENTS:
        spec = RunSpec(
            plugin_key="fx.local_vol_regression",
            pipeline="train",
            dataset_id="fx:local_vol_dataset:latent_jpy:v1",
            data_spec={
                "dataset_path": dataset_path,
                "split_name": "default",
            },
            params={
                "experiment": {
                    "name": experiment_name,
                    "target_cols": target_cols,
                    "feature_set": "A",
                    "model_kind": "gbr",
                }
            },
            strict=True,
            tags={"suite": "fx_local_latent_history_ladder", "family": "latent"},
        )
        result = run_pipeline(spec, registry=registry, tracking=tracking)
        print(experiment_name, result.status, result.run_id)


if __name__ == "__main__":
    main()
