from __future__ import annotations

import os

from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from plugins.fx_local_vol_regression import FxLocalVolRegressionPlugin
from tracking_clients import MlflowTrackingClient


def _resolve_tracking_uri() -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return tracking_uri or "http://localhost:5000"


def _resolve_experiment_name() -> str:
    local_experiment = os.environ.get("FX_LOCAL_MLFLOW_EXPERIMENT")
    if local_experiment:
        return local_experiment
    return os.environ.get("MLFLOW_EXPERIMENT", "fx-local-vol")


def _resolve_dataset_path() -> str:
    dataset_two = os.environ.get("DATASET_PATH_TWO")
    if dataset_two:
        return dataset_two
    dataset_one = os.environ.get("DATASET_PATH")
    if dataset_one:
        return dataset_one
    raise RuntimeError("DATASET_PATH_TWO or DATASET_PATH must be set.")


def _resolve_experiment_name_override() -> str:
    return os.environ.get("FX_LOCAL_EXPERIMENT", "local_base_shift1")


def main() -> None:
    spec = RunSpec(
        plugin_key="fx.local_vol_regression",
        pipeline="train",
        dataset_id="fx:local_vol_dataset:v1",
        data_spec={
            "dataset_path": _resolve_dataset_path(),
            "split_name": "default",
        },
        params={
            "experiment": {
                "name": _resolve_experiment_name_override(),
                "target_cols": [],
                "feature_set": "A",
                "model_kind": "gbr",
            }
        },
        strict=True,
    )

    registry = DictPluginRegistry({"fx.local_vol_regression": FxLocalVolRegressionPlugin()})
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )
    result = run_pipeline(spec, registry=registry, tracking=tracking)
    print(result)


if __name__ == "__main__":
    main()
