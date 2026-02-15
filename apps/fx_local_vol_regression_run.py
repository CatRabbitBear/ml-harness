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
    explicit = os.environ.get("FX_LOCAL_DATASET_PATH")
    if explicit:
        return explicit
    dataset_one = os.environ.get("DATASET_PATH")
    if dataset_one:
        return dataset_one
    dataset_two = os.environ.get("DATASET_PATH_TWO")
    if dataset_two:
        return dataset_two
    raise RuntimeError("DATASET_PATH_TWO or DATASET_PATH must be set.")


def _resolve_experiment_name_override() -> str:
    return os.environ.get("FX_LOCAL_EXPERIMENT", "local_lat_hist_abs_gbr")


def _resolve_target_cols() -> list[str]:
    raw = os.environ.get("FX_LOCAL_TARGET_COLS", "")
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _resolve_feature_set(experiment_name: str) -> str:
    if "base_" in experiment_name:
        return "A"
    if "_features_c_" in experiment_name:
        return "C"
    if "_features_b_" in experiment_name:
        return "B"
    return "A"


def main() -> None:
    experiment_name = _resolve_experiment_name_override()
    model_kind = "gbr"
    if experiment_name == "local_base_shift1":
        model_kind = "shift1"
    if experiment_name == "local_base_zero":
        model_kind = "zero"
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
                "name": experiment_name,
                "target_cols": _resolve_target_cols(),
                "feature_set": _resolve_feature_set(experiment_name),
                "model_kind": model_kind,
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
