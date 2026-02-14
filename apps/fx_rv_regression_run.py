from __future__ import annotations

import os

from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from plugins.fx_rv_regression import FxRvRegressionPlugin
from tracking_clients import MlflowTrackingClient


def _resolve_tracking_uri() -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return tracking_uri or "http://localhost:5000"


def _resolve_experiment_name() -> str:
    fx_experiment = os.environ.get("FX_RV_MLFLOW_EXPERIMENT")
    if fx_experiment:
        return fx_experiment
    return os.environ.get("MLFLOW_EXPERIMENT", "fx-rv-regression")


def _resolve_dataset_path() -> str:
    dataset_path = os.environ.get("DATASET_PATH")
    if dataset_path:
        return dataset_path
    fallback_fx = os.environ.get("FX_RV_DATASET_PATH")
    if fallback_fx:
        return fallback_fx
    fallback = os.environ.get("HMM_DATASET_PATH")
    if fallback:
        return fallback
    raise RuntimeError("DATASET_PATH is not set.")


def _resolve_experiment_name_override() -> str:
    return os.environ.get("FX_RV_EXPERIMENT", "rv_base_persist_shift1")


def main() -> None:
    spec = RunSpec(
        plugin_key="fx.rv_regression",
        pipeline="train",
        dataset_id="fx:rv_dataset:v1",
        data_spec={
            "dataset_path": _resolve_dataset_path(),
            "split_name": "default",
        },
        params={
            "experiment": {
                "name": _resolve_experiment_name_override(),
                "target_cols": ["rv_fwd5__mean", "rv_fwd10__mean", "rv_fwd20__mean"],
            }
        },
        strict=True,
    )

    registry = DictPluginRegistry({"fx.rv_regression": FxRvRegressionPlugin()})
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )
    result = run_pipeline(spec, registry=registry, tracking=tracking)
    print(result)


if __name__ == "__main__":
    main()
