from __future__ import annotations

import os

from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from plugins.fx_rv_regression import FxRvRegressionPlugin
from tracking_clients import MlflowTrackingClient

LADDER_EXPERIMENTS = [
    "rv_base_persist_shift1",
    "rv_rms5_stats_ridge",
    "rv_rms5_stats_gbr",
    "rv_rms5_vec8_ridge",
    "rv_rms5_vec8_gbr",
    "rv_pca6_ridge",
    "rv_pca6_gbr",
    "rv_pca6_abs12_ridge",
    "rv_pca6_abs12_gbr",
    "rv_combo15_ridge",
    "rv_combo15_gbr",
    "rv_stress1_ridge",
    "rv_stress1_gbr",
]


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
    raise RuntimeError("DATASET_PATH is not set.")


def main() -> None:
    dataset_path = _resolve_dataset_path()
    registry = DictPluginRegistry({"fx.rv_regression": FxRvRegressionPlugin()})
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )
    for experiment_name in LADDER_EXPERIMENTS:
        spec = RunSpec(
            plugin_key="fx.rv_regression",
            pipeline="train",
            dataset_id="fx:rv_dataset:v1",
            data_spec={
                "dataset_path": dataset_path,
                "split_name": "default",
            },
            params={
                "experiment": {
                    "name": experiment_name,
                    "target_cols": ["rv_fwd5__mean", "rv_fwd10__mean", "rv_fwd20__mean"],
                }
            },
            strict=True,
            tags={"suite": "fx_rv_baseline_ladder"},
        )
        result = run_pipeline(spec, registry=registry, tracking=tracking)
        print(experiment_name, result.status, result.run_id)


if __name__ == "__main__":
    main()
