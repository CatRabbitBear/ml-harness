from __future__ import annotations

import os

from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from plugins.hmm_fx_daily import HmmFxDailyPlugin
from tracking_clients import MlflowTrackingClient


def _resolve_tracking_uri() -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return tracking_uri or "http://localhost:5000"


def _resolve_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "ml-harness-dev")


def _resolve_dataset_path() -> str:
    dataset_path = os.environ.get("HMM_DATASET_PATH")
    if not dataset_path:
        raise RuntimeError("HMM_DATASET_PATH is not set.")
    return dataset_path


def main() -> None:
    registry = DictPluginRegistry(plugins={"hmm.fx_daily": HmmFxDailyPlugin()})
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )

    spec = RunSpec(
        plugin_key="hmm.fx_daily",
        dataset_id="fx:latent_returns_daily:v7",
        data_spec={
            "data": {
                "dataset_path": _resolve_dataset_path(),
                "split_name": "default",
            },
            "model": {
                "n_components": 2,
                "covariance_type": "full",
                "transmat_prior_strength": 50.0,
                "transmat_prior_mode": "sticky_diag",
            },
            "train": {
                "n_init": 5,
                "n_iter": 300,
                "tol": 1e-3,
                "init_strategy": "kmeans",
            },
            "preprocess": {
                "scaler": "robust",
                "winsorize_vol": False,
            },
            "eval": {
                "eval_scheme": "last_n_days",
                "eval_last_n_days": 252,
            },
        },
        tags={"purpose": "realdata"},
    )

    result = run_pipeline(spec, registry=registry, tracking=tracking)
    print(result)


if __name__ == "__main__":
    main()
