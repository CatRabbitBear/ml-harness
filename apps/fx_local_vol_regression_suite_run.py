from __future__ import annotations

import os

from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from plugins.fx_local_vol_regression import FxLocalVolRegressionPlugin
from tracking_clients import MlflowTrackingClient

LADDER_EXPERIMENTS = [
    "local_base_shift1",
    "local_features_a_gbr",
    "local_features_b_gbr",
    "local_features_c_gbr",
]


def _resolve_tracking_uri() -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return tracking_uri or "http://localhost:5000"


def _resolve_experiment_name() -> str:
    local_experiment = os.environ.get("FX_LOCAL_MLFLOW_EXPERIMENT")
    if local_experiment:
        return local_experiment
    return os.environ.get("MLFLOW_EXPERIMENT", "fx-local-vol")


def _resolve_suite_mode() -> str:
    mode = os.environ.get("FX_LOCAL_SUITE", "both").strip().lower()
    print(f"Suite mode: {mode}")
    if mode not in {"latent", "pca", "both"}:
        raise RuntimeError("FX_LOCAL_SUITE must be one of: latent, pca, both")
    return mode


def _resolve_datasets() -> list[tuple[str, str]]:
    mode = _resolve_suite_mode()
    latent = os.environ.get("DATASET_PATH")
    pca = os.environ.get("DATASET_PATH_TWO")

    datasets: list[tuple[str, str]] = []
    if mode in {"latent", "both"}:
        if not latent:
            raise RuntimeError("DATASET_PATH must be set for latent or both suite mode.")
        datasets.append(("latent", latent))
    if mode in {"pca", "both"}:
        if not pca:
            raise RuntimeError("DATASET_PATH_TWO must be set for pca or both suite mode.")
        datasets.append(("pca", pca))

    return datasets


def _resolve_target_cols() -> list[str]:
    raw = os.environ.get("FX_LOCAL_TARGET_COLS", "")
    if not raw.strip():
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _feature_set_for_experiment(experiment_name: str) -> str:
    if "_features_c_" in experiment_name:
        return "C"
    if "_features_b_" in experiment_name:
        return "B"
    return "A"


def _model_kind_for_experiment(experiment_name: str) -> str:
    if experiment_name == "local_base_shift1":
        return "shift1"
    if experiment_name == "local_base_zero":
        return "zero"
    return "gbr"


def main() -> None:
    registry = DictPluginRegistry({"fx.local_vol_regression": FxLocalVolRegressionPlugin()})
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )
    target_cols = _resolve_target_cols()

    for family, dataset_path in _resolve_datasets():
        for experiment_name in LADDER_EXPERIMENTS:
            spec = RunSpec(
                plugin_key="fx.local_vol_regression",
                pipeline="train",
                dataset_id=f"fx:local_vol_dataset:{family}:v1",
                data_spec={
                    "dataset_path": dataset_path,
                    "split_name": "default",
                },
                params={
                    "experiment": {
                        "name": experiment_name,
                        "target_cols": target_cols,
                        "feature_set": _feature_set_for_experiment(experiment_name),
                        "model_kind": _model_kind_for_experiment(experiment_name),
                    }
                },
                strict=True,
                tags={"suite": "fx_local_vol_condensed_ladder", "family": family},
            )
            result = run_pipeline(spec, registry=registry, tracking=tracking)
            print(family, experiment_name, result.status, result.run_id)


if __name__ == "__main__":
    main()
