from __future__ import annotations

import os

from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from core.tracking.mlflow_client import MlflowTrackingClient
from plugins.smoke_test import SmokeTestPlugin


def _resolve_tracking_uri() -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return tracking_uri or "http://localhost:5000"


def _resolve_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "ml-harness-dev")


def main() -> None:
    registry = DictPluginRegistry(plugins={"smoke.test": SmokeTestPlugin()})
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )

    spec = RunSpec(
        plugin_key="smoke.test",
        data_spec={"smoke": True},
        tags={"purpose": "smoke"},
    )

    result = run_pipeline(spec, registry=registry, tracking=tracking)
    print(result)


if __name__ == "__main__":
    main()
