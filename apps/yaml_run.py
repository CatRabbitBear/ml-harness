from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.api import run_from_yaml, run_sweep_from_yaml
from core.orchestration.registry import DictPluginRegistry
from plugins.hmm_fx_daily import HmmFxDailyPlugin
from plugins.iris_classification import IrisClassificationPlugin
from plugins.smoke_test import SmokeTestPlugin
from tracking_clients import MlflowTrackingClient


def _resolve_tracking_uri() -> str:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    return tracking_uri or "http://localhost:5000"


def _resolve_experiment_name() -> str:
    return os.environ.get("MLFLOW_EXPERIMENT", "ml-harness-dev")


def build_registry() -> DictPluginRegistry:
    plugins = {
        "smoke.test": SmokeTestPlugin(),
        "sklearn.iris_classification": IrisClassificationPlugin(),
        "hmm.fx_daily": HmmFxDailyPlugin(),
    }
    return DictPluginRegistry(plugins=plugins)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ml-harness from YAML config.")
    parser.add_argument("base_yaml", type=Path, help="Path to base run YAML")
    parser.add_argument(
        "--sweep",
        dest="sweep_yaml",
        type=Path,
        default=None,
        help="Optional sweep YAML path for grid expansion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    registry = build_registry()
    tracking = MlflowTrackingClient(
        tracking_uri=_resolve_tracking_uri(),
        experiment_name=_resolve_experiment_name(),
    )

    if args.sweep_yaml is None:
        result = run_from_yaml(args.base_yaml, registry=registry, tracking=tracking)
        print(result)
        return

    sweep_result = run_sweep_from_yaml(
        args.base_yaml,
        args.sweep_yaml,
        registry=registry,
        tracking=tracking,
    )
    print(sweep_result)


if __name__ == "__main__":
    main()
