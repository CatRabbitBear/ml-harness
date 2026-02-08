from __future__ import annotations

from pathlib import Path

import pytest

from core.api import run_pipeline
from core.contracts.run_contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from core.tracking.fakes import FakeTrackingClient
from plugins.iris_classification import IrisClassificationPlugin


def _ensure_sklearn_deps() -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("matplotlib")


def test_iris_plugin_info() -> None:
    plugin = IrisClassificationPlugin()
    info = plugin.info
    assert info.key == "sklearn.iris_classification"
    assert info.name == "Iris Classification"
    assert info.version == "0.1.0"
    assert info.description


def test_iris_plugin_writes_expected_artifacts(tmp_path: Path, monkeypatch) -> None:
    _ensure_sklearn_deps()
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(tmp_path))
    plugin = IrisClassificationPlugin()
    registry = DictPluginRegistry(plugins={plugin.info.key: plugin})
    tracking = FakeTrackingClient()
    spec = RunSpec(plugin_key=plugin.info.key, dataset_id="sklearn:iris", data_spec={})

    result = run_pipeline(spec, registry=registry, tracking=tracking)
    assert result.status == "ok", result.message

    run_dir = tmp_path / "runs" / result.run_id
    assert (run_dir / "data" / "data_summary.json").exists()
    assert (run_dir / "metrics" / "metrics.json").exists()
    assert (run_dir / "models" / "model.joblib").exists()
    assert (run_dir / "plots" / "confusion_matrix.png").exists()


def test_iris_plugin_logs_metrics_and_artifacts(tmp_path: Path, monkeypatch) -> None:
    _ensure_sklearn_deps()
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(tmp_path))
    plugin = IrisClassificationPlugin()
    registry = DictPluginRegistry(plugins={plugin.info.key: plugin})
    tracking = FakeTrackingClient()
    spec = RunSpec(plugin_key=plugin.info.key, dataset_id="sklearn:iris", data_spec={})

    result = run_pipeline(spec, registry=registry, tracking=tracking)
    assert result.status == "ok", result.message

    metric_calls = [call for call in tracking.calls if call.name == "log_metric"]
    metric_names = {call.kwargs["key"] for call in metric_calls}
    expected_metrics = {
        "train_accuracy",
        "val_accuracy",
        "test_accuracy",
        "train_f1_macro",
        "val_f1_macro",
        "test_f1_macro",
    }
    assert expected_metrics.issubset(metric_names)

    artifact_calls = [call for call in tracking.calls if call.name == "log_artifact"]
    artifact_paths = {call.kwargs.get("artifact_path") for call in artifact_calls}
    assert {"data", "metrics", "models", "plots"}.issubset(artifact_paths)
