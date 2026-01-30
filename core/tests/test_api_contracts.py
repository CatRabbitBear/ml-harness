import pytest

from core.api import SpecValidationError, run_pipeline
from core.contracts import RunResult, RunSpec
from core.orchestration.registry import DictPluginRegistry
from core.testkit.dummies import DummyPlugin
from core.tracking.fakes import FakeTrackingClient


def test_run_pipeline_requires_dataset_or_data_spec():
    spec = RunSpec(plugin_key="fx_hmm", dataset_id=None, data_spec={})
    registry = DictPluginRegistry(plugins={"dummy": DummyPlugin()})
    tracking = FakeTrackingClient()

    with pytest.raises(
        SpecValidationError,
        match="Either dataset_id must be set or data_spec must be non-empty",
    ):
        run_pipeline(spec, registry=registry, tracking=tracking)


def test_run_pipeline_returns_failed_when_plugin_missing():
    spec = RunSpec(plugin_key="missing", dataset_id="ds_123")
    registry = DictPluginRegistry(plugins={})
    tracking = FakeTrackingClient()

    result = run_pipeline(spec, registry=registry, tracking=tracking)

    assert isinstance(result, RunResult)
    assert result.status == "failed"
    assert result.run_id == "not-started"
    assert "Plugin not found" in (result.message or "")
    assert tracking.calls == []
