from core.api import run_pipeline
from core.contracts import RunSpec
from core.orchestration.registry import DictPluginRegistry
from core.testkit.dummies import DummyPlugin
from core.tracking.fakes import FakeTrackingClient


def test_run_pipeline_uses_injected_registry():
    registry = DictPluginRegistry(plugins={"dummy": DummyPlugin()})
    tracking = FakeTrackingClient()
    spec = RunSpec(plugin_key="dummy", dataset_id="ds_123")

    result = run_pipeline(spec, registry=registry, tracking=tracking)

    assert result.outputs["used_registry"] is True
    assert result.outputs["plugin_key"] == "dummy"
