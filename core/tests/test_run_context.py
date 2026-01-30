import logging

from core.contracts.plugin import Plugin, PluginInfo
from core.contracts.run_context import RunContext
from core.contracts.run_result import RunResult
from core.contracts.run_spec import RunSpec
from core.tracking.fakes import FakeTrackingClient


def test_run_context_construction(tmp_path):
    spec = RunSpec(plugin_key="demo", dataset_id="ds_1")
    tracking = FakeTrackingClient()
    logger = logging.getLogger("test.run_context")

    context = RunContext(
        run_id="run_123",
        spec=spec,
        tracking=tracking,
        artifact_dir=tmp_path,
        logger=logger,
    )

    assert context.run_id == "run_123"
    assert context.spec is spec
    assert context.tracking is tracking
    assert context.artifact_dir == tmp_path
    assert context.logger is logger


def test_plugin_protocol_accepts_run_context():
    class ExamplePlugin:
        @property
        def info(self) -> PluginInfo:
            return PluginInfo(key="example", name="Example")

        def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
            return RunResult(
                run_id=context.run_id,
                status="ok",
                started_at_utc="1970-01-01T00:00:00+00:00",
                ended_at_utc="1970-01-01T00:00:00+00:00",
                duration_s=0.0,
                outputs={"plugin_key": spec.plugin_key},
            )

    plugin = ExamplePlugin()

    assert isinstance(plugin, Plugin)
