from __future__ import annotations

from core.contracts import Plugin, PluginInfo, RunResult, RunSpec
from core.contracts.run_contracts.run_context import RunContext


class DummyPlugin(Plugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(key="dummy", name="Dummy Plugin", version="0.1.0")

    def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
        return RunResult(
            run_id="dummy-run",
            status="ok",
            started_at_utc="1970-01-01T00:00:00+00:00",
            ended_at_utc="1970-01-01T00:00:00+00:00",
            duration_s=0.0,
            outputs={"used_registry": True, "plugin_key": spec.plugin_key},
            message="Dummy plugin executed",
        )
