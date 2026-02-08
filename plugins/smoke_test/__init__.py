from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from core.contracts import Plugin, PluginInfo, RunResult, RunSpec
from core.contracts.run_contracts.run_context import RunContext


class SmokeTestPlugin(Plugin):
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            key="smoke.test",
            name="Smoke Test Plugin",
            version="0.1.0",
            description="Evergreen smoke test plugin for MLflow tracking.",
        )

    def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
        start = datetime.now(UTC)

        context.tracking.log_param("smoke", "true")
        context.tracking.log_metric("smoke_metric", 1.0)

        artifact_path = self._write_smoke_artifact(context.artifact_dir)
        context.tracking.log_artifact(str(artifact_path), artifact_path="smoke")

        end = datetime.now(UTC)
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={"smoke_artifact": "smoke/hello.txt"},
        )

    @staticmethod
    def _write_smoke_artifact(artifact_dir: Path) -> Path:
        smoke_dir = artifact_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        path = smoke_dir / "hello.txt"
        path.write_text("hello from smoke test\n", encoding="utf-8")
        return path
