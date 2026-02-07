from __future__ import annotations

from core.api import run_pipeline
from core.contracts import PluginInfo, RunResult, RunSpec
from core.orchestration.registry import DictPluginRegistry
from core.tracking.fakes import FakeTrackingClient


class _CapturingPlugin:
    def __init__(self) -> None:
        self.context = None

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(key="capture", name="Capture Plugin", version="1.2.3")

    def run(self, spec: RunSpec, *, context) -> RunResult:
        self.context = context
        plugin_artifact = context.artifact_dir / "plugin" / "captured.txt"
        plugin_artifact.parent.mkdir(parents=True, exist_ok=True)
        plugin_artifact.write_text("ok", encoding="utf-8")
        context.tracking.log_metric("plugin_metric", 1.0)
        context.tracking.log_artifact(str(plugin_artifact), artifact_path="plugin")
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc="1970-01-01T00:00:00+00:00",
            ended_at_utc="1970-01-01T00:00:00+00:00",
            duration_s=0.0,
            outputs={"plugin_output": "ok"},
            message="plugin ok",
        )


class _FailingPlugin:
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(key="fail", name="Fail Plugin", version="0.0.1")

    def run(self, spec: RunSpec, *, context) -> RunResult:
        raise ValueError("boom")


class _FailingTrackingClient:
    def __init__(self) -> None:
        self.end_called = False

    @property
    def active_run_id(self) -> str | None:
        return None

    def start_run(self, *, run_name: str, tags: dict[str, str]) -> str:
        raise RuntimeError("start failed")

    def end_run(self, *, status: str) -> None:
        self.end_called = True


def test_run_pipeline_success_tracks_and_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(tmp_path))
    tracking = FakeTrackingClient(base_artifact_uri="file:///artifacts")
    plugin = _CapturingPlugin()
    registry = DictPluginRegistry(plugins={"capture": plugin})
    spec = RunSpec(plugin_key="capture", dataset_id="ds_1")

    result = run_pipeline(spec, registry=registry, tracking=tracking)

    assert result.run_id == "run_1"
    assert result.status == "ok"
    expected_dir = tmp_path / "runs" / "run_1"
    assert result.outputs["artifact_dir"] == str(expected_dir)
    assert result.outputs["artifact_uri"] == "file:///artifacts/run_1"
    assert plugin.context.run_id == "run_1"
    assert plugin.context.artifact_dir == expected_dir
    assert (expected_dir / "summary" / "run_summary.json").exists()
    assert [call.name for call in tracking.calls].count("start_run") == 1
    assert [call.name for call in tracking.calls].count("log_metric") == 1
    assert [call.name for call in tracking.calls].count("log_artifact") == 0
    assert [call.name for call in tracking.calls].count("log_artifacts") == 1
    assert [call.name for call in tracking.calls].count("end_run") == 1
    assert tracking.calls[-2].name == "log_artifacts"
    assert tracking.calls[-2].kwargs["local_dir"] == str(expected_dir)
    assert tracking.calls[-2].kwargs["artifact_path"] == "run"
    assert tracking.calls[-1].kwargs["status"] == "ok"


def test_run_pipeline_failure_writes_exception_and_ends_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(tmp_path))
    tracking = FakeTrackingClient()
    registry = DictPluginRegistry(plugins={"fail": _FailingPlugin()})
    spec = RunSpec(plugin_key="fail", dataset_id="ds_1")

    result = run_pipeline(spec, registry=registry, tracking=tracking)

    assert result.run_id == "run_1"
    assert result.status == "failed"
    expected_dir = tmp_path / "runs" / "run_1"
    assert (expected_dir / "errors" / "exception.txt").exists()
    assert (expected_dir / "summary" / "run_summary.json").exists()
    assert [call.name for call in tracking.calls].count("log_artifacts") == 1
    assert tracking.calls[-2].name == "log_artifacts"
    assert tracking.calls[-2].kwargs["local_dir"] == str(expected_dir)
    assert tracking.calls[-2].kwargs["artifact_path"] == "run"
    assert [call.name for call in tracking.calls].count("end_run") == 1
    assert tracking.calls[-1].kwargs["status"] == "failed"


def test_run_pipeline_tracking_start_failure_returns_failed():
    tracking = _FailingTrackingClient()
    registry = DictPluginRegistry(plugins={"capture": _CapturingPlugin()})
    spec = RunSpec(plugin_key="capture", dataset_id="ds_1")

    result = run_pipeline(spec, registry=registry, tracking=tracking)

    assert result.status == "failed"
    assert result.run_id == "not-started"
    assert tracking.end_called is False
