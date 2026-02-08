from __future__ import annotations

import json
import logging
import traceback
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from core.contracts import PluginNotFoundError, PluginRegistry, RunResult, RunSpec
from core.contracts.run_contracts.run_context import RunContext
from core.contracts.run_contracts.run_result import RunStatus
from core.contracts.tracking import TrackingClient
from core.runtime.artifacts import build_run_artifact_dir
from core.tracking import FakeTrackingClient


class SpecValidationError(ValueError):
    pass


def validate_spec(spec: RunSpec) -> None:
    errors: list[str] = []

    if not spec.plugin_key or not spec.plugin_key.strip():
        errors.append("plugin_key must be a non-empty string")

    if spec.dataset_id is None and not spec.data_spec:
        # v1 rule: you must either reference an existing frozen dataset_id
        # OR provide a data_spec so a dataset can be built.
        errors.append("Either dataset_id must be set or data_spec must be non-empty")

    if spec.seed is not None and spec.seed < 0:
        errors.append("seed must be >= 0 when provided")

    if errors:
        raise SpecValidationError("; ".join(errors))


def run_pipeline(
    spec: RunSpec,
    *,
    registry: PluginRegistry,
    tracking: TrackingClient | None = None,
) -> RunResult:
    """Run the v1 orchestration lifecycle for a spec."""
    validate_spec(spec)

    try:
        plugin = registry.get(spec.plugin_key)
    except PluginNotFoundError:
        start = datetime.now(UTC)
        end = datetime.now(UTC)
        return RunResult(
            run_id="not-started",
            status="failed",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={"plugin_key": spec.plugin_key},
            message=f"Plugin not found: {spec.plugin_key}",
        )

    tracking_client = tracking or FakeTrackingClient()
    run_name = spec.short_name()
    tags = _build_run_tags(spec, plugin)
    start = datetime.now(UTC)
    try:
        run_id = tracking_client.start_run(run_name=run_name, tags=tags)
    except Exception as exc:
        end = datetime.now(UTC)
        return RunResult(
            run_id="not-started",
            status="failed",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs={"plugin_key": spec.plugin_key},
            message=f"Tracking start failed: {exc}",
        )

    end_status: RunStatus = "failed"
    result: RunResult | None = None
    artifact_dir: Path | None = None

    try:
        artifact_dir = build_run_artifact_dir(run_id)
        logger = logging.getLogger(f"ml_harness.run.{run_id}")
        plugin_tracking = _ArtifactDeferredTrackingClient(tracking_client)
        context = RunContext(
            run_id=run_id,
            spec=spec,
            tracking=plugin_tracking,
            artifact_dir=artifact_dir,
            logger=logger,
        )

        try:
            plugin_result = plugin.run(spec, context=context)
        except Exception as exc:
            end = datetime.now(UTC)
            duration_s = (end - start).total_seconds()
            outputs = _merge_outputs(
                plugin_outputs=None,
                artifact_dir=artifact_dir,
                artifact_uri=tracking_client.get_artifact_uri(),
            )
            message = str(exc)
            try:
                _write_exception_artifact(artifact_dir, exc)
            except Exception:
                logging.getLogger("ml_harness.run_artifacts").warning(
                    "Failed to write exception artifact for %s",
                    run_id,
                    exc_info=True,
                )
            result = RunResult(
                run_id=run_id,
                status="failed",
                started_at_utc=start.isoformat(),
                ended_at_utc=end.isoformat(),
                duration_s=duration_s,
                outputs=outputs,
                message=message,
            )
            _write_run_summary_best_effort(
                artifact_dir=artifact_dir,
                spec=spec,
                plugin=plugin,
                run_result=result,
            )
            end_status = "failed"
        else:
            end = datetime.now(UTC)
            duration_s = (end - start).total_seconds()
            outputs = _merge_outputs(
                plugin_outputs=plugin_result.outputs,
                artifact_dir=artifact_dir,
                artifact_uri=tracking_client.get_artifact_uri(),
            )
            message = plugin_result.message if plugin_result.message else None
            result = RunResult(
                run_id=run_id,
                status="ok",
                started_at_utc=start.isoformat(),
                ended_at_utc=end.isoformat(),
                duration_s=duration_s,
                outputs=outputs,
                message=message,
            )
            _write_run_summary_best_effort(
                artifact_dir=artifact_dir,
                spec=spec,
                plugin=plugin,
                run_result=result,
            )
            end_status = "ok"
    except Exception as exc:
        end = datetime.now(UTC)
        outputs: dict[str, object] = {"plugin_key": spec.plugin_key}
        if artifact_dir is not None:
            outputs["artifact_dir"] = str(artifact_dir)
        result = RunResult(
            run_id=run_id,
            status="failed",
            started_at_utc=start.isoformat(),
            ended_at_utc=end.isoformat(),
            duration_s=(end - start).total_seconds(),
            outputs=outputs,
            message=f"Run failed: {exc}",
        )
        end_status = "failed"
    finally:
        if artifact_dir is not None and artifact_dir.exists():
            try:
                tracking_client.log_artifacts(str(artifact_dir), artifact_path="run")
            except Exception:
                logging.getLogger("ml_harness.run_artifacts").warning(
                    "Failed to bulk log artifacts for %s",
                    run_id,
                    exc_info=True,
                )
        tracking_client.end_run(status=end_status)

    return result


def _build_run_tags(spec: RunSpec, plugin: object) -> dict[str, str]:
    tags = dict(spec.tags)
    tags["plugin_key"] = spec.plugin_key
    tags["pipeline"] = spec.pipeline
    tags["run_mode"] = spec.run_mode
    if spec.dataset_id:
        tags["dataset_id"] = spec.dataset_id
    if spec.request_id:
        tags["request_id"] = spec.request_id
    if spec.seed is not None:
        tags["seed"] = str(spec.seed)
    plugin_info = getattr(plugin, "info", None)
    if plugin_info is not None:
        tags["plugin_name"] = plugin_info.name
        tags["plugin_version"] = plugin_info.version
    return tags


def _merge_outputs(
    *,
    plugin_outputs: Mapping[str, object] | None,
    artifact_dir: Path,
    artifact_uri: str | None,
) -> dict[str, object]:
    outputs: dict[str, object] = dict(plugin_outputs or {})
    outputs["artifact_dir"] = str(artifact_dir)
    if artifact_uri is not None:
        outputs["artifact_uri"] = artifact_uri
    return outputs


def _write_run_summary_best_effort(
    *,
    artifact_dir: Path,
    spec: RunSpec,
    plugin: object,
    run_result: RunResult,
) -> None:
    try:
        summary_path = artifact_dir / "summary" / "run_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        plugin_info = getattr(plugin, "info", None)
        summary = {
            "run_id": run_result.run_id,
            "status": run_result.status,
            "started_at_utc": run_result.started_at_utc,
            "ended_at_utc": run_result.ended_at_utc,
            "duration_s": run_result.duration_s,
            "spec": asdict(spec),
            "plugin": {
                "key": plugin_info.key,
                "name": plugin_info.name,
                "version": plugin_info.version,
            }
            if plugin_info is not None
            else None,
            "outputs": dict(run_result.outputs),
            "message": run_result.message,
        }
        with summary_path.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, default=str)
    except Exception:
        logging.getLogger("ml_harness.run_artifacts").warning(
            "Failed to write run summary for %s",
            run_result.run_id,
            exc_info=True,
        )


def _write_exception_artifact(artifact_dir: Path, exc: Exception) -> None:
    error_path = artifact_dir / "errors" / "exception.txt"
    error_path.parent.mkdir(parents=True, exist_ok=True)
    with error_path.open("w", encoding="utf-8") as handle:
        handle.write(str(exc))
        handle.write("\n")
        handle.write(traceback.format_exc())


class _ArtifactDeferredTrackingClient:
    """
    Delegate tracking calls while suppressing plugin-level artifact uploads.

    Artifacts are uploaded once in run_pipeline() finally via bulk log_artifacts.
    """

    def __init__(self, delegate: TrackingClient) -> None:
        self._delegate = delegate

    @property
    def active_run_id(self) -> str | None:
        return self._delegate.active_run_id

    def start_run(self, *, run_name: str, tags: Mapping[str, str]) -> str:
        return self._delegate.start_run(run_name=run_name, tags=tags)

    def end_run(self, *, status: RunStatus) -> None:
        self._delegate.end_run(status=status)

    def log_param(self, key: str, value: object) -> None:
        self._delegate.log_param(key, value)

    def log_params(self, params: Mapping[str, object]) -> None:
        self._delegate.log_params(params)

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        self._delegate.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        self._delegate.log_metrics(metrics, step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        self._delegate.set_tags(tags)

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        return None

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None:
        return None

    def get_artifact_uri(self) -> str | None:
        return self._delegate.get_artifact_uri()
