from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from core.contracts import PluginNotFoundError, PluginRegistry, RunResult, RunSpec


class SpecValidationError(ValueError):
    pass


class RegistryNotConfiguredError(RuntimeError):
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


def run_pipeline(spec: RunSpec, *, registry: PluginRegistry | None = None) -> RunResult:
    validate_spec(spec)

    if registry is None:
        # v1 decision: either keep dummy behaviour, OR require registry explicitly.
        return _dummy_run(spec)

    try:
        plugin = registry.get(spec.plugin_key)
    except PluginNotFoundError as e:
        raise RegistryNotConfiguredError(f"Plugin not found: {spec.plugin_key}") from e

    # context is empty for now; later we’ll add mlflow backend, settings, artifact root, logger etc.
    return plugin.run(spec, context={})


def _dummy_run(spec: RunSpec) -> RunResult:
    # In real v1, this becomes: resolve plugin -> orchestrator lifecycle -> mlflow logging -> report artifact.
    start = datetime.now(UTC)

    # Pretend we did some work:
    # - resolved dataset_id (either used existing or "built" one)
    # - produced a model URI (or just a placeholder)
    resolved_dataset_id = spec.dataset_id or f"ds_{uuid4().hex[:10]}"
    run_id = uuid4().hex

    # lightweight outputs that look like the real system will later
    outputs = {
        "plugin_key": spec.plugin_key,
        "pipeline": spec.pipeline,
        "run_mode": spec.run_mode,
        "dataset_id": resolved_dataset_id,
        "echo_spec": asdict(spec),
        "artifacts": {
            "run_report": f"runs/{run_id}/run_report.json",
        },
    }

    end = datetime.now(UTC)
    duration_s = (end - start).total_seconds()

    return RunResult(
        run_id=run_id,
        status="ok",
        started_at_utc=start.isoformat(),
        ended_at_utc=end.isoformat(),
        duration_s=duration_s,
        outputs=outputs,
        message="Dummy run executed successfully (no real training yet).",
    )
