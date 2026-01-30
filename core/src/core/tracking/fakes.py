from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from core.contracts.tracking import RunStatus


@dataclass(frozen=True, slots=True)
class TrackingCall:
    """Record of a tracking call for assertions in tests."""

    name: str
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class FakeTrackingClient:
    """
    In-memory TrackingClient for unit tests.
    """

    def __init__(self, *, base_artifact_uri: str | None = None) -> None:
        self._base_artifact_uri = base_artifact_uri
        self._active_run_id: str | None = None
        self._run_counter = 0
        self._calls: list[TrackingCall] = []

    @property
    def active_run_id(self) -> str | None:
        """Return the active run id, if any."""
        return self._active_run_id

    @property
    def calls(self) -> list[TrackingCall]:
        """Return the recorded calls in order."""
        return list(self._calls)

    def start_run(self, *, run_name: str, tags: Mapping[str, str]) -> str:
        """Start a fake run and return its id."""
        self._ensure_no_active_run()
        self._run_counter += 1
        self._active_run_id = f"run_{self._run_counter}"
        self._record("start_run", run_name=run_name, tags=dict(tags))
        return self._active_run_id

    def end_run(self, *, status: RunStatus) -> None:
        """End the active fake run."""
        self._ensure_active_run()
        self._record("end_run", status=status)
        self._active_run_id = None

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._ensure_active_run()
        self._record("log_param", key=key, value=value)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log multiple parameters."""
        self._ensure_active_run()
        self._record("log_params", params=dict(params))

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Log a single metric."""
        self._ensure_active_run()
        self._record("log_metric", key=key, value=value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log multiple metrics."""
        self._ensure_active_run()
        self._record("log_metrics", metrics=dict(metrics), step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set tags on the active run."""
        self._ensure_active_run()
        self._record("set_tags", tags=dict(tags))

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        """Log a single artifact file."""
        self._ensure_active_run()
        self._record("log_artifact", local_path=local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None:
        """Log all artifacts within a directory."""
        self._ensure_active_run()
        self._record("log_artifacts", local_dir=local_dir, artifact_path=artifact_path)

    def get_artifact_uri(self) -> str | None:
        """Return a fake artifact URI for the active run, if any."""
        if self._active_run_id is None or self._base_artifact_uri is None:
            return None
        return f"{self._base_artifact_uri}/{self._active_run_id}"

    def _record(self, name: str, **kwargs: Any) -> None:
        self._calls.append(TrackingCall(name=name, args=(), kwargs=kwargs))

    def _ensure_active_run(self) -> None:
        if self._active_run_id is None:
            raise RuntimeError("No active run. Call start_run first.")

    def _ensure_no_active_run(self) -> None:
        if self._active_run_id is not None:
            raise RuntimeError("A run is already active.")
