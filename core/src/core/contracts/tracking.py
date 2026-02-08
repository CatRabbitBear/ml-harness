from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from core.contracts.run_contracts.run_result import RunStatus


@runtime_checkable
class TrackingClient(Protocol):
    """
    Facade contract for experiment/run tracking backends.
    """

    @property
    def active_run_id(self) -> str | None:
        """Return the active run id, if any."""
        ...

    def start_run(self, *, run_name: str, tags: Mapping[str, str]) -> str:
        """Start a new run and return its run id."""
        ...

    def end_run(self, *, status: RunStatus) -> None:
        """End the active run with the given status."""
        ...

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        ...

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log multiple parameters."""
        ...

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Log a single metric."""
        ...

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log multiple metrics."""
        ...

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set tags on the active run."""
        ...

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        """Log a single artifact file."""
        ...

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None:
        """Log all artifacts within a directory."""
        ...

    def get_artifact_uri(self) -> str | None:
        """Return the active run's artifact URI, if any."""
        ...
