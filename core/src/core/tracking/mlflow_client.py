from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.contracts.tracking import RunStatus

try:
    import mlflow as _mlflow
except Exception:  # pragma: no cover - handled via runtime error
    _mlflow = None


def _require_mlflow() -> Any:
    if _mlflow is None:
        raise RuntimeError(
            "mlflow is not installed. Install mlflow or provide a fake module for tests."
        )
    return _mlflow


class MlflowTrackingClient:
    """
    MLflow-backed implementation of the TrackingClient facade.
    """

    def __init__(
        self,
        *,
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
    ) -> None:
        """Create a tracking client with optional MLflow configuration."""
        if experiment_name and experiment_id:
            raise ValueError("Provide either experiment_name or experiment_id, not both.")

        self._mlflow = _require_mlflow()
        self._experiment_id = experiment_id
        self._active_run_id: str | None = None

        if tracking_uri is not None:
            self._mlflow.set_tracking_uri(tracking_uri)
        if experiment_name is not None:
            self._mlflow.set_experiment(experiment_name)

    @property
    def active_run_id(self) -> str | None:
        """Return the active run id, if any."""
        return self._active_run_id

    def start_run(self, *, run_name: str, tags: Mapping[str, str]) -> str:
        """Start a new MLflow run and return its id."""
        self._ensure_no_active_run()
        run = self._mlflow.start_run(
            run_name=run_name,
            tags=dict(tags),
            experiment_id=self._experiment_id,
        )
        self._active_run_id = run.info.run_id
        return self._active_run_id

    def end_run(self, *, status: RunStatus) -> None:
        """End the active MLflow run."""
        self._ensure_active_run()
        self._mlflow.end_run(status=_map_run_status(status))
        self._active_run_id = None

    def log_param(self, key: str, value: Any) -> None:
        """Log a single parameter."""
        self._ensure_active_run()
        self._mlflow.log_param(key, value)

    def log_params(self, params: Mapping[str, Any]) -> None:
        """Log multiple parameters."""
        self._ensure_active_run()
        self._mlflow.log_params(dict(params))

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        """Log a single metric."""
        self._ensure_active_run()
        self._mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Mapping[str, float], *, step: int | None = None) -> None:
        """Log multiple metrics."""
        self._ensure_active_run()
        self._mlflow.log_metrics(dict(metrics), step=step)

    def set_tags(self, tags: Mapping[str, str]) -> None:
        """Set tags on the active run."""
        self._ensure_active_run()
        self._mlflow.set_tags(dict(tags))

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        """Log a single artifact file."""
        self._ensure_active_run()
        self._mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None:
        """Log all artifacts within a directory."""
        self._ensure_active_run()
        self._mlflow.log_artifacts(local_dir, artifact_path=artifact_path)

    def get_artifact_uri(self) -> str | None:
        """Return the active run's artifact URI, if any."""
        if self._active_run_id is None:
            return None
        return self._mlflow.get_artifact_uri()

    def _ensure_active_run(self) -> None:
        if self._active_run_id is None:
            raise RuntimeError("No active MLflow run. Call start_run first.")

    def _ensure_no_active_run(self) -> None:
        if self._active_run_id is not None or self._mlflow.active_run() is not None:
            raise RuntimeError("An MLflow run is already active.")


def _map_run_status(status: RunStatus) -> str:
    mapping = {
        "ok": "FINISHED",
        "failed": "FAILED",
        "skipped": "KILLED",
    }
    return mapping[status]
