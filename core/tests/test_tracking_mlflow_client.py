import importlib
import sys

import pytest

from core.tracking import mlflow_client


class _FakeRunInfo:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id


class _FakeRun:
    def __init__(self, run_id: str) -> None:
        self.info = _FakeRunInfo(run_id)


class FakeMlflow:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[object, ...], dict[str, object]]] = []
        self._active_run: _FakeRun | None = None

    def set_tracking_uri(self, uri: str) -> None:
        self.calls.append(("set_tracking_uri", (uri,), {}))

    def set_experiment(self, name: str) -> None:
        self.calls.append(("set_experiment", (name,), {}))

    def start_run(
        self,
        *,
        run_name: str,
        tags: dict[str, str],
        experiment_id: str | None = None,
    ) -> _FakeRun:
        self.calls.append(
            ("start_run", (run_name,), {"tags": tags, "experiment_id": experiment_id})
        )
        self._active_run = _FakeRun("run_123")
        return self._active_run

    def end_run(self, *, status: str) -> None:
        self.calls.append(("end_run", (), {"status": status}))
        self._active_run = None

    def active_run(self) -> _FakeRun | None:
        return self._active_run

    def log_param(self, key: str, value: object) -> None:
        self.calls.append(("log_param", (key, value), {}))

    def log_params(self, params: dict[str, object]) -> None:
        self.calls.append(("log_params", (), {"params": params}))

    def log_metric(self, key: str, value: float, *, step: int | None = None) -> None:
        self.calls.append(("log_metric", (key, value), {"step": step}))

    def log_metrics(self, metrics: dict[str, float], *, step: int | None = None) -> None:
        self.calls.append(("log_metrics", (), {"metrics": metrics, "step": step}))

    def set_tags(self, tags: dict[str, str]) -> None:
        self.calls.append(("set_tags", (), {"tags": tags}))

    def log_artifact(self, local_path: str, *, artifact_path: str | None = None) -> None:
        self.calls.append(("log_artifact", (local_path,), {"artifact_path": artifact_path}))

    def log_artifacts(self, local_dir: str, *, artifact_path: str | None = None) -> None:
        self.calls.append(("log_artifacts", (local_dir,), {"artifact_path": artifact_path}))

    def get_artifact_uri(self) -> str | None:
        if self._active_run is None:
            return None
        return f"fake://{self._active_run.info.run_id}"


def test_mlflow_tracking_client_uses_fake_module(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    module = importlib.reload(mlflow_client)
    client = module.MlflowTrackingClient(
        tracking_uri="http://mlflow",
        experiment_name="demo",
    )

    run_id = client.start_run(run_name="demo-run", tags={"env": "test"})
    client.log_param("p", 1)
    client.log_metric("m", 0.9, step=2)
    client.log_metrics({"m2": 0.2}, step=3)
    client.set_tags({"tag": "x"})
    client.log_artifact("/tmp/file.txt")
    client.log_artifacts("/tmp/dir", artifact_path="dir")

    assert client.get_artifact_uri() == f"fake://{run_id}"

    client.end_run(status="ok")
    assert client.active_run_id is None

    call_names = [call[0] for call in fake.calls]
    assert "set_tracking_uri" in call_names
    assert "set_experiment" in call_names
    assert "start_run" in call_names
    assert "log_metric" in call_names
    assert "end_run" in call_names


def test_mlflow_tracking_client_strict_lifecycle(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)

    module = importlib.reload(mlflow_client)
    client = module.MlflowTrackingClient(
        tracking_uri="http://mlflow",
        experiment_name="demo",
    )

    with pytest.raises(RuntimeError, match="No active MLflow run"):
        client.log_metric("loss", 1.0)

    client.start_run(run_name="demo-run", tags={})

    with pytest.raises(RuntimeError, match="already active"):
        client.start_run(run_name="dup", tags={})

    client.end_run(status="ok")

    with pytest.raises(RuntimeError, match="No active MLflow run"):
        client.end_run(status="failed")
