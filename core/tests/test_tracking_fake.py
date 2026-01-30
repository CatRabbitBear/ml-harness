import pytest

from core.tracking.fakes import FakeTrackingClient


def test_fake_tracking_client_records_calls_in_order():
    client = FakeTrackingClient(base_artifact_uri="file:///tmp/artifacts")

    run_id = client.start_run(run_name="demo", tags={"team": "core"})
    client.log_param("alpha", 1)
    client.log_metric("loss", 0.42, step=1)
    client.set_tags({"stage": "train"})
    client.log_artifact("/tmp/model.bin", artifact_path="models")
    client.log_artifacts("/tmp/plots", artifact_path="plots")

    assert client.get_artifact_uri() == f"file:///tmp/artifacts/{run_id}"

    client.end_run(status="ok")

    names = [call.name for call in client.calls]
    assert names == [
        "start_run",
        "log_param",
        "log_metric",
        "set_tags",
        "log_artifact",
        "log_artifacts",
        "end_run",
    ]
    assert client.active_run_id is None
    assert client.get_artifact_uri() is None


def test_fake_tracking_client_strict_lifecycle():
    client = FakeTrackingClient()

    with pytest.raises(RuntimeError, match="No active run"):
        client.log_metric("loss", 1.0)

    client.start_run(run_name="demo", tags={})

    with pytest.raises(RuntimeError, match="already active"):
        client.start_run(run_name="dup", tags={})

    client.end_run(status="ok")

    with pytest.raises(RuntimeError, match="No active run"):
        client.end_run(status="failed")
