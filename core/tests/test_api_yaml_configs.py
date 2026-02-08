from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from core.api import run_from_yaml, run_sweep_from_yaml
from core.contracts import PluginInfo, RunResult, RunSpec
from core.orchestration.registry import DictPluginRegistry
from core.tracking.fakes import FakeTrackingClient


class _ConfigurableCapturePlugin:
    def __init__(self) -> None:
        self.last_spec: RunSpec | None = None

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(key="capture.config", name="Capture Config", version="0.1.0")

    def default_params(self) -> dict[str, Any]:
        return {
            "model": {"depth": 3, "learning_rate": 0.1},
            "train": {"epochs": 10},
        }

    def validate_params(self, params: dict[str, Any], *, strict: bool = True) -> dict[str, Any]:
        payload = dict(params)
        allowed_top = {"model", "train"}
        if strict:
            unknown = sorted(set(payload) - allowed_top)
            if unknown:
                raise ValueError(f"params: unknown keys {unknown}")
        model = dict(payload.get("model", {}))
        train = dict(payload.get("train", {}))
        if "depth" in model:
            model["depth"] = int(model["depth"])
        if "learning_rate" in model:
            model["learning_rate"] = float(model["learning_rate"])
        if "epochs" in train:
            train["epochs"] = int(train["epochs"])
        payload["model"] = model
        payload["train"] = train
        return payload

    def run(self, spec: RunSpec, *, context) -> RunResult:
        self.last_spec = spec
        started = datetime.now(UTC)
        ended = datetime.now(UTC)
        return RunResult(
            run_id=context.run_id,
            status="ok",
            started_at_utc=started.isoformat(),
            ended_at_utc=ended.isoformat(),
            duration_s=(ended - started).total_seconds(),
            outputs={"captured": True},
        )


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def test_run_from_yaml_merges_defaults_resolves_env_and_writes_resolved_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setenv("HMM_DATASET_PATH", str(tmp_path / "dataset.parquet"))

    base_yaml = tmp_path / "run.yaml"
    _write_yaml(
        base_yaml,
        {
            "plugin": {"key": "capture.config"},
            "run": {"seed": 7, "strict": True, "tags": {"purpose": "test"}},
            "data": {"dataset_id": "demo:ds", "dataset_path": "${HMM_DATASET_PATH}"},
            "params": {"model": {"depth": 5}},
        },
    )

    plugin = _ConfigurableCapturePlugin()
    registry = DictPluginRegistry(plugins={plugin.info.key: plugin})
    tracking = FakeTrackingClient()

    result = run_from_yaml(base_yaml, registry=registry, tracking=tracking)

    assert result.status == "ok"
    assert plugin.last_spec is not None
    assert plugin.last_spec.params["model"]["depth"] == 5
    assert plugin.last_spec.params["model"]["learning_rate"] == 0.1
    assert plugin.last_spec.params["train"]["epochs"] == 10
    assert plugin.last_spec.data_spec["dataset_path"] == str(tmp_path / "dataset.parquet")

    run_dir = tmp_path / "runs" / result.run_id
    resolved_path = run_dir / "resolved" / "run_config.yaml"
    assert resolved_path.exists()
    resolved_payload = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
    assert resolved_payload["params"]["model"]["depth"] == 5
    assert resolved_payload["params"]["train"]["epochs"] == 10


def test_run_sweep_from_yaml_expands_grid_and_writes_sweep_summary(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("CORE_ARTIFACT_ROOT", str(tmp_path))

    base_yaml = tmp_path / "run.yaml"
    sweep_yaml = tmp_path / "sweep.yaml"

    _write_yaml(
        base_yaml,
        {
            "plugin": {"key": "capture.config"},
            "run": {"strict": True, "tags": {"purpose": "sweep"}},
            "data": {"dataset_id": "demo:ds"},
            "params": {"model": {"depth": 2}},
        },
    )
    _write_yaml(
        sweep_yaml,
        {
            "sweep": {
                "mode": "grid",
                "overrides": {
                    "params.model.depth": [2, 3],
                    "params.train.epochs": [5, 6],
                },
            }
        },
    )

    plugin = _ConfigurableCapturePlugin()
    registry = DictPluginRegistry(plugins={plugin.info.key: plugin})
    tracking = FakeTrackingClient()

    sweep_result = run_sweep_from_yaml(base_yaml, sweep_yaml, registry=registry, tracking=tracking)

    assert len(sweep_result.results) == 4
    assert len(sweep_result.failures) == 0
    summary_path = Path(str(sweep_result.metadata["sweep_summary_path"]))
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["variant_count"] == 4
    assert summary["failed_count"] == 0

    start_calls = [call for call in tracking.calls if call.name == "start_run"]
    assert len(start_calls) == 4
    for call in start_calls:
        tags = call.kwargs["tags"]
        assert tags["sweep_id"] == sweep_result.sweep_id
        assert "variant_id" in tags
        assert "params_hash" in tags
