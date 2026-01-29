import pytest

from core.api import SpecValidationError, run_pipeline
from core.contracts import RunSpec


def test_run_pipeline_requires_dataset_or_data_spec():
    spec = RunSpec(plugin_key="fx_hmm", dataset_id=None, data_spec={})
    with pytest.raises(
        SpecValidationError,
        match="Either dataset_id must be set or data_spec must be non-empty",
    ):
        run_pipeline(spec)


def test_run_pipeline_ok_with_dataset_id():
    spec = RunSpec(plugin_key="fx_hmm", dataset_id="ds_123", seed=42)
    result = run_pipeline(spec)

    assert result.status == "ok"
    assert isinstance(result.run_id, str) and len(result.run_id) > 0
    assert result.outputs["dataset_id"] == "ds_123"
    assert result.outputs["plugin_key"] == "fx_hmm"


def test_run_pipeline_ok_with_data_spec_builds_dataset_id():
    spec = RunSpec(
        plugin_key="fx_hmm",
        dataset_id=None,
        data_spec={"source": "timescale", "pairs": ["EURUSD"]},
    )
    result = run_pipeline(spec)

    assert result.status == "ok"
    assert result.outputs["dataset_id"].startswith("ds_")
