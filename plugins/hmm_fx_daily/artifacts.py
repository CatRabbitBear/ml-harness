from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .data import HmmDataset


def _dump_model(model: Any) -> Any:
    if model is None:
        return None
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    return model


def write_dataset_summary(
    artifact_dir: Path,
    dataset: HmmDataset,
    *,
    dataset_id: str | None,
    split_name: str,
) -> Path:
    data_dir = artifact_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    split_sizes = {name: int(len(frame)) for name, frame in dataset.splits.items()}
    columns = list(dataset.loaded.df.columns) if dataset.loaded.df is not None else []

    summary = {
        "dataset_id": dataset_id,
        "dataset_path": str(dataset.path),
        "split_name": split_name,
        "index_col": dataset.index_col,
        "feature_cols": dataset.feature_cols,
        "target_cols": dataset.target_cols,
        "rows": split_sizes,
        "columns": columns,
        "manifest": _dump_model(dataset.loaded.manifest),
        "schema": _dump_model(dataset.loaded.schema),
        "roles": _dump_model(dataset.loaded.roles),
        "stats": _dump_model(dataset.loaded.stats),
        "split_specs": {
            name: _dump_model(spec) for name, spec in dataset.loaded.split_specs.items()
        },
        "warnings": dataset.loaded.warnings,
        "defaults_used": dataset.loaded.defaults_used,
    }

    path = data_dir / "dataset_summary.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return path
