from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

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


def write_state_summary(
    artifact_dir: Path,
    summary: pd.DataFrame,
) -> tuple[Path, Path]:
    data_dir = artifact_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "state_summary.csv"
    parquet_path = data_dir / "state_summary.parquet"
    summary.to_csv(csv_path, index=True)
    summary.to_parquet(parquet_path, index=True)
    return csv_path, parquet_path


def write_transmat(
    artifact_dir: Path,
    matrix: pd.DataFrame,
    *,
    name: str = "transmat",
) -> tuple[Path, Path]:
    data_dir = artifact_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / f"{name}.csv"
    parquet_path = data_dir / f"{name}.parquet"
    matrix.to_csv(csv_path, index=True)
    matrix.to_parquet(parquet_path, index=True)
    return csv_path, parquet_path


def write_state_timeseries(
    artifact_dir: Path,
    df: pd.DataFrame,
) -> tuple[Path, Path]:
    data_dir = artifact_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "state_timeseries.csv"
    parquet_path = data_dir / "state_timeseries.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)
    return csv_path, parquet_path


def write_json_artifact(artifact_dir: Path, name: str, payload: Any) -> Path:
    data_dir = artifact_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / name
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_model_artifacts(
    artifact_dir: Path,
    *,
    model: Any,
    scaler: Any | None,
) -> tuple[Path, Path | None]:
    import joblib

    models_dir = artifact_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "hmm_model.joblib"
    joblib.dump(model, model_path)

    scaler_path: Path | None = None
    if scaler is not None:
        scaler_path = models_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)

    return model_path, scaler_path
