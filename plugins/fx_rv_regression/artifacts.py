from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def write_data_summary(
    artifact_dir: Path,
    *,
    dataset_id: str,
    dataset_path: str,
    index_col: str,
    row_counts: dict[str, int],
    split_dates: dict[str, str],
    experiment_name: str,
    target_cols: list[str],
) -> Path:
    path = artifact_dir / "data" / "data_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset_id": dataset_id,
        "dataset_path": dataset_path,
        "index_col": index_col,
        "row_counts": row_counts,
        "split_dates": split_dates,
        "experiment_name": experiment_name,
        "target_cols": target_cols,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path


def write_metrics(artifact_dir: Path, metrics: dict[str, float]) -> Path:
    path = artifact_dir / "metrics" / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return path


def write_predictions(
    artifact_dir: Path,
    *,
    target_col: str,
    split_name: str,
    frame: pd.DataFrame,
) -> Path:
    path = artifact_dir / "data" / "predictions" / f"{target_col}__{split_name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=True)
    return path


def write_model(artifact_dir: Path, *, target_col: str, model: Any) -> Path:
    from joblib import dump

    path = artifact_dir / "models" / f"{target_col}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, path)
    return path


def write_baseline_spec(artifact_dir: Path, *, target_col: str, strategy: str) -> Path:
    path = artifact_dir / "models" / f"{target_col}_{strategy}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"target_col": target_col, "strategy": strategy}
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
