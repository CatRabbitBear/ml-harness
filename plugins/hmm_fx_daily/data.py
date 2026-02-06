from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DataConfig

if TYPE_CHECKING:
    import pandas as pd
    from mlh_data.io.artifact_loader import LoadedArtifact


@dataclass(frozen=True, slots=True)
class HmmDataset:
    path: Path
    loaded: LoadedArtifact
    splits: dict[str, pd.DataFrame]
    feature_cols: list[str]
    target_cols: list[str]
    index_col: str


def load_dataset(config: DataConfig) -> HmmDataset:
    from mlh_data.io.artifact_loader import get_split_frames, load_artifact

    path = Path(config.dataset_path)
    loaded = load_artifact(path, load_dataframe=config.load_dataframe)

    splits: dict[str, pd.DataFrame] = {}
    if config.load_dataframe:
        splits = get_split_frames(loaded, split_name=config.split_name)

    roles = loaded.roles
    feature_cols = list(roles.feature_cols) if roles else []
    target_cols = list(roles.target_cols) if roles else []

    return HmmDataset(
        path=path,
        loaded=loaded,
        splits=splits,
        feature_cols=feature_cols,
        target_cols=target_cols,
        index_col=loaded.index_col,
    )


def get_feature_frame(dataset: HmmDataset) -> pd.DataFrame:
    df = dataset.loaded.df
    if df is None or df.empty:
        raise ValueError("Dataset dataframe is empty; cannot build feature matrix.")

    if not dataset.feature_cols:
        raise ValueError("No feature columns defined in roles.json.")

    missing = [c for c in dataset.feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Feature columns missing from dataframe: {missing}")

    feature_df = df.loc[:, dataset.feature_cols].copy()
    feature_df = feature_df.astype("float64")
    return feature_df


def get_time_index(dataset: HmmDataset) -> pd.Series:
    import pandas as pd

    df = dataset.loaded.df
    if df is None or df.empty:
        raise ValueError("Dataset dataframe is empty; cannot build time index.")

    index_col = dataset.index_col
    if index_col in df.columns:
        series = pd.to_datetime(df[index_col], utc=True, errors="coerce")
    else:
        if df.index.name != index_col:
            raise ValueError(f"Index column '{index_col}' not found in dataframe columns or index.")
        series = pd.to_datetime(df.index, utc=True, errors="coerce")

    if series.isna().any():
        raise ValueError("Time index contains NaT values after parsing.")

    return series
