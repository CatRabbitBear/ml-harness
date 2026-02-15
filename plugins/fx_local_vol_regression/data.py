from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DataConfig

if TYPE_CHECKING:
    import pandas as pd
    from mlh_data.io.artifact_loader import LoadedArtifact


@dataclass(frozen=True, slots=True)
class LocalDataset:
    path: Path
    loaded: LoadedArtifact
    full_df: pd.DataFrame
    index_col: str
    feature_cols: list[str]
    target_cols: list[str]


def load_dataset(config: DataConfig) -> LocalDataset:
    from mlh_data.io.artifact_loader import load_artifact

    path = Path(config.dataset_path)
    loaded = load_artifact(path, load_dataframe=config.load_dataframe)
    full_df = loaded.df
    if full_df is None or full_df.empty:
        raise ValueError("Dataset dataframe is empty.")

    roles = loaded.roles
    feature_cols = list(roles.feature_cols) if roles else []
    target_cols = list(roles.target_cols) if roles else []
    return LocalDataset(
        path=path,
        loaded=loaded,
        full_df=full_df.copy(),
        index_col=loaded.index_col,
        feature_cols=feature_cols,
        target_cols=target_cols,
    )
