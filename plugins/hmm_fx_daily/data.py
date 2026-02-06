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
