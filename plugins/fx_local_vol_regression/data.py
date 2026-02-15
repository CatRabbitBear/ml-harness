from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from .config import DataConfig, SplitConfig

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


@dataclass(frozen=True, slots=True)
class DataFrameSplits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


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


def split_by_date(df: pd.DataFrame, *, index_col: str, split_cfg: SplitConfig) -> DataFrameSplits:
    work = df.copy()
    dates = _resolve_date_index(work, index_col=index_col)
    work["__date__"] = dates
    work = work.sort_values("__date__")

    train_end = _as_ts(split_cfg.train_end_date)
    val_end = _as_ts(split_cfg.val_end_date)
    test_end = _as_ts(split_cfg.test_end_date)
    if not (train_end < val_end < test_end):
        raise ValueError("split dates must satisfy train_end < val_end < test_end")

    train = work[work["__date__"] <= train_end]
    val = work[(work["__date__"] > train_end) & (work["__date__"] <= val_end)]
    test = work[(work["__date__"] > val_end) & (work["__date__"] <= test_end)]

    if train.empty:
        raise ValueError("Train split is empty. Check split dates and dataset coverage.")
    if val.empty:
        raise ValueError("Validation split is empty. Check split dates and dataset coverage.")
    if test.empty:
        raise ValueError("Test split is empty. Check split dates and dataset coverage.")

    return DataFrameSplits(train=train, val=val, test=test)


def _resolve_date_index(df: pd.DataFrame, *, index_col: str):
    import pandas as pd

    if index_col in df.columns:
        parsed = pd.to_datetime(df[index_col], utc=True, errors="coerce")
    else:
        parsed = pd.to_datetime(df.index, utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError("Failed to parse datetime index from dataset.")
    return parsed


def _as_ts(date_str: str):
    import pandas as pd

    return pd.Timestamp(date_str, tz="UTC")
