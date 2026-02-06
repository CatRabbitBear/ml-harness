from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DataConfig:
    dataset_path: str
    split_name: str = "default"
    load_dataframe: bool = True


@dataclass(frozen=True, slots=True)
class HmmConfig:
    data: DataConfig


def parse_config(data_spec: Mapping[str, Any] | None) -> HmmConfig:
    data_spec = data_spec or {}
    data_block = data_spec.get("data", {}) or {}

    dataset_path = (
        data_block.get("dataset_path")
        or data_spec.get("dataset_path")
        or os.getenv("HMM_DATASET_PATH")
        or ""
    )
    dataset_path = str(dataset_path)
    if not dataset_path:
        raise ValueError(
            "Dataset path not provided. Set data.dataset_path in RunSpec.data_spec "
            "or define HMM_DATASET_PATH in the environment."
        )

    split_name = str(data_block.get("split_name") or data_spec.get("split_name") or "default")
    load_dataframe = bool(data_block.get("load_dataframe", True))

    return HmmConfig(
        data=DataConfig(
            dataset_path=dataset_path,
            split_name=split_name,
            load_dataframe=load_dataframe,
        )
    )
