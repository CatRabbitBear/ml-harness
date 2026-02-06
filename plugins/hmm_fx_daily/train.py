from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class HmmTrainConfig:
    n_states: int = 3
    covariance_type: str = "full"
    n_iter: int = 100


def build_hmm_model(config: HmmTrainConfig, *, seed: int | None) -> Any:
    """
    Placeholder for HMM model construction.
    """
    raise NotImplementedError("HMM model construction not implemented yet.")


def fit_hmm_model(model: Any, X: Any) -> Any:
    """
    Placeholder for HMM model fitting.
    """
    raise NotImplementedError("HMM model fitting not implemented yet.")
