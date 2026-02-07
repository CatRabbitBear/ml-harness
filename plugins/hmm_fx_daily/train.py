from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import ModelConfig, TrainConfig


@dataclass(frozen=True, slots=True)
class HmmFitResult:
    model: Any
    train_log_likelihood: float
    converged: bool
    iterations: int


def build_transmat_prior(
    *,
    n_components: int,
    mode: str,
    strength: float,
    custom_matrix: list[list[float]] | None,
) -> np.ndarray:
    if mode == "sticky_diag":
        prior = np.ones((n_components, n_components), dtype=float)
        prior += np.eye(n_components) * float(strength)
        return prior
    if mode == "uniform":
        return np.ones((n_components, n_components), dtype=float)
    if mode == "custom_matrix":
        if custom_matrix is None:
            raise ValueError("custom_matrix mode requires transmat_prior_custom")
        arr = np.array(custom_matrix, dtype=float)
        if arr.shape != (n_components, n_components):
            raise ValueError(
                f"custom_matrix must have shape ({n_components}, {n_components}), got {arr.shape}"
            )
        return arr
    raise ValueError(f"Unsupported transmat prior mode: {mode}")


def fit_best_hmm(
    X_train: np.ndarray,
    *,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    transmat_prior: np.ndarray,
) -> HmmFitResult:
    from hmmlearn.hmm import GaussianHMM

    rng = np.random.RandomState(train_cfg.random_seed)
    best_ll = -np.inf
    best_model: GaussianHMM | None = None
    best_converged = False
    best_iter = 0

    for _i in range(train_cfg.n_init):
        seed = int(rng.randint(0, 2**31 - 1))
        model = GaussianHMM(
            n_components=model_cfg.n_components,
            covariance_type=model_cfg.covariance_type,
            n_iter=train_cfg.n_iter,
            tol=train_cfg.tol,
            random_state=seed,
            init_params="tmc",
            params="stmc",
            transmat_prior=transmat_prior,
        )
        _apply_initialization(model, X_train, strategy=train_cfg.init_strategy, seed=seed)
        model.fit(X_train)

        ll = float(model.score(X_train))
        converged = bool(getattr(model.monitor_, "converged", False))
        iterations = int(getattr(model.monitor_, "iter", 0))

        if ll > best_ll:
            best_ll = ll
            best_model = model
            best_converged = converged
            best_iter = iterations

    if best_model is None:
        raise RuntimeError("Failed to fit HMM model (no successful initializations).")

    return HmmFitResult(
        model=best_model,
        train_log_likelihood=best_ll,
        converged=best_converged,
        iterations=best_iter,
    )


def _apply_initialization(
    model: Any,
    X: np.ndarray,
    *,
    strategy: str,
    seed: int,
) -> None:
    if strategy == "random":
        return

    if strategy != "kmeans":
        raise ValueError(f"Unsupported init_strategy: {strategy}")

    from sklearn.cluster import KMeans

    n_components = model.n_components
    kmeans = KMeans(n_clusters=n_components, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(X)
    model.means_ = kmeans.cluster_centers_

    n_features = X.shape[1]
    covariance_type = getattr(model, "covariance_type", "full")
    if covariance_type == "diag":
        covars = np.zeros((n_components, n_features), dtype=float)
        for k in range(n_components):
            cluster = X[labels == k]
            if cluster.shape[0] < 2:
                covars[k] = np.var(X, axis=0) + 1e-6
            else:
                covars[k] = np.var(cluster, axis=0) + 1e-6
        model.covars_ = covars
    else:
        covars = np.zeros((n_components, n_features, n_features), dtype=float)
        for k in range(n_components):
            cluster = X[labels == k]
            if cluster.shape[0] < 2:
                covars[k] = np.cov(X.T) + np.eye(n_features) * 1e-6
            else:
                covars[k] = np.cov(cluster.T) + np.eye(n_features) * 1e-6
        model.covars_ = covars
    model.startprob_ = np.full(n_components, 1.0 / n_components)
    model.init_params = "t"
