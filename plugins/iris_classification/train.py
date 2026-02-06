from __future__ import annotations

from .config import ModelConfig


def build_model(*, config: ModelConfig, seed: int | None):
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=config.C,
        max_iter=config.max_iter,
        solver="lbfgs",
        # multi_class="auto",
        random_state=seed,
    )


def fit_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model
