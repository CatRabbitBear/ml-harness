from __future__ import annotations

from dataclasses import dataclass

from .config import SplitConfig


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    X_train: object
    X_val: object
    X_test: object
    y_train: object
    y_val: object
    y_test: object
    feature_names: list[str]
    target_names: list[str]


def load_iris_splits(*, split: SplitConfig, seed: int | None) -> DatasetSplits:
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    dataset = load_iris()
    X = dataset.data
    y = dataset.target

    feature_names = list(dataset.feature_names)
    target_names = list(dataset.target_names)

    holdout_size = split.val + split.test
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X,
        y,
        test_size=holdout_size,
        random_state=seed,
        stratify=y,
    )

    test_ratio = split.test / holdout_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout,
        y_holdout,
        test_size=test_ratio,
        random_state=seed,
        stratify=y_holdout,
    )

    return DatasetSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
    )
