from __future__ import annotations

from .data import DatasetSplits


def evaluate_model(model, splits: DatasetSplits) -> tuple[dict[str, float], object]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

    metrics: dict[str, float] = {}

    metrics["train_accuracy"] = float(accuracy_score(splits.y_train, model.predict(splits.X_train)))
    metrics["val_accuracy"] = float(accuracy_score(splits.y_val, model.predict(splits.X_val)))
    metrics["test_accuracy"] = float(accuracy_score(splits.y_test, model.predict(splits.X_test)))

    metrics["train_f1_macro"] = float(
        f1_score(splits.y_train, model.predict(splits.X_train), average="macro")
    )
    metrics["val_f1_macro"] = float(
        f1_score(splits.y_val, model.predict(splits.X_val), average="macro")
    )
    metrics["test_f1_macro"] = float(
        f1_score(splits.y_test, model.predict(splits.X_test), average="macro")
    )

    cm = confusion_matrix(splits.y_test, model.predict(splits.X_test))
    return metrics, cm
