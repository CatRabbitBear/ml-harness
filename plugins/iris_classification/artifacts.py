from __future__ import annotations

import json
from pathlib import Path

from .data import DatasetSplits


def write_data_summary(
    artifact_dir: Path,
    *,
    dataset_id: str,
    feature_names: list[str],
    target_names: list[str],
    splits: DatasetSplits,
    seed: int | None,
) -> Path:
    summary_path = artifact_dir / "data" / "data_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset_id": dataset_id,
        "seed": seed,
        "feature_names": feature_names,
        "target_names": target_names,
        "row_counts": {
            "train": int(len(splits.y_train)),
            "val": int(len(splits.y_val)),
            "test": int(len(splits.y_test)),
        },
        "class_distribution": {
            "train": _class_distribution(splits.y_train, target_names),
            "val": _class_distribution(splits.y_val, target_names),
            "test": _class_distribution(splits.y_test, target_names),
        },
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    return summary_path


def write_metrics(artifact_dir: Path, metrics: dict[str, float]) -> Path:
    metrics_path = artifact_dir / "metrics" / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics_path


def write_model(artifact_dir: Path, model) -> Path:
    from joblib import dump

    model_path = artifact_dir / "models" / "model.joblib"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    dump(model, model_path)
    return model_path


def write_confusion_matrix_plot(
    artifact_dir: Path,
    *,
    confusion_matrix,
    labels: list[str],
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plot_path = artifact_dir / "plots" / "confusion_matrix.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Test)")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)

    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, str(confusion_matrix[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    return plot_path


def _class_distribution(targets, target_names: list[str]) -> dict[str, int]:
    from numpy import bincount

    counts = bincount(targets, minlength=len(target_names))
    return {name: int(counts[i]) for i, name in enumerate(target_names)}
