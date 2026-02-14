from __future__ import annotations

import math

import numpy as np
import pandas as pd


def evaluate_predictions(
    *,
    predictions_by_split: dict[str, pd.DataFrame],
    top_quantile: float,
    target_epsilon: float,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_name, frame in predictions_by_split.items():
        if frame.empty:
            continue

        y_true = frame["y_true_raw"].to_numpy(dtype="float64")
        y_pred = frame["y_pred_raw"].to_numpy(dtype="float64")
        y_true_log = np.log(np.clip(y_true + target_epsilon, a_min=target_epsilon, a_max=None))
        y_pred_log = np.log(np.clip(y_pred + target_epsilon, a_min=target_epsilon, a_max=None))

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))
        mae_log = np.mean(np.abs(y_true_log - y_pred_log))
        rmse_log = math.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
        spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")

        precision, recall, lift = _topq_scores(y_true, y_pred, quantile=top_quantile)

        metrics[f"{split_name}_mae"] = float(mae)
        metrics[f"{split_name}_rmse"] = float(rmse)
        metrics[f"{split_name}_mae_log"] = float(mae_log)
        metrics[f"{split_name}_rmse_log"] = float(rmse_log)
        metrics[f"{split_name}_spearman"] = float(0.0 if pd.isna(spearman) else spearman)
        metrics[f"{split_name}_topq_precision"] = float(precision)
        metrics[f"{split_name}_topq_recall"] = float(recall)
        metrics[f"{split_name}_topq_lift"] = float(lift)
    return metrics


def compute_test_deltas(
    *,
    model_metrics: dict[str, float],
    persistence_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "test_delta_rmse_log_vs_persist": float(
            model_metrics.get("test_rmse_log", 0.0) - persistence_metrics.get("test_rmse_log", 0.0)
        ),
        "test_delta_spearman_vs_persist": float(
            model_metrics.get("test_spearman", 0.0) - persistence_metrics.get("test_spearman", 0.0)
        ),
        "test_delta_topq_precision_vs_persist": float(
            model_metrics.get("test_topq_precision", 0.0)
            - persistence_metrics.get("test_topq_precision", 0.0)
        ),
    }


def _topq_scores(
    y_true: np.ndarray, y_pred: np.ndarray, *, quantile: float
) -> tuple[float, float, float]:
    if y_true.size == 0:
        return 0.0, 0.0, 0.0
    true_cut = np.quantile(y_true, 1.0 - quantile)
    pred_cut = np.quantile(y_pred, 1.0 - quantile)
    true_top = y_true >= true_cut
    pred_top = y_pred >= pred_cut

    predicted_count = int(pred_top.sum())
    actual_count = int(true_top.sum())
    if predicted_count == 0:
        return 0.0, 0.0, 0.0

    hits = int((true_top & pred_top).sum())
    precision = float(hits / predicted_count)
    recall = float(hits / actual_count) if actual_count > 0 else 0.0
    lift = float(precision / quantile)
    return precision, recall, lift
