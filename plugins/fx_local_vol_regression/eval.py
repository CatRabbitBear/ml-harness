from __future__ import annotations

import math

import numpy as np
import pandas as pd


def evaluate_predictions(
    *,
    predictions_by_split: dict[str, pd.DataFrame],
    top_quantile: float,
    target_epsilon: float,
    use_log_metrics: bool,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for split_name, frame in predictions_by_split.items():
        if frame.empty:
            continue

        y_true = frame["y_true_raw"].to_numpy(dtype="float64")
        y_pred = frame["y_pred_raw"].to_numpy(dtype="float64")

        mae = np.mean(np.abs(y_true - y_pred))
        rmse = math.sqrt(np.mean((y_true - y_pred) ** 2))

        if use_log_metrics:
            y_true_log = np.log(np.clip(y_true + target_epsilon, a_min=target_epsilon, a_max=None))
            y_pred_log = np.log(np.clip(y_pred + target_epsilon, a_min=target_epsilon, a_max=None))
            mae_log = np.mean(np.abs(y_true_log - y_pred_log))
            rmse_log = math.sqrt(np.mean((y_true_log - y_pred_log) ** 2))
        else:
            mae_log = mae
            rmse_log = rmse

        spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")

        precision, recall, lift, mean_pred_topq, mean_true_topq = _topq_scores(
            y_true, y_pred, quantile=top_quantile
        )
        tp_precision, tp_recall, tp_f1, tp_rate_true, tp_rate_pred, delta_sign_acc = (
            _turning_point_scores(y_true, y_pred)
        )

        mean_pred = float(np.mean(y_pred))
        mean_true = float(np.mean(y_true))
        mean_error = float(np.mean(y_pred - y_true))
        top_true_cut = np.quantile(y_true, 1.0 - top_quantile)
        top_mask = y_true >= top_true_cut
        mean_error_topq = float(np.mean((y_pred - y_true)[top_mask])) if np.any(top_mask) else 0.0

        metrics[f"{split_name}_mae"] = float(mae)
        metrics[f"{split_name}_rmse"] = float(rmse)
        metrics[f"{split_name}_mae_log"] = float(mae_log)
        metrics[f"{split_name}_rmse_log"] = float(rmse_log)
        metrics[f"{split_name}_spearman"] = float(0.0 if pd.isna(spearman) else spearman)
        metrics[f"{split_name}_topq_precision"] = float(precision)
        metrics[f"{split_name}_topq_recall"] = float(recall)
        metrics[f"{split_name}_topq_lift"] = float(lift)
        metrics[f"{split_name}_topq_mean_pred"] = float(mean_pred_topq)
        metrics[f"{split_name}_topq_mean_true"] = float(mean_true_topq)
        metrics[f"{split_name}_mean_pred"] = mean_pred
        metrics[f"{split_name}_mean_true"] = mean_true
        metrics[f"{split_name}_mean_error"] = mean_error
        metrics[f"{split_name}_mean_error_topq"] = mean_error_topq
        metrics[f"{split_name}_turn_precision"] = float(tp_precision)
        metrics[f"{split_name}_turn_recall"] = float(tp_recall)
        metrics[f"{split_name}_turn_f1"] = float(tp_f1)
        metrics[f"{split_name}_turn_rate_true"] = float(tp_rate_true)
        metrics[f"{split_name}_turn_rate_pred"] = float(tp_rate_pred)
        metrics[f"{split_name}_delta_sign_acc"] = float(delta_sign_acc)
    return metrics


def compute_test_deltas(
    *,
    model_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
) -> dict[str, float]:
    return {
        "test_delta_rmse_vs_zero": float(
            model_metrics.get("test_rmse", 0.0) - baseline_metrics.get("test_rmse", 0.0)
        ),
        "test_delta_spearman_vs_zero": float(
            model_metrics.get("test_spearman", 0.0) - baseline_metrics.get("test_spearman", 0.0)
        ),
        "test_delta_topq_precision_vs_zero": float(
            model_metrics.get("test_topq_precision", 0.0)
            - baseline_metrics.get("test_topq_precision", 0.0)
        ),
    }


def _topq_scores(
    y_true: np.ndarray, y_pred: np.ndarray, *, quantile: float
) -> tuple[float, float, float, float, float]:
    if y_true.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    true_cut = np.quantile(y_true, 1.0 - quantile)
    pred_cut = np.quantile(y_pred, 1.0 - quantile)
    true_top = y_true >= true_cut
    pred_top = y_pred >= pred_cut

    predicted_count = int(pred_top.sum())
    actual_count = int(true_top.sum())
    if predicted_count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    hits = int((true_top & pred_top).sum())
    precision = float(hits / predicted_count)
    recall = float(hits / actual_count) if actual_count > 0 else 0.0
    lift = float(precision / quantile)
    mean_pred_topq = float(np.mean(y_pred[pred_top])) if np.any(pred_top) else 0.0
    mean_true_topq = float(np.mean(y_true[true_top])) if np.any(true_top) else 0.0
    return precision, recall, lift, mean_pred_topq, mean_true_topq


def _turning_point_scores(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float, float, float, float]:
    n = y_true.size
    if n < 3:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    true_delta = np.diff(y_true)
    pred_delta = np.diff(y_pred)
    true_sign = np.sign(true_delta)
    pred_sign = np.sign(pred_delta)

    # Turning event at t means slope sign changed from t-1 to t.
    true_turn = true_sign[1:] != true_sign[:-1]
    pred_turn = pred_sign[1:] != pred_sign[:-1]

    tp = int(np.sum(true_turn & pred_turn))
    pred_pos = int(np.sum(pred_turn))
    true_pos = int(np.sum(true_turn))

    precision = float(tp / pred_pos) if pred_pos > 0 else 0.0
    recall = float(tp / true_pos) if true_pos > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    rate_true = float(true_pos / true_turn.size) if true_turn.size > 0 else 0.0
    rate_pred = float(pred_pos / pred_turn.size) if pred_turn.size > 0 else 0.0

    valid = (true_sign != 0) & (pred_sign != 0)
    if np.any(valid):
        delta_sign_acc = float(np.mean(true_sign[valid] == pred_sign[valid]))
    else:
        delta_sign_acc = 0.0

    return precision, recall, f1, rate_true, rate_pred, delta_sign_acc
