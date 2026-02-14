from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_diagnostic_plots(
    artifact_dir: Path,
    *,
    experiment_name: str,
    target_col: str,
    test_frame: pd.DataFrame,
    top_quantile: float,
    target_epsilon: float,
    overlay_years: list[int],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    out_dir = artifact_dir / "plots" / target_col
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    y_true_log = np.log(
        np.clip(
            test_frame["y_true_raw"].to_numpy(dtype="float64") + target_epsilon,
            target_epsilon,
            None,
        )
    )
    y_pred_log = np.log(
        np.clip(
            test_frame["y_pred_raw"].to_numpy(dtype="float64") + target_epsilon,
            target_epsilon,
            None,
        )
    )
    dates = pd.to_datetime(test_frame["date"], utc=True)

    scatter_path = out_dir / f"{experiment_name}__scatter_log.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true_log, y_pred_log, alpha=0.35, s=14)
    bound_min = float(min(y_true_log.min(), y_pred_log.min()))
    bound_max = float(max(y_true_log.max(), y_pred_log.max()))
    ax.plot([bound_min, bound_max], [bound_min, bound_max], linestyle="--")
    ax.set_xlabel("Actual log(rv)")
    ax.set_ylabel("Predicted log(rv)")
    ax.set_title(f"{target_col} | Pred vs Actual (log)")
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=150)
    plt.close(fig)
    written.append(scatter_path)

    for year in overlay_years:
        year_mask = dates.dt.year == year
        year_frame = test_frame.loc[year_mask]
        if year_frame.empty:
            continue
        year_dates = pd.to_datetime(year_frame["date"], utc=True)
        year_true_log = np.log(
            np.clip(
                year_frame["y_true_raw"].to_numpy(dtype="float64") + target_epsilon,
                target_epsilon,
                None,
            )
        )
        year_pred_log = np.log(
            np.clip(
                year_frame["y_pred_raw"].to_numpy(dtype="float64") + target_epsilon,
                target_epsilon,
                None,
            )
        )

        overlay_path = out_dir / f"{experiment_name}__overlay_{year}.png"
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(year_dates, year_true_log, label="actual_log", linewidth=1.5)
        ax.plot(year_dates, year_pred_log, label="pred_log", linewidth=1.2)
        ax.set_title(f"{target_col} | Actual vs Predicted log(rv) | {year}")
        ax.set_xlabel("Date")
        ax.set_ylabel("log(rv)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(overlay_path, dpi=150)
        plt.close(fig)
        written.append(overlay_path)

    event_path = out_dir / f"{experiment_name}__topq_events.png"
    true_vals = test_frame["y_true_raw"].to_numpy(dtype="float64")
    pred_vals = test_frame["y_pred_raw"].to_numpy(dtype="float64")
    true_cut = np.quantile(true_vals, 1.0 - top_quantile)
    pred_cut = np.quantile(pred_vals, 1.0 - top_quantile)
    true_top = true_vals >= true_cut
    pred_top = pred_vals >= pred_cut
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(dates, true_top.astype(int), label="actual_topq", linewidth=1.2)
    ax.plot(dates, pred_top.astype(int), label="pred_topq", linewidth=1.0)
    ax.set_title(f"{target_col} | Top-{int(top_quantile * 100)}% Event Flags")
    ax.set_xlabel("Date")
    ax.set_ylabel("Event")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(event_path, dpi=150)
    plt.close(fig)
    written.append(event_path)

    if "rms5__mean" in test_frame.columns and not test_frame["rms5__mean"].isna().all():
        residual_path = out_dir / f"{experiment_name}__residual_vs_rms5mean.png"
        residual_log = y_true_log - y_pred_log
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(test_frame["rms5__mean"], residual_log, alpha=0.35, s=14)
        ax.axhline(0.0, linestyle="--")
        ax.set_xlabel("rms5__mean")
        ax.set_ylabel("Residual (actual_log - pred_log)")
        ax.set_title(f"{target_col} | Residual vs rms5__mean")
        fig.tight_layout()
        fig.savefig(residual_path, dpi=150)
        plt.close(fig)
        written.append(residual_path)

    return written
