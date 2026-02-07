from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_regime_overlay(
    out_path: Path,
    *,
    dates: pd.Series,
    proxy: pd.Series,
    states: np.ndarray,
    max_posterior: np.ndarray | None,
) -> Path:
    dates = _coerce_dates(dates)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, proxy, color="black", linewidth=1.0, label="proxy")

    colors = _state_colors(int(states.max()) + 1 if states.size else 1)
    segments = _state_segments(states)
    for start, end, state in segments:
        alpha = 0.2
        if max_posterior is not None:
            alpha = float(np.clip(np.mean(max_posterior[start:end]), 0.1, 0.6))
        ax.axvspan(dates.iloc[start], dates.iloc[end - 1], color=colors[state], alpha=alpha)

    ax.set_title("Regime Overlay")
    ax.set_xlabel("Date")
    ax.set_ylabel("Proxy")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_state_duration_hist(
    out_path: Path,
    *,
    durations_per_state: dict[int, list[int]],
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4))
    for state, durations in durations_per_state.items():
        if not durations:
            continue
        ax.hist(durations, bins=30, alpha=0.5, label=f"state {state}")
    ax.set_title("State Duration Histogram")
    ax.set_xlabel("Duration (days)")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_state_occupancy_rolling(
    out_path: Path,
    *,
    dates: pd.Series,
    states: np.ndarray,
    n_components: int,
    window: int = 126,
) -> Path:
    dates = _coerce_dates(dates)
    fig, ax = plt.subplots(figsize=(12, 4))
    state_series = pd.Series(states, index=dates)
    for state in range(n_components):
        occupancy = (state_series == state).rolling(window=window, min_periods=1).mean()
        ax.plot(occupancy.index, occupancy.values, label=f"state {state}")
    ax.set_title(f"Rolling State Occupancy (window={window})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Occupancy")
    ax.legend()
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_feature_means_by_state(
    out_path: Path,
    *,
    state_summary: pd.DataFrame,
    feature_cols: Iterable[str],
) -> Path:
    means = []
    for col in feature_cols:
        means.append(state_summary[f"mean_{col}"].values)
    means = np.array(means)

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(feature_cols))
    width = 0.8 / state_summary.shape[0]
    for i, state in enumerate(state_summary.index):
        ax.bar(
            x + i * width,
            means[:, i],
            width=width,
            label=f"state {state}",
        )
    ax.set_xticks(x + width * (state_summary.shape[0] - 1) / 2)
    ax.set_xticklabels(list(feature_cols), rotation=45, ha="right")
    ax.set_title("Feature Means by State")
    ax.set_ylabel("Mean")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_transmat_heatmap(
    out_path: Path,
    *,
    transmat: np.ndarray,
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(transmat, cmap="viridis")
    ax.set_title("Transition Matrix")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for i in range(transmat.shape[0]):
        for j in range(transmat.shape[1]):
            ax.text(j, i, f"{transmat[i, j]:.2f}", ha="center", va="center", color="white")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _state_segments(states: np.ndarray) -> list[tuple[int, int, int]]:
    if states.size == 0:
        return []
    segments: list[tuple[int, int, int]] = []
    start = 0
    current = int(states[0])
    for i in range(1, len(states)):
        s = int(states[i])
        if s != current:
            segments.append((start, i, current))
            start = i
            current = s
    segments.append((start, len(states), current))
    return segments


def _state_colors(n: int) -> list[str]:
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]


def _coerce_dates(dates: pd.Series | pd.DatetimeIndex | Iterable) -> pd.Series:
    if isinstance(dates, pd.Series):
        return dates
    if isinstance(dates, pd.DatetimeIndex):
        return dates.to_series(index=dates)
    return pd.to_datetime(pd.Series(dates), utc=True)
