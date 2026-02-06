from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def score_log_likelihood(model: Any, X: np.ndarray) -> float:
    return float(model.score(X))


def compute_state_series(
    model: Any,
    X: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    states = model.predict(X)
    posteriors = model.predict_proba(X)
    max_posterior = posteriors.max(axis=1)
    entropy = -np.sum(posteriors * np.log(posteriors + 1e-12), axis=1)
    return states, max_posterior, entropy


def compute_switch_metrics(states: np.ndarray) -> dict[str, float]:
    if states.size == 0:
        return {
            "n_switches": 0.0,
            "switch_rate": 0.0,
        }
    switches = np.sum(states[1:] != states[:-1])
    return {
        "n_switches": float(switches),
        "switch_rate": float(switches) / float(len(states)),
    }


def compute_duration_metrics(
    states: np.ndarray, n_components: int
) -> tuple[dict[str, float], dict[int, list[int]]]:
    durations_per_state: dict[int, list[int]] = {i: [] for i in range(n_components)}
    if states.size == 0:
        overall = {
            "mean_duration_overall": 0.0,
            "median_duration_overall": 0.0,
        }
        return overall, durations_per_state

    current_state = int(states[0])
    length = 1
    for s in states[1:]:
        s = int(s)
        if s == current_state:
            length += 1
        else:
            durations_per_state[current_state].append(length)
            current_state = s
            length = 1
    durations_per_state[current_state].append(length)

    all_durations = [d for durations in durations_per_state.values() for d in durations]
    if all_durations:
        overall = {
            "mean_duration_overall": float(np.mean(all_durations)),
            "median_duration_overall": float(np.median(all_durations)),
        }
    else:
        overall = {
            "mean_duration_overall": 0.0,
            "median_duration_overall": 0.0,
        }
    return overall, durations_per_state


def build_state_summary(
    X: pd.DataFrame,
    states: np.ndarray,
    n_components: int,
    durations_per_state: dict[int, list[int]],
) -> pd.DataFrame:
    summary_rows = []
    for i in range(n_components):
        mask = states == i
        occupancy = float(np.mean(mask)) if states.size else 0.0
        durations = durations_per_state.get(i, [])
        mean_duration = float(np.mean(durations)) if durations else float("nan")
        median_duration = float(np.median(durations)) if durations else float("nan")

        means = (
            X.loc[mask].mean(axis=0)
            if mask.any()
            else pd.Series([float("nan")] * X.shape[1], index=X.columns)
        )
        stds = (
            X.loc[mask].std(axis=0)
            if mask.any()
            else pd.Series([float("nan")] * X.shape[1], index=X.columns)
        )

        row = {
            "state": i,
            "occupancy": occupancy,
            "mean_duration": mean_duration,
            "median_duration": median_duration,
        }
        for col in X.columns:
            row[f"mean_{col}"] = float(means[col])
            row[f"std_{col}"] = float(stds[col])
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).set_index("state")
    return summary


def compute_state_metrics(
    states: np.ndarray,
    durations_per_state: dict[int, list[int]],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for state, durations in durations_per_state.items():
        occupancy = float(np.mean(states == state)) if states.size else 0.0
        metrics[f"state_{state}_occupancy"] = occupancy
        if durations:
            metrics[f"state_{state}_mean_duration"] = float(np.mean(durations))
            metrics[f"state_{state}_median_duration"] = float(np.median(durations))
        else:
            metrics[f"state_{state}_mean_duration"] = float("nan")
            metrics[f"state_{state}_median_duration"] = float("nan")
    return metrics
