from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

RunMode = Literal["once", "backfill", "scheduled"]
Pipeline = Literal["train", "score"]


@dataclass(frozen=True, slots=True)
class RunSpec:
    """
    Public run request contract.

    Intent-only: no infrastructure details (no MLflow URLs, no file paths, no DB URLs).
    """

    plugin_key: str
    pipeline: Pipeline = "train"
    run_mode: RunMode = "once"

    # Data identity / data build intent
    dataset_id: str | None = None
    data_spec: Mapping[str, Any] = field(default_factory=dict)

    # Reproducibility & semantics
    seed: int | None = None
    tags: Mapping[str, str] = field(default_factory=dict)
    notes: str | None = None

    # Optional: caller can force an idempotency key (useful for schedules)
    request_id: str | None = None

    def short_name(self) -> str:
        """Human-friendly identifier for logs/UI."""
        return f"{self.plugin_key}:{self.pipeline}:{self.run_mode}"
