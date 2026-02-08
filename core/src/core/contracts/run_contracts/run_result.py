from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

RunStatus = Literal["ok", "failed", "skipped"]


@dataclass(frozen=True, slots=True)
class RunResult:
    """
    Public run outcome contract.

    Keep this stable: API callers should not depend on internals.
    """

    run_id: str
    status: RunStatus

    started_at_utc: str
    ended_at_utc: str
    duration_s: float

    # Freeform outputs: model_uri, dataset_id, artifact paths, etc.
    outputs: Mapping[str, Any] = field(default_factory=dict)

    # Human-readable message for quick debugging
    message: str | None = None
