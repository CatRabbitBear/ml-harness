from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .run_result import RunResult


@dataclass(frozen=True, slots=True)
class SweepVariantFailure:
    variant_id: str
    overrides: Mapping[str, Any]
    error: str


@dataclass(frozen=True, slots=True)
class SweepResult:
    sweep_id: str
    started_at_utc: str
    ended_at_utc: str
    duration_s: float
    results: Sequence[RunResult] = field(default_factory=list)
    failures: Sequence[SweepVariantFailure] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)
