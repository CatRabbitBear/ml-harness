from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from core.contracts.run_spec import RunSpec
from core.contracts.tracking import TrackingClient


@dataclass(frozen=True, slots=True)
class RunContext:
    """
    Core-provided runtime context for plugin execution.

    Keep this stable: plugins should only depend on these fields.
    """

    run_id: str
    spec: RunSpec
    tracking: TrackingClient
    artifact_dir: Path
    logger: logging.Logger
