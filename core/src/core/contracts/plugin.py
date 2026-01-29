from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from core.contracts.run_result import RunResult
from core.contracts.run_spec import RunSpec


@dataclass(frozen=True, slots=True)
class PluginInfo:
    key: str
    name: str
    version: str = "0.1.0"
    description: str | None = None


@runtime_checkable
class Plugin(Protocol):
    """
    Plugin interface contract.

    Plugins are concrete implementations (HMM, RL, etc.) that core orchestrates.
    """

    @property
    def info(self) -> PluginInfo: ...

    def run(self, spec: RunSpec, *, context: Mapping[str, Any]) -> RunResult:
        """
        Execute the plugin for the given spec.

        `context` is reserved for core-provided services (mlflow backend, settings, artifact root, etc.)
        but kept untyped for v1 to avoid coupling / bikeshedding.
        """
        ...
