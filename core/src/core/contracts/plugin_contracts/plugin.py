from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.contracts.run_contracts.run_context import RunContext
from core.contracts.run_contracts.run_result import RunResult
from core.contracts.run_contracts.run_spec import RunSpec


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

    def run(self, spec: RunSpec, *, context: RunContext) -> RunResult:
        """
        Execute the plugin for the given spec.

        `context` is the stable contract for core-provided services.
        """
        ...
