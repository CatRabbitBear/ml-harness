from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from core.contracts.plugin_contracts.plugin import Plugin, PluginInfo


class PluginNotFoundError(KeyError):
    pass


@runtime_checkable
class PluginRegistry(Protocol):
    def get(self, plugin_key: str) -> Plugin:
        """Return plugin for key or raise PluginNotFoundError."""
        ...

    def list(self) -> Iterable[PluginInfo]:
        """List available plugins (for UI / debugging)."""
        ...
