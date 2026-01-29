from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from core.contracts import Plugin, PluginInfo, PluginNotFoundError, PluginRegistry


@dataclass
class DictPluginRegistry(PluginRegistry):
    plugins: dict[str, Plugin]

    def get(self, plugin_key: str) -> Plugin:
        try:
            return self.plugins[plugin_key]
        except KeyError as e:
            raise PluginNotFoundError(plugin_key) from e

    def list(self) -> Iterable[PluginInfo]:
        return [p.info for p in self.plugins.values()]
