from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class ConfigurablePlugin(Protocol):
    """
    Optional plugin extension for centralized params defaults + validation.
    """

    def default_params(self) -> Mapping[str, Any]:
        """Return plugin-owned default params payload."""
        ...

    def validate_params(
        self, params: Mapping[str, Any], *, strict: bool = True
    ) -> Mapping[str, Any]:
        """
        Validate plugin params payload and return normalized values.
        """
        ...
