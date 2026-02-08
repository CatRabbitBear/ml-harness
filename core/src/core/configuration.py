from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from core.contracts import RunConfig, SweepConfig

_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ConfigError(f"YAML root must be a mapping: {path}")
    return payload


def load_run_config(path: str | Path) -> RunConfig:
    payload = resolve_env_vars(load_yaml(path))
    try:
        return RunConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error("run", exc)) from exc


def load_sweep_config(path: str | Path) -> SweepConfig:
    payload = resolve_env_vars(load_yaml(path))
    try:
        return SweepConfig.model_validate(payload)
    except ValidationError as exc:
        raise ConfigError(_format_validation_error("sweep", exc)) from exc


def resolve_env_vars(payload: Any) -> Any:
    return _resolve_env_vars(payload, path="$")


def _resolve_env_vars(payload: Any, *, path: str) -> Any:
    if isinstance(payload, Mapping):
        return {
            str(key): _resolve_env_vars(value, path=f"{path}.{key}")
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _resolve_env_vars(value, path=f"{path}[{index}]") for index, value in enumerate(payload)
        ]
    if isinstance(payload, str):
        return _substitute_env(payload, path=path)
    return payload


def _substitute_env(value: str, *, path: str) -> str:
    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        env_value = os.environ.get(key)
        if env_value is None:
            raise ConfigError(f"Missing environment variable '{key}' at {path}")
        return env_value

    return _ENV_VAR_PATTERN.sub(replace, value)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], Mapping) and isinstance(value, Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def apply_dotpath_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    target = deepcopy(dict(base))
    for path, value in overrides.items():
        _set_dotpath(target, path, value)
    return target


def _set_dotpath(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if any(not part for part in parts):
        raise ConfigError(f"Invalid override path '{path}'")

    cursor: dict[str, Any] = target
    for part in parts[:-1]:
        next_value = cursor.get(part)
        if next_value is None:
            next_value = {}
            cursor[part] = next_value
        if not isinstance(next_value, dict):
            raise ConfigError(f"Override path '{path}' collides with non-mapping key '{part}'")
        cursor = next_value
    cursor[parts[-1]] = value


def expand_grid_overrides(overrides: Mapping[str, list[Any]]) -> Iterable[dict[str, Any]]:
    keys = sorted(overrides.keys())
    grids = [overrides[key] for key in keys]
    for values in itertools.product(*grids):
        yield dict(zip(keys, values, strict=True))


def stable_hash(payload: Any, *, length: int = 12) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()[:length]


def variant_id_from_overrides(overrides: Mapping[str, Any]) -> str:
    return f"v-{stable_hash(dict(overrides), length=10)}"


def coerce_mapping(payload: Any) -> dict[str, Any]:
    if payload is None:
        return {}
    if isinstance(payload, Mapping):
        return dict(payload)
    if is_dataclass(payload):
        return dict(asdict(payload))
    if hasattr(payload, "model_dump"):
        dumped = payload.model_dump()  # type: ignore[attr-defined]
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise ConfigError(f"Expected mapping-like value, got {type(payload).__name__}")


def dump_yaml(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _format_validation_error(prefix: str, exc: ValidationError) -> str:
    details: list[str] = []
    for error in exc.errors(include_url=False):
        loc = ".".join(str(part) for part in error["loc"])
        details.append(f"{prefix}.{loc}: {error['msg']}")
    return "; ".join(details)
