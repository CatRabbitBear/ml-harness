from __future__ import annotations

import os
import tempfile
from pathlib import Path

_ENV_ARTIFACT_ROOT = "CORE_ARTIFACT_ROOT"
_LOCAL_ARTIFACT_DIRNAME = ".artifacts"
_TEMP_ARTIFACT_DIRNAME = "ml-harness-artifacts"


def resolve_artifact_root() -> Path:
    """Resolve a writable artifact root directory and ensure it exists."""
    candidates: list[Path] = []

    env_value = os.environ.get(_ENV_ARTIFACT_ROOT)
    if env_value:
        candidates.append(Path(env_value).expanduser())

    candidates.append(Path.cwd() / _LOCAL_ARTIFACT_DIRNAME)
    candidates.append(Path(tempfile.gettempdir()) / _TEMP_ARTIFACT_DIRNAME)

    for candidate in candidates:
        if _ensure_writable_dir(candidate):
            return candidate

    raise RuntimeError("Unable to resolve a writable artifact root directory.")


def build_run_artifact_dir(run_id: str, *, artifact_root: Path | None = None) -> Path:
    """Build and create the per-run artifact directory for a run id."""
    root = artifact_root if artifact_root is not None else resolve_artifact_root()
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _ensure_writable_dir(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False

    return _validate_writable(path)


def _validate_writable(path: Path) -> bool:
    test_file = path / ".write_test"
    try:
        with test_file.open("w", encoding="utf-8") as handle:
            handle.write("ok")
        test_file.unlink(missing_ok=True)
        return True
    except OSError:
        return False
