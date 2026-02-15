from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_bootstrap_summary(artifact_dir: Path, payload: dict[str, Any]) -> Path:
    path = artifact_dir / "reports" / "bootstrap_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    return path
