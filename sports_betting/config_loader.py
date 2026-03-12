from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))
