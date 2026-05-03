"""Central configuration loader.

Reads ``config.yaml`` from the project root once and exposes the parsed dict
plus typed convenience accessors.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text()) or {}


def get(section: str, default: dict | None = None) -> dict:
    return load_config().get(section, default or {})
