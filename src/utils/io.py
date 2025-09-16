"""Utility functions for IO and simple dataset handling."""

from __future__ import annotations

import pandas as pd
from pathlib import Path


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found")
    return pd.read_csv(p)


def ensure_dir(path: str | Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
