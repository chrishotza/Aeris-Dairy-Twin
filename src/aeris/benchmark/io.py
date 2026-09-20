"""Simple benchmark I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_behavior_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={"animal_id": str, "track_id": str})


def write_projection(
    projected: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    projected.to_csv(output / "cvb_aeris_projection.csv", index=False)
    summary.to_csv(output / "cvb_aeris_projection_summary.csv", index=False)
