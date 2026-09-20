"""Portable dataset/file manifest helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _scan(root: Path, extensions: set[str], kind: str) -> list[dict[str, object]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions:
            rows.append(
                {
                    "file_name": path.name,
                    "relative_path": str(path.relative_to(root)),
                    "extension": path.suffix.lower(),
                    "file_size_bytes": path.stat().st_size,
                    "kind": kind,
                }
            )
    return rows


def build_data_manifest(
    root: str | Path,
    *,
    include_videos: bool = True,
    include_images: bool = True,
) -> pd.DataFrame:
    """Build a portable manifest without copying raw media into the repository."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(root_path)

    rows = []
    if include_videos:
        rows.extend(_scan(root_path, VIDEO_EXTENSIONS, "video"))
    if include_images:
        rows.extend(_scan(root_path, IMAGE_EXTENSIONS, "image"))

    return pd.DataFrame(
        rows,
        columns=[
            "file_name",
            "relative_path",
            "extension",
            "file_size_bytes",
            "kind",
        ],
    )
