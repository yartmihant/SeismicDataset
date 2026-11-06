"""Project paths, image destinations and immutable experiment provenance."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Final

from PIL import ImageFont

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
SOURCE: Final = PROJECT_ROOT / "src"
DATA_11: Final = PROJECT_ROOT / "data" / "dev_1_1_preprocessing"
DATA_12: Final = PROJECT_ROOT / "data" / "dev_1_2_smoothing"
IMG_11: Final = PROJECT_ROOT / "img" / "dev_1_1_preprocessing"
IMG_12: Final = PROJECT_ROOT / "img" / "dev_1_2_smoothing"
BINARY: Final = Path(os.environ.get("SEISMIC_DATASET_BINARY", str(DATA_11 / "binary"))).expanduser().resolve()


def image_directory(data_directory: Path) -> Path:
    """Mirror a project data directory below img; reject outputs outside data."""
    relative = data_directory.resolve().relative_to(PROJECT_ROOT / "data")
    destination = PROJECT_ROOT / "img" / relative
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def report_font(size: int) -> ImageFont.FreeTypeFont:
    """Use the same bundled font on Linux and Windows."""
    from matplotlib import get_data_path

    path = Path(get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
    return ImageFont.truetype(str(path), size)


def resolve_provenance(original: str, expected_sha256: str) -> Path:
    """Resolve an old path by the migration map without weakening its hash check."""
    manifest = json.loads((PROJECT_ROOT / "docs/migration/file_map.json").read_text(encoding="utf-8"))
    candidates = [Path(original)]
    for record in manifest["files"]:
        if record["sha256"] == expected_sha256:
            candidates.append(PROJECT_ROOT / record["new"])
            if "original_snapshot" in record:
                candidates.append(PROJECT_ROOT / record["original_snapshot"])
    runtime = manifest["runtime_by_sha256"].get(expected_sha256)
    if runtime is not None:
        candidates.append(PROJECT_ROOT / runtime)
    for path in candidates:
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256:
            return path
    raise FileNotFoundError(f"Restore provenance file with SHA256 {expected_sha256}: {original}")
