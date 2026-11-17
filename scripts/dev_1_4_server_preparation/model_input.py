"""Read one original model without unpacking or modifying the source dataset."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray

from scripts.dev_1_1_preprocessing.prepare_data import read_field
from scripts.project_paths import SOURCE

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ModelInput:
    dataset: str
    model_id: int
    vp: FloatArray
    vs: FloatArray
    provenance: dict[str, object]


def read_model(dataset: str, model_id: int, source: Path = SOURCE) -> ModelInput:
    """Read exactly VP/VS for a requested ID; check the established physical bounds."""
    if dataset not in ("ECVB", "EFVB") or not 0 <= model_id < 30000:
        raise ValueError("Expected ECVB/EFVB model ID 00000..29999")
    names = [f"{component}_{model_id:05d}.txt" for component in ("vp", "vs")]
    archive = source / f"data_{dataset}_out.zip"
    if archive.is_file():
        with ZipFile(archive) as stream:
            members: list[str] = []
            for name in names:
                matches = [member for member in stream.namelist() if Path(member).name == name]
                if len(matches) != 1:
                    raise ValueError(f"Expected one ZIP member {name}, got {len(matches)}")
                members.append(matches[0])
            raw = [stream.read(member) for member in members]
        origin = str(archive.resolve())
    else:
        directory = source / f"data_{dataset}_out"
        members = names
        raw = [(directory / name).read_bytes() for name in names]
        origin = str(directory.resolve())
    vp, vs = (read_field(content, name) for content, name in zip(raw, names, strict=True))
    if np.any(vp**2 <= (4.0 / 3.0) * vs**2):
        raise ValueError("Nonpositive isotropic bulk modulus")
    provenance: dict[str, object] = {"source": origin, "members": members,
        "raw_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in zip(names, raw, strict=True)},
        "array_sha256": {"vp": hashlib.sha256(vp.tobytes()).hexdigest(), "vs": hashlib.sha256(vs.tobytes()).hexdigest()},
        "axes": "row top to bottom; column left to right", "shape": [70, 70], "dtype": "float64"}
    return ModelInput(dataset, model_id, vp, vs, provenance)
