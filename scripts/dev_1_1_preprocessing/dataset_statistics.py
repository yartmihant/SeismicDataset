"""Recount exact Vp/Vs bounds and material counts in each dataset shard."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
from typing import Dict, Set

import numpy as np


@dataclass
class DatasetStatistics:
    models: int = 0
    vp_min: float = float("inf")
    vp_max: float = float("-inf")
    vs_min: float = float("inf")
    vs_max: float = float("-inf")
    materials_min: int = 4901
    materials_max: int = 0
    material_count_histogram: Dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelStatistics:
    model_id: int
    vp_min: float
    vp_max: float
    vs_min: float
    vs_max: float
    materials: int
    raw_sha256: str


def read_model(directory: Path, model_id: int) -> ModelStatistics:
    raws = [(directory / f"{component}_{model_id:05d}.txt").read_bytes() for component in ("vp", "vs")]
    vp, vs = (np.loadtxt(io.BytesIO(raw), dtype=np.float64) for raw in raws)
    if vp.shape != (70, 70) or vs.shape != (70, 70):
        raise ValueError(f"Invalid shape: {directory}/{model_id}")
    if not np.isfinite(vp).all() or not np.isfinite(vs).all() or (vs <= 0).any() or (vp * vp <= 4 / 3 * vs * vs).any():
        raise ValueError(f"Invalid material: {directory}/{model_id}")
    count = int(np.unique(np.column_stack((vp.ravel(), vs.ravel())), axis=0).shape[0])
    digest = hashlib.sha256(raws[0] + b"\0" + raws[1]).hexdigest()
    return ModelStatistics(model_id, float(vp.min()), float(vp.max()), float(vs.min()), float(vs.max()), count, digest)


def recount_source(source: Path, output: Path) -> None:
    groups = {name: DatasetStatistics() for name in ("EFVB", "ECVB")}
    digests: Dict[str, str] = {}
    expected = json.loads((BINARY / "manifest.json").read_text())
    for name, result in groups.items():
        directory = source / f"data_{name}_out"
        for component in ("vp", "vs"):
            observed = {path.name for path in directory.glob(f"{component}_*.txt")}
            if observed != {f"{component}_{i:05d}.txt" for i in range(30000)}:
                raise ValueError(f"Expected exactly 30000 paired fields: {directory}")
        aggregate = hashlib.sha256()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for model in pool.map(lambda model_id: read_model(directory, model_id), range(30000)):
                result.models += 1
                result.vp_min = min(result.vp_min, model.vp_min)
                result.vp_max = max(result.vp_max, model.vp_max)
                result.vs_min = min(result.vs_min, model.vs_min)
                result.vs_max = max(result.vs_max, model.vs_max)
                result.materials_min = min(result.materials_min, model.materials)
                result.materials_max = max(result.materials_max, model.materials)
                result.material_count_histogram[model.materials] = result.material_count_histogram.get(model.materials, 0) + 1
                aggregate.update(f"{model.model_id:05d}:{model.raw_sha256}\n".encode())
                if result.models % 1000 == 0:
                    print(f"{name}: {result.models}/30000", flush=True)
        for key in ("vp_min", "vp_max", "vs_min", "vs_max"):
            if getattr(result, key) != expected["class_bounds"][name][key]:
                raise ValueError(f"Historical speed bounds differ: {name} {key}")
        digests[name] = aggregate.hexdigest()
    combined = {"models": 60000,
                "vp_min": min(value.vp_min for value in groups.values()),
                "vp_max": max(value.vp_max for value in groups.values()),
                "vs_min": min(value.vs_min for value in groups.values()),
                "vs_max": max(value.vs_max for value in groups.values()),
                "materials_min": min(value.materials_min for value in groups.values()),
                "materials_max": max(value.materials_max for value in groups.values())}
    report = {"source": str(source.resolve()),
              "material_definition": "Exact float64 (Vp, Vs) pair at fixed density; count per model, not connected regions",
              "groups": {name: asdict(value) for name, value in groups.items()},
              "combined": combined, "ordered_raw_digest": digests,
              "digest_recipe": "SHA256 of ascending lines ID:SHA256(vp_raw + NUL + vs_raw) + LF",
              "validation": "All 120000 text fields: paired IDs, shape, finite positive speeds, positive bulk modulus, bounds equal historical manifest"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recount(binary: Path, output: Path) -> None:
    manifest_path = binary / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    groups = {name: DatasetStatistics() for name in ("EFVB", "ECVB")}
    ids: Dict[str, Set[int]] = {name: set() for name in groups}
    hashes: Dict[str, str] = {}
    for shard in manifest["shards"]:
        group = str(shard["dataset"])
        path = binary / shard["file"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != shard["sha256"]:
            raise ValueError(f"SHA256 mismatch: {path}")
        hashes[path.name] = digest
        result = groups[group]
        with np.load(path, allow_pickle=False) as data:
            vp, vs = data["vp"], data["vs"]
            if vp.shape != vs.shape or vp.shape[1:] != (70, 70):
                raise ValueError(f"Invalid shapes: {path}")
            if vp.dtype != np.float64 or vs.dtype != np.float64:
                raise ValueError(f"Expected exact float64 data: {path}")
            if not np.isfinite(vp).all() or not np.isfinite(vs).all():
                raise ValueError(f"Nonfinite speed: {path}")
            result.vp_min = min(result.vp_min, float(vp.min()))
            result.vp_max = max(result.vp_max, float(vp.max()))
            result.vs_min = min(result.vs_min, float(vs.min()))
            result.vs_max = max(result.vs_max, float(vs.max()))
            for model_id, p, s in zip(data["ids"], vp, vs, strict=True):
                if int(model_id) in ids[group]:
                    raise ValueError(f"Repeated model ID: {group} {model_id}")
                ids[group].add(int(model_id))
                # A material is an exact pair, regardless of connectivity.
                count = int(np.unique(np.column_stack((p.ravel(), s.ravel())), axis=0).shape[0])
                result.materials_min = min(result.materials_min, count)
                result.materials_max = max(result.materials_max, count)
                result.material_count_histogram[count] = result.material_count_histogram.get(count, 0) + 1
                result.models += 1
        print(f"{path.name}: {result.models} models", flush=True)
    for group, observed in ids.items():
        if observed != set(range(30000)):
            raise ValueError(f"Incomplete dataset: {group}")
        expected = manifest["class_bounds"][group]
        for key in ("vp_min", "vp_max", "vs_min", "vs_max"):
            if getattr(groups[group], key) != expected[key]:
                raise ValueError(f"Manifest bounds differ: {group} {key}")
    combined = {
        "models": sum(value.models for value in groups.values()),
        "vp_min": min(value.vp_min for value in groups.values()),
        "vp_max": max(value.vp_max for value in groups.values()),
        "vs_min": min(value.vs_min for value in groups.values()),
        "vs_max": max(value.vs_max for value in groups.values()),
        "materials_min": min(value.materials_min for value in groups.values()),
        "materials_max": max(value.materials_max for value in groups.values()),
    }
    report = {
        "material_definition": "Exact float64 (Vp, Vs) pair at fixed density; count per model, not connected regions",
        "groups": {name: asdict(value) for name, value in groups.items()},
        "combined": combined,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "verified_shards": hashes,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--binary", type=Path)
    inputs.add_argument("--source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.source is not None:
        recount_source(args.source, args.out)
    else:
        recount(args.binary, args.out)


if __name__ == "__main__":
    main()
