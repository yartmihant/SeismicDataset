"""Read-only machine smoke comparison against the chapter 1.3 CU reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json, sequence

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DEFAULT_REFERENCE = PROJECT_ROOT / "data/dev_1_3_point_source/central_cu_v1/12404/curved/p4/x350/surface.npz"
RELATIVE_L2_LIMIT = 1e-8
MAX_ABS_LIMIT_M_S = 1e-14


@dataclass(frozen=True)
class Surface:
    times: FloatArray
    coordinates: FloatArray
    node_ids: IntArray
    velocity: FloatArray


def file_evidence(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_surface(path: Path, *, reference: bool) -> Surface:
    count, receivers = (2001, 141) if reference else (61, 70)
    keys = {"time_s", "coordinates_m", "node_ids", "velocity_m_s"}
    if reference:
        keys.add("displacement_m")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != keys:
            raise ValueError(f"Unexpected surface schema: {path}")
        expected = {"time_s": (count,), "coordinates_m": (receivers, 2),
                    "velocity_m_s": (count, receivers, 2)}
        if reference:
            expected["displacement_m"] = (count, receivers, 2)
        arrays: dict[str, FloatArray] = {}
        for name, shape in expected.items():
            array = np.asarray(archive[name])
            if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
                raise ValueError(f"Wrong shape/dtype or nonfinite {name}: {path}")
            arrays[name] = np.asarray(array, dtype=np.float64)
        node_ids = np.asarray(archive["node_ids"])
        if node_ids.dtype != np.int64 or node_ids.shape != (receivers,) or np.any(node_ids < 0) or len(np.unique(node_ids)) != receivers:
            raise ValueError(f"Invalid receiver node IDs: {path}")
        times = np.linspace(0.0, 2.0 if reference else 0.06, count, dtype=np.float64)
        xs = np.arange(0.0, 705.0, 5.0) if reference else np.arange(5.0, 700.0, 10.0)
        if not np.array_equal(arrays["time_s"], times):
            raise ValueError(f"Unexpected time axis: {path}")
        if not np.allclose(arrays["coordinates_m"][:, 0], xs, atol=1e-9, rtol=0) or not np.allclose(arrays["coordinates_m"][:, 1], 0, atol=1e-9, rtol=0):
            raise ValueError(f"Unexpected receiver coordinates: {path}")
        if not np.array_equal(arrays["velocity_m_s"][0], np.zeros((receivers, 2))):
            raise ValueError(f"Nonzero initial velocity: {path}")
        return Surface(arrays["time_s"], arrays["coordinates_m"], np.asarray(node_ids, dtype=np.int64), arrays["velocity_m_s"])


def source_metadata(surface_path: Path, *, reference: bool) -> dict[str, object]:
    path = surface_path.with_name("metadata.json")
    metadata = read_json(path)
    if metadata.get("status") != "completed":
        raise ValueError(f"Source is not completed: {path}")
    spec = mapping(metadata["run_spec"])
    expected: dict[str, object] = {"model_id": 12404, "kind": "curved", "order": 4,
                                   "source_x_m": 350.0, "substeps": 64, "backend": "cu",
                                   "t_end_s": 2.0 if reference else 0.06}
    if not reference:
        expected["dataset"] = "ECVB"
    if any(spec.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Source is not the requested ECVB12404/K64 smoke/reference: {path}")
    if "native" in metadata:
        native = mapping(metadata["native"])
    else:
        files = [mapping(item) for item in sequence(mapping(metadata["provenance"])["files"])]
        matches = [item for item in files if "lin_elast_2d_tenzor_newmark_kim_cu/_native" in str(item.get("path", ""))]
        if len(matches) != 1:
            raise ValueError(f"Missing unique CU native provenance: {path}")
        native = matches[0]
    sha = native.get("sha256")
    if not isinstance(native.get("path"), str) or not isinstance(sha, str) or len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError(f"Invalid native fingerprint: {path}")
    if "surface_file" in metadata and mapping(metadata["surface_file"]).get("sha256") != file_evidence(surface_path)["sha256"]:
        raise ValueError(f"Surface SHA differs from source metadata: {path}")
    return {"metadata_file": file_evidence(path), "native": native, "run_spec": spec,
            "provenance": metadata.get("provenance"), "fixture": metadata.get("fixture")}


def metrics(new: FloatArray, reference: FloatArray) -> dict[str, object]:
    difference = new - reference
    if not np.isfinite(difference).all():
        raise ValueError("Nonfinite velocity difference")
    reference_l2 = float(np.linalg.norm(reference))
    error_l2 = float(np.linalg.norm(difference))
    if reference_l2 <= 0 or not np.isfinite([reference_l2, error_l2]).all():
        raise ValueError("Degenerate or overflowing reference/error norm")
    return {"relative_l2": error_l2 / reference_l2, "error_l2_m_s": error_l2,
            "reference_l2_m_s": reference_l2, "max_abs_m_s": float(np.max(np.abs(difference))),
            "rms_error_m_s": float(np.sqrt(np.mean(difference**2))),
            "rms_reference_m_s": float(np.sqrt(np.mean(reference**2))),
            "rms_new_m_s": float(np.sqrt(np.mean(new**2)))}


def compare(new_path: Path, reference_path: Path = DEFAULT_REFERENCE) -> dict[str, object]:
    new = load_surface(new_path, reference=False)
    reference = load_surface(reference_path, reference=True)
    target = reference.velocity[:61, 1::2, :]
    if not np.array_equal(new.times, reference.times[:61]) or not np.allclose(new.coordinates, reference.coordinates[1::2], atol=1e-9, rtol=0):
        raise ValueError("New and reference receivers/time axes do not align")
    combined = metrics(new.velocity, target)
    passed = number(combined["relative_l2"]) <= RELATIVE_L2_LIMIT and number(combined["max_abs_m_s"]) <= MAX_ABS_LIMIT_M_S
    return {"status": "passed" if passed else "failed", "utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Short machine/receiver smoke comparison only; not independent scientific solver validation",
            "reference_selection": "velocity_m_s[:61, 1::2, :]", "shape": [61, 70, 2], "finite": True,
            "times_exactly_equal": True, "coordinates_exactly_equal": bool(np.array_equal(new.coordinates, reference.coordinates[1::2])),
            "coordinates_match_atol_m": 1e-9, "node_ids_exactly_equal": bool(np.array_equal(new.node_ids, reference.node_ids[1::2])),
            "metrics": combined, "components": {name: metrics(new.velocity[:, :, axis], target[:, :, axis]) for axis, name in enumerate(("Vx", "Vy"))},
            "thresholds": {"relative_l2_max": RELATIVE_L2_LIMIT, "max_abs_m_s_max": MAX_ABS_LIMIT_M_S, "combination": "both"},
            "new": {**file_evidence(new_path), **source_metadata(new_path, reference=False)},
            "reference": {**file_evidence(reference_path), **source_metadata(reference_path, reference=True)},
            "comparison_code": file_evidence(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.out.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if output.exists():
        raise FileExistsError(output)
    result = compare(args.new, args.reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "metrics": result["metrics"], "out": str(output)}, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
