"""Verify animation fields and unchanged surface solution against a historical run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import cast

import numpy as np

from scripts.dev_1_4_server_preparation.compare_smoke import metrics
from scripts.dev_1_4_server_preparation.velocity_fields_run import (
    CoreSpec, ROOT, SurfaceSpec, Variant, default_inputs, stream_record, verify,
)
from scripts.reporting import mapping, number, read_json, sequence


def code_hashes(record: dict[str, object]) -> dict[str, object]:
    files = sequence(mapping(record["provenance"])["files"])
    return {Path(str(mapping(item)["path"])).name: mapping(item)["sha256"] for item in files}


def compare(directory: Path) -> dict[str, object]:
    preliminary = read_json(directory / "metadata.json")
    variant = preliminary.get("variant")
    if variant not in ("padded", "core") or preliminary.get("short_test") is not False:
        raise ValueError("A completed two-second padded/core animation run is required")
    spec = CoreSpec("EFVB", 0, 350., 96) if variant == "core" else SurfaceSpec("EFVB", 0, 350., 96)
    record = verify(directory, spec)
    _, _, reference_path = default_inputs(cast(Variant, variant))
    baseline = read_json(reference_path / "metadata.json")
    if baseline["status"] != "completed" or baseline["run_spec"] != record["run_spec"]:
        raise ValueError("Reference specification/status differs")
    for key in ("nodes_sha256", "source_input_sha256", "initial_state", "steps_count", "dt_s"):
        if baseline[key] != record[key]:
            raise ValueError(f"Reference input differs: {key}")
    for key in ("mesh_evidence", "stability_evidence", "native"):
        if mapping(baseline[key])["sha256"] != mapping(record[key])["sha256"]:
            raise ValueError(f"Reference file differs: {key}")
    new_code, old_code = code_hashes(record), code_hashes(baseline)
    common = sorted(new_code.keys() & old_code.keys())
    if not common or any(new_code[key] != old_code[key] for key in common):
        raise ValueError("Common physics/recorder code differs")
    if stream_record(reference_path / "surface.npz")["sha256"] != mapping(baseline["surface_file"])["sha256"]:
        raise ValueError("Historical surface checksum differs")
    with np.load(directory / "surface.npz", allow_pickle=False) as current, np.load(reference_path / "surface.npz", allow_pickle=False) as previous:
        for key in ("time_s", "coordinates_m", "node_ids"):
            if not np.array_equal(current[key], previous[key]):
                raise ValueError(f"Surface axes differ: {key}")
        difference = metrics(current["velocity_m_s"], previous["velocity_m_s"])
        exact = bool(np.array_equal(current["velocity_m_s"], previous["velocity_m_s"]))
    relative_tolerance, absolute_tolerance = 1e-10, 1e-18
    passed = number(difference["relative_l2"]) <= relative_tolerance and number(difference["max_abs_m_s"]) <= absolute_tolerance
    if not passed:
        raise ValueError(f"Field recording changed the surface solution beyond repeat tolerances: {difference}")
    return {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(), "variant": variant,
        "scope": "Repetition consistency with historical surface-only run, not independent solver validation",
        "run": str(directory.resolve()), "reference": str(reference_path),
        "field_shape": record["field_shape"], "fields_and_surface_verified": True,
        "source_force_native_mesh_stability_sha_equal": True, "common_python_sha_equal": True,
        "surface_time_coordinates_node_ids_exact": True, "surface_velocity_bitwise_equal": exact,
        "all_node_surface_values_exact": record["all_node_surface_values_exact"],
        "thresholds": {"relative_l2_max": relative_tolerance, "absolute_max_m_s": absolute_tolerance},
        "metrics": difference, "common_python_files": common,
        "inputs": [stream_record(directory / "metadata.json"), stream_record(reference_path / "metadata.json")],
        "code": stream_record(Path(__file__))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    result = compare(args.run)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
