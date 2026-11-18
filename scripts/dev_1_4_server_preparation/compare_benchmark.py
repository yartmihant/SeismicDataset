"""Compare full Vx/Vy70 source runs and their host-dependent pipeline timings."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
from pathlib import Path

import numpy as np

from scripts.dev_1_4_server_preparation.compare_smoke import (
    MAX_ABS_LIMIT_M_S, RELATIVE_L2_LIMIT, FloatArray, Surface, file_evidence, metrics,
)
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json, sequence


@dataclass(frozen=True)
class BenchmarkRun:
    directory: Path
    metadata: dict[str, object]
    surface: Surface
    evidence: dict[str, object]


@dataclass(frozen=True)
class AuditPaths:
    local_mesh: Path
    reference_mesh: Path
    local_inputs: Path
    reference_inputs: Path


def audit_geometry(local_mesh: Path, reference_mesh: Path, local_metadata: dict[str, object],
                   reference_metadata: dict[str, object]) -> dict[str, object]:
    """Bind both frozen geometries and rebuilt SEM coordinates to the actual runs."""
    from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen

    for path, metadata in ((local_mesh, local_metadata), (reference_mesh, reference_metadata)):
        if file_evidence(path)["sha256"] != mapping(metadata["mesh_evidence"])["sha256"]:
            raise ValueError("Audit mesh SHA differs from the corresponding actual run")
    arrays: dict[str, object] = {}
    with np.load(local_mesh, allow_pickle=False) as left, np.load(reference_mesh, allow_pickle=False) as right:
        expected = {"nodes", "cells", "vp", "vs", "regions", "model_id", "kind", "dataset"}
        if set(left.files) != expected or set(right.files) != expected:
            raise ValueError("Unexpected frozen geometry audit schema")
        for key in sorted(expected):
            a, b = np.asarray(left[key]), np.asarray(right[key])
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError(f"Frozen array shape/dtype differs: {key}")
            equal = bool(np.array_equal(a, b))
            record: dict[str, object] = {"shape": list(a.shape), "dtype": str(a.dtype), "exactly_equal": equal}
            if key == "nodes":
                if a.dtype != np.float64 or a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() or not np.isfinite(b).all():
                    raise ValueError("Invalid frozen node coordinates")
                maximum = float(np.max(np.abs(a - b)))
                record.update(max_abs_difference_m=maximum, unequal_coordinate_values=int(np.count_nonzero(a != b)))
                if maximum > 1e-9:
                    raise ValueError("Frozen geometry exceeds the explicit 1e-9 m roundoff tolerance")
            elif not equal:
                raise ValueError(f"Frozen topology/material/identity differs: {key}")
            arrays[key] = record
    left_case, right_case = (build_sem(load_frozen(path), 4) for path in (local_mesh, reference_mesh))
    left_nodes, right_nodes = left_case.mesh.nodes_coords, right_case.mesh.nodes_coords
    for nodes, metadata in ((left_nodes, local_metadata), (right_nodes, reference_metadata)):
        if hashlib.sha256(nodes.tobytes()).hexdigest() != metadata["nodes_sha256"]:
            raise ValueError("Rebuilt SEM nodes do not reproduce the actual run's nodes SHA")
    maximum_sem = float(np.max(np.abs(left_nodes - right_nodes)))
    connectivity_equal = bool(np.array_equal(left_case.mesh.strata[1].p_nodes, right_case.mesh.strata[1].p_nodes))
    if maximum_sem > 1e-9 or not connectivity_equal:
        raise ValueError("SEM geometry/connectivity exceeds the explicit equivalence conditions")
    return {"status": "passed", "local_mesh": file_evidence(local_mesh), "reference_mesh": file_evidence(reference_mesh),
            "arrays": arrays, "coordinate_tolerance_m": 1e-9, "sem_max_abs_difference_m": maximum_sem,
            "sem_cells_exactly_equal": connectivity_equal, "rebuilt_sem_node_hashes_match_actual_runs": True,
            "local_nodes_sha256": local_metadata["nodes_sha256"], "reference_nodes_sha256": reference_metadata["nodes_sha256"],
            "mesh_builder_code": file_evidence(PROJECT_ROOT / "scripts/dev_1_3_point_source/mesh.py")}


def load_force_audit(path: Path, metadata: dict[str, object]) -> tuple[FloatArray, FloatArray, dict[str, object]]:
    record_path = path.with_suffix(".json")
    record = read_json(record_path)
    if record.get("schema") != "chapter14.benchmark_inputs.v1" or record.get("solver_native_imported") is not False:
        raise ValueError("Expected a CPU-only benchmark input audit")
    if mapping(record["output"])["sha256"] != file_evidence(path)["sha256"]:
        raise ValueError("CPU source input archive SHA changed")
    if mapping(record["mesh"])["sha256"] != mapping(metadata["mesh_evidence"])["sha256"] or record["nodes_sha256"] != metadata["nodes_sha256"]:
        raise ValueError("CPU input audit does not describe the actual run's mesh/nodes")
    spec = mapping(metadata["run_spec"])
    if any(spec.get(key) != value for key, value in mapping(record["spec"]).items()):
        raise ValueError("CPU input audit run parameters differ")
    if record["experiment_spec"] != metadata["experiment_spec"] or record["source_node_id"] != metadata["source_node_id"]:
        raise ValueError("CPU input audit physical/source parameters differ")
    samples = 2000 * int(number(spec["substeps"])) + 1
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"source_times_s", "source_values"}:
            raise ValueError("Unexpected CPU force audit arrays")
        times, values = np.asarray(archive["source_times_s"]), np.asarray(archive["source_values"])
        for name, array, shape in (("source_times_s", times, (samples,)), ("source_values", values, (samples, 1, 2))):
            saved = mapping(mapping(record["arrays"])[name])
            digest = hashlib.sha256(array.tobytes()).hexdigest()
            if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all() or saved["sha256"] != digest:
                raise ValueError(f"Invalid CPU input array/hash: {name}")
        if hashlib.sha256(values.tobytes()).hexdigest() != metadata["source_input_sha256"]:
            raise ValueError("Reconstructed force does not reproduce the actual source input SHA")
        if not np.array_equal(times, np.linspace(0., 2., samples)):
            raise ValueError("CPU audit source schedule differs from the actual source schedule")
        return np.asarray(times, dtype=np.float64), np.asarray(values, dtype=np.float64), {
            "archive": file_evidence(path), "metadata": file_evidence(record_path), "record": record}


def audit_inputs(paths: AuditPaths, local: BenchmarkRun, reference: BenchmarkRun) -> dict[str, object]:
    geometry = audit_geometry(paths.local_mesh, paths.reference_mesh, local.metadata, reference.metadata)
    left_times, left_force, left_record = load_force_audit(paths.local_inputs, local.metadata)
    right_times, right_force, right_record = load_force_audit(paths.reference_inputs, reference.metadata)
    if not np.array_equal(left_times, right_times):
        raise ValueError("Input force schedules differ between hosts")
    difference = left_force - right_force
    maximum = float(np.max(np.abs(difference)))
    norm = float(np.linalg.norm(right_force))
    relative = float(np.linalg.norm(difference)) / norm if norm > 0 else float("inf")
    if not isfinite(relative) or maximum > 1e-14 or relative > 1e-13:
        raise ValueError("Source force exceeds the explicit absolute/relative roundoff tolerance")
    return {"status": "passed", "geometry": geometry,
            "force": {"source_times_exactly_equal": True, "actual_run_force_hashes_reproduced": True,
                      "max_abs_difference_n_per_m": maximum, "relative_l2": relative,
                      "max_abs_tolerance_n_per_m": 1e-14, "relative_l2_tolerance": 1e-13,
                      "local": left_record, "reference": right_record},
            "interpretation": "Explicitly audited cross-host floating-point roundoff; differing input/geometry fingerprints are retained and are not claimed equal"}


def sha256_value(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Missing or malformed SHA256 fingerprint")
    return value


def code_fingerprints(metadata: dict[str, object]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in sequence(mapping(metadata["provenance"])["files"]):
        record = mapping(item)
        parts = Path(str(record["path"])).parts
        if "scripts" not in parts:
            continue
        relative = "/".join(parts[parts.index("scripts"):])
        if relative in result:
            raise ValueError("Duplicate source-code fingerprint")
        result[relative] = sha256_value(record["sha256"])
    required = {"scripts/dev_1_4_server_preparation/surface_runner.py",
                "scripts/dev_1_3_point_source/physics.py", "scripts/dev_1_3_point_source/run.py"}
    if not required.issubset(result):
        raise ValueError("Incomplete source-code provenance")
    return result


def smoke_timing(directory: Path, spec: dict[str, object]) -> dict[str, object] | None:
    path = directory.parent / "smoke.json"
    if not path.is_file():
        return None
    record = read_json(path)
    if record.get("status") != "completed" or any(record.get(key) != spec[key] for key in ("dataset", "model_id", "source_x_m", "t_end_s", "order")):
        raise ValueError("Adjacent smoke timing does not describe this source")
    preparation, total = (number(record[key]) for key in ("preparation_seconds", "wall_seconds"))
    if not isfinite(preparation) or not isfinite(total) or not 0 <= preparation <= total:
        raise ValueError("Invalid smoke preparation/total time")
    return {"preparation_seconds": preparation, "full_smoke_wall_seconds": total,
            "evidence": file_evidence(path), "scope": "CPU geometry, certificate and SEM construction preparation; complete smoke total includes source run and wrapper overhead"}


def load_run(directory: Path) -> BenchmarkRun:
    directory = directory.resolve()
    path = directory / "surface.npz"
    metadata = read_json(directory / "metadata.json")
    if metadata.get("status") != "completed" or (directory / "metadata.failed.json").exists() or (directory / "surface.partial").exists():
        raise ValueError(f"Incomplete/failed source: {directory}")
    spec = mapping(metadata["run_spec"])
    if any(spec.get(key) != value for key, value in {"t_end_s": 2.0, "order": 4, "kind": "curved", "backend": "cu"}.items()):
        raise ValueError("Benchmark requires a full two-second curved p4 CU source")
    substeps = number(spec["substeps"])
    if not isfinite(substeps) or substeps < 1 or substeps != int(substeps):
        raise ValueError("Invalid temporal substep count")
    runtime = mapping(metadata["runtime"])
    if number(runtime["result_code"]) != number(runtime["time_end_code"]) or number(runtime["steps_count"]) != 2000 * substeps or not np.isclose(number(runtime["current_time_s"]), 2., atol=2e-10, rtol=0):
        raise ValueError("Source did not complete its requested time/steps")
    if metadata.get("components") != ["Vx", "Vy"] or metadata.get("receiver_count") != 70 or metadata.get("recorded_counts") != {"surface": 2001, "source_history": 0, "fields": 0}:
        raise ValueError("Source did not use the Vx/Vy70-only recorder")
    if not isfinite(number(metadata["wall_seconds"])) or number(metadata["wall_seconds"]) <= 0:
        raise ValueError("Source has no positive finite pipeline time")
    native = mapping(metadata["native"])
    sha256_value(native["sha256"])
    for key in ("nodes_sha256", "source_input_sha256"):
        sha256_value(metadata[key])
    sha256_value(mapping(metadata["mesh_evidence"])["sha256"])
    surface_evidence = file_evidence(path)
    if sha256_value(mapping(metadata["surface_file"])["sha256"]) != surface_evidence["sha256"]:
        raise ValueError(f"Surface SHA differs from completed metadata: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"time_s", "coordinates_m", "node_ids", "velocity_m_s"}:
            raise ValueError("Unexpected surface arrays")
        times, coordinates, ids, velocity = (np.asarray(archive[key]) for key in ("time_s", "coordinates_m", "node_ids", "velocity_m_s"))
        for array, shape in ((times, (2001,)), (coordinates, (70, 2)), (velocity, (2001, 70, 2))):
            if array.dtype != np.float64 or array.shape != shape or not np.isfinite(array).all():
                raise ValueError("Wrong surface dtype/shape or nonfinite values")
        if ids.dtype != np.int64 or ids.shape != (70,) or np.any(ids < 0) or len(np.unique(ids)) != 70:
            raise ValueError("Invalid receiver node IDs")
        if not np.array_equal(times, np.linspace(0., 2., 2001)) or not np.allclose(coordinates[:, 0], np.arange(5., 700., 10.), atol=1e-9, rtol=0) or not np.allclose(coordinates[:, 1], 0., atol=1e-9, rtol=0):
            raise ValueError("Unexpected time/receiver coordinates")
        if not np.array_equal(velocity[0], np.zeros((70, 2))):
            raise ValueError("Nonzero initial velocity")
        surface = Surface(np.asarray(times, dtype=np.float64), np.asarray(coordinates, dtype=np.float64),
                          np.asarray(ids, dtype=np.int64), np.asarray(velocity, dtype=np.float64))
    evidence = {"surface": surface_evidence, "metadata": file_evidence(directory / "metadata.json"),
                "native": native, "provenance": metadata["provenance"], "code_sha256": code_fingerprints(metadata),
                "mesh_evidence": metadata["mesh_evidence"], "nodes_sha256": metadata["nodes_sha256"],
                "source_input_sha256": metadata["source_input_sha256"], "source_node_id": metadata["source_node_id"],
                "fixture": metadata.get("fixture"), "smoke_timing": smoke_timing(directory, spec)}
    return BenchmarkRun(directory, metadata, surface, evidence)


def compare(local_directory: Path, reference_directory: Path, audit_paths: AuditPaths | None = None) -> dict[str, object]:
    local, reference = load_run(local_directory), load_run(reference_directory)
    for key in ("run_spec", "experiment_spec", "mesh", "source_node_id", "initial_state"):
        if local.metadata[key] != reference.metadata[key]:
            raise ValueError(f"Runs are not comparable: different {key}")
    if local.evidence["code_sha256"] != reference.evidence["code_sha256"]:
        raise ValueError("Runs use different recorded Python physics/runner code")
    mesh_equal = mapping(local.metadata["mesh_evidence"])["sha256"] == mapping(reference.metadata["mesh_evidence"])["sha256"]
    nodes_equal = local.metadata["nodes_sha256"] == reference.metadata["nodes_sha256"]
    force_equal = local.metadata["source_input_sha256"] == reference.metadata["source_input_sha256"]
    coords_equal = bool(np.array_equal(local.surface.coordinates, reference.surface.coordinates))
    audit = audit_inputs(audit_paths, local, reference) if audit_paths is not None else None
    if audit is None and not (mesh_equal and nodes_equal and force_equal and coords_equal):
        raise ValueError("Input/geometry fingerprints differ; provide all four explicit mesh/input audit paths to test roundoff equivalence")
    if not np.array_equal(local.surface.times, reference.surface.times) or not np.array_equal(local.surface.node_ids, reference.surface.node_ids):
        raise ValueError("Runs have different receiver times/node IDs")
    coordinate_difference = float(np.max(np.abs(local.surface.coordinates - reference.surface.coordinates)))
    if coordinate_difference > (1e-9 if audit is not None else 0.):
        raise ValueError("Receiver coordinates differ beyond the selected equivalence mode")
    combined = metrics(local.surface.velocity, reference.surface.velocity)
    passed = number(combined["relative_l2"]) <= RELATIVE_L2_LIMIT and number(combined["max_abs_m_s"]) <= MAX_ABS_LIMIT_M_S
    local_seconds, reference_seconds = (number(run.metadata["wall_seconds"]) for run in (local, reference))
    return {"schema": "chapter14.benchmark_comparison.v1", "status": "passed" if passed else "failed",
            "utc": datetime.now(timezone.utc).isoformat(), "run_spec": local.metadata["run_spec"],
            "shape": [2001, 70, 2], "finite": True,
            "equivalence_mode": "audited_roundoff" if audit is not None else "exact_hashes", "input_audit": audit,
            "equivalence": {"run_spec": True, "experiment_spec": True, "mesh_artifact_sha256": mesh_equal,
                "nodes_sha256": nodes_equal, "source_force_sha256": force_equal, "source_node_id": True, "python_code_sha256": True,
                "receiver_times_and_ids_exact": True, "receiver_coordinates_exact": coords_equal,
                "receiver_coordinate_max_difference_m": coordinate_difference,
                "native_binary_sha256_equal": mapping(local.metadata["native"])["sha256"] == mapping(reference.metadata["native"])["sha256"]},
            "metrics": combined,
            "components": {name: metrics(local.surface.velocity[:, :, axis], reference.surface.velocity[:, :, axis]) for axis, name in enumerate(("Vx", "Vy"))},
            "thresholds": {"relative_l2_max": RELATIVE_L2_LIMIT, "max_abs_m_s_max": MAX_ABS_LIMIT_M_S, "combination": "both"},
            "timing": {"local_source_wall_seconds": local_seconds, "reference_source_wall_seconds": reference_seconds,
                "reference_speedup_over_local": local_seconds / reference_seconds,
                "scope": "surface_runner wall_seconds: native construction/prepare/run, recording and surface NPZ write/hash; excludes geometry preparation and final metadata/verification overhead",
                "caveat": "One full source pipeline on different hosts, not pure GPU-kernel performance. CPU, storage, load, build, context/cache warm-up and prior sources can differ; no timing distribution or independent scientific solver validation."},
            "local": local.evidence, "reference": reference.evidence,
            "comparison_code": [file_evidence(Path(__file__)), file_evidence(Path(__file__).with_name("compare_smoke.py"))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", type=Path, required=True, help="Local x350 directory")
    parser.add_argument("--reference", type=Path, required=True, help="A100 x350 directory")
    parser.add_argument("--local-mesh", type=Path, help="Explicit audited-roundoff mode: local frozen mesh")
    parser.add_argument("--reference-mesh", type=Path, help="Explicit audited-roundoff mode: reference frozen mesh")
    parser.add_argument("--local-inputs", type=Path, help="Local benchmark_inputs NPZ with companion JSON")
    parser.add_argument("--reference-inputs", type=Path, help="Reference benchmark_inputs NPZ with companion JSON")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.out.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if output.exists():
        raise FileExistsError(output)
    options = (args.local_mesh, args.reference_mesh, args.local_inputs, args.reference_inputs)
    if any(path is not None for path in options) and not all(path is not None for path in options):
        raise ValueError("Audited-roundoff mode requires both mesh paths and both CPU input archives")
    audit_paths = AuditPaths(*options) if all(path is not None for path in options) else None
    result = compare(args.local, args.reference, audit_paths)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "metrics": result["metrics"], "timing": result["timing"], "out": str(output)}, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
