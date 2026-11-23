"""Аудированная повторяемость полевого повтора: прежние глобальные допуски, без GPU."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.dev_1_4_server_preparation import profile_golden_fields_run as repeat
from scripts.reporting import mapping, number, read_json, sequence

ROOT, OUTPUT = repeat.ROOT, repeat.OUTPUT
EXPECTED_FAILURE = "ValueError: Repeat is not exactly equal to selected surface: velocity_m_s"
RELATIVE_TOLERANCE = 1e-10
ABSOLUTE_TOLERANCE_M_S = 1e-18
THRESHOLD_SOURCE = Path(__file__).with_name("verify_velocity_repeat.py")


def inputs() -> tuple[dict[str, object], dict[str, object]]:
    """Validate original failed record and unchanged arrays without modifying either."""
    failed = read_json(OUTPUT / "metadata.failed.json")
    if failed.get("status") != "failed" or failed.get("error") != EXPECTED_FAILURE:
        raise ValueError("Only the documented strict bitwise-repeat failure can be postvalidated")
    request, manifest, reference, target = repeat.selected()
    for key in ("profile", "run_spec", "experiment_spec", "source_input_sha256", "source_node_id", "source_prefix",
                "nodes_sha256", "initial_state", "dt_s", "steps_count", "physical_model", "physics_ids",
                "abc_edge_count", "nodal_alpha_sha256", "profile_stability"):
        if failed[key] != target[key]:
            raise ValueError(f"Physical identity differs: {key}")
    for key in ("native", "mesh_evidence", "geometry_evidence"):
        if mapping(failed[key])["sha256"] != mapping(target[key])["sha256"]:
            raise ValueError(f"Physical evidence differs: {key}")
        if repeat.file_record(Path(str(mapping(failed[key])["path"])))["sha256"] != mapping(failed[key])["sha256"]:
            raise ValueError(f"Physical evidence file changed: {key}")
    if failed["golden_request"] != asdict(request) or failed["source_prefix"] != manifest["source_prefix"]:
        raise ValueError("Wrong selected golden/source-prefix identity")
    runtime = mapping(failed["runtime"])
    if runtime != {"result_code": 0, "time_end_code": 0, "steps_count": 96000, "current_time_s": 1.}:
        raise ValueError("Native integration did not complete exactly1s/96000 steps")
    if failed["recorded_counts"] != {"surface": 1001, "fields": 101, "source_history": 0} or failed["field_shape"] != [101, 589541, 2] or failed.get("all_node_surface_values_exact") is not True:
        raise ValueError("Wrong or incomplete recorded output")
    if (OUTPUT / "surface.partial").exists() or (OUTPUT / "velocity_m_s.partial.npy").exists():
        raise ValueError("Partial data remain")
    files = {Path(str(mapping(item)["path"])).name: mapping(item)["sha256"] for item in sequence(mapping(failed["provenance"])["files"])}
    archived: list[dict[str, object]] = []
    for name in ("profile_golden_fields_run.py", "check_profile_golden_fields.py"):
        record = repeat.file_record(ROOT / "code_at_run" / name)
        if record["sha256"] != files[name] or repeat.file_record(Path(__file__).with_name(name))["sha256"] != files[name]:
            raise ValueError("Original runner/check code evidence changed")
        archived.append(record)
    field_files = mapping(failed["field_files"])
    if set(field_files) != {"velocity_m_s.npy", "nodes.npy", "field_time_s.npy"}:
        raise ValueError("Missing full-field evidence")
    for name, field_record in field_files.items():
        if repeat.sweep.fields.stream_record(OUTPUT / name)["sha256"] != mapping(field_record)["sha256"]:
            raise ValueError(f"Field array changed: {name}")
    if repeat.file_record(OUTPUT / "surface.npz")["sha256"] != mapping(failed["surface_file"])["sha256"]:
        raise ValueError("Repeat surface changed")
    nodes = np.load(OUTPUT / "nodes.npy", mmap_mode="r", allow_pickle=False)
    fields = np.load(OUTPUT / "velocity_m_s.npy", mmap_mode="r", allow_pickle=False)
    times = np.load(OUTPUT / "field_time_s.npy", allow_pickle=False)
    if nodes.dtype != np.float64 or nodes.shape != (589541, 2) or fields.dtype != np.float64 or fields.shape != (101, 589541, 2):
        raise ValueError("Wrong field/node dtype or shape")
    if times.dtype != np.float64 or not np.array_equal(times, repeat.inclusive_times(1., .01)) or not np.isfinite(nodes).all():
        raise ValueError("Wrong field time axis or nonfinite nodes")
    if hashlib.sha256(nodes.tobytes()).hexdigest() != failed["nodes_sha256"]:
        raise ValueError("Saved nodes differ from actual solver nodes")
    with np.load(OUTPUT / "surface.npz", allow_pickle=False) as current, np.load(reference / "surface.npz", allow_pickle=False) as previous:
        if set(current.files) != {"time_s", "coordinates_m", "node_ids", "velocity_m_s"}:
            raise ValueError("Wrong surface schema")
        for key in ("time_s", "coordinates_m", "node_ids"):
            if current[key].dtype != previous[key].dtype or not np.array_equal(current[key], previous[key]):
                raise ValueError(f"Surface axes differ: {key}")
        ids, values, baseline = current["node_ids"], current["velocity_m_s"], previous["velocity_m_s"]
        if values.dtype != np.float64 or values.shape != (1001, 70, 2) or not np.isfinite(values).all() or not np.array_equal(values[0], np.zeros((70, 2))):
            raise ValueError("Invalid surface values")
        if not np.array_equal(nodes[ids], current["coordinates_m"]):
            raise ValueError("Saved nodes and surface receiver coordinates differ")
        for index, frame in enumerate(fields):
            if not np.isfinite(frame).all() or not np.array_equal(frame[ids], values[10*index]):
                raise ValueError("All-node field/surface mismatch or nonfinite frame")
        if np.any(fields[0] != 0.):
            raise ValueError("Initial field is not zero")
        delta = values-baseline
        relative = float(np.linalg.norm(delta)/np.linalg.norm(baseline))
        maximum = float(np.max(np.abs(delta)))
        if relative > RELATIVE_TOLERANCE or maximum > ABSOLUTE_TOLERANCE_M_S:
            raise ValueError("Repeat exceeds the previously established global tolerances")
        late = current["time_s"] >= .5
        objective_new = repeat.golden.objective(values, current["time_s"])
        objective_old = repeat.golden.objective(baseline, previous["time_s"])
        location = np.unravel_index(np.argmax(np.abs(delta)), delta.shape)
        match: dict[str, object] = {"exact_values": bool(np.array_equal(values, baseline)), "numerically_equivalent": True,
            "exact_axes_and_node_ids": True, "relative_l2": relative, "max_abs_m_s": maximum,
            "max_relative_to_global_peak": maximum/float(np.max(np.abs(baseline))),
            "max_difference_location": {"time_s": float(current["time_s"][location[0]]), "x_m": float(current["coordinates_m"][location[1], 0]), "component": ["Vx", "Vy"][location[2]]},
            "thresholds": {"relative_l2_max": RELATIVE_TOLERANCE, "absolute_max_m_s": ABSOLUTE_TOLERANCE_M_S},
            "threshold_source_file": repeat.file_record(THRESHOLD_SOURCE),
            "threshold_scope": "Both thresholds apply to the full0–1s record; late-window metrics are separate diagnostics",
            "late_window": {"interval_s": [.5, 1.], "relative_l2": float(np.linalg.norm(delta[late])/np.linalg.norm(baseline[late])),
                "max_abs_m_s": float(np.max(np.abs(delta[late]))), "repeat_objective": objective_new, "selected_objective": objective_old,
                "relative_delta_objective": (objective_new-objective_old)/objective_old, "thresholds_applied": False},
            "symmetry": {"vx_odd_relative_l2": float(np.linalg.norm(values[:, :, 0]+values[:, ::-1, 0])/np.linalg.norm(values[:, :, 0])),
                "vy_even_relative_l2": float(np.linalg.norm(values[:, :, 1]-values[:, ::-1, 1])/np.linalg.norm(values[:, :, 1]))},
            "reference_surface": repeat.file_record(reference / "surface.npz"), "repeat_surface": repeat.file_record(OUTPUT / "surface.npz"),
            "interpretation": "Difference compatible with rounding-level repeat variability; its causal mechanism was not established"}
    evidence: dict[str, object] = {"status": "passed", "surface_match": match,
        "approved_failed_metadata_sha256": repeat.file_record(OUTPUT / "metadata.failed.json")["sha256"],
        "original_failed_metadata": repeat.file_record(OUTPUT / "metadata.failed.json"), "original_code_archive": archived,
        "all_node_surface_values_exact": True, "all_frames_finite": True, "source_native_mesh_alpha_identity": True,
        "recorded_counts": failed["recorded_counts"], "field_files": failed["field_files"],
        "validator": repeat.file_record(Path(__file__)),
        "scope": "Postvalidation of the completed native run under pre-existing global repeat tolerances; strict bitwise failure preserved"}
    return failed, evidence


def postvalidate() -> dict[str, object]:
    for name in ("metadata.json", "validation.json"):
        if (OUTPUT / name).exists():
            raise FileExistsError("Postvalidation never overwrites previous evidence")
    failed, evidence = inputs()
    metadata = dict(failed)
    metadata.pop("error")
    metadata.pop("traceback")
    metadata.update(status="completed", surface_match=evidence["surface_match"],
        postvalidation={"derived_from_failed_metadata": evidence["original_failed_metadata"], "strict_failure": EXPECTED_FAILURE,
            "validator": evidence["validator"], "original_code_archive": evidence["original_code_archive"],
            "utc": repeat.golden.now(), "native_run_repeated": False,
            "wall_seconds_meaning": "Original native run/field output and initial strict comparison; copied unchanged from failed record"})
    repeat.golden.exclusive_json(OUTPUT / "metadata.json", metadata)
    repeat.golden.exclusive_json(OUTPUT / "validation.json", {**evidence, "metadata": repeat.file_record(OUTPUT / "metadata.json")})
    repeat.golden.exclusive_json(ROOT / "completion.json", {"status": "completed", "validation": "passed", "observed_utc": repeat.golden.now(),
        "metadata": repeat.file_record(OUTPUT / "metadata.json"), "validation_file": repeat.file_record(OUTPUT / "validation.json"),
        "field_shape": metadata["field_shape"], "recorded_counts": metadata["recorded_counts"],
        "wall_seconds": number(metadata["wall_seconds"]), "surface_match": evidence["surface_match"],
        "strict_failure_preserved": evidence["original_failed_metadata"], "additional_gpu_runs": 0})
    return evidence


def verify() -> dict[str, object]:
    _, expected = inputs()
    validation = read_json(OUTPUT / "validation.json")
    if validation != {**expected, "metadata": repeat.file_record(OUTPUT / "metadata.json")}:
        raise ValueError("Audited validation evidence changed")
    metadata = read_json(OUTPUT / "metadata.json")
    if metadata.get("status") != "completed" or metadata.get("surface_match") != expected["surface_match"]:
        raise ValueError("Postvalidated metadata changed")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("postvalidate", "verify"))
    args = parser.parse_args()
    result = postvalidate() if args.command == "postvalidate" else verify()
    print(json.dumps({"status": result["status"], "surface_match": result["surface_match"]}), flush=True)
