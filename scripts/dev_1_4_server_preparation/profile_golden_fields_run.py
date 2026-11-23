"""Единственный повтор выбранного золотым сечением слоя с записью101 кадров Vx/Vy."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import tempfile
from time import perf_counter
import traceback

import numpy as np

from scripts.dev_1_3_point_source.physics import ExperimentSpec, inclusive_times
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_4_server_preparation import profile_golden_run as golden
from scripts.dev_1_4_server_preparation import profile_sweep_run as sweep
from scripts.dev_1_4_server_preparation.surface_runner import _check_write, _write_metadata
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json

ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation/profile_golden_fields_1s_3060_v1"
OUTPUT = ROOT / "run"
FACTOR = 0.5684038628158912


def selected() -> tuple[golden.Request, dict[str, object], Path, dict[str, object]]:
    manifest = golden.verify_manifest()
    result = read_json(golden.ROOT / "result.json")
    best = mapping(result["best_observed"])
    if result.get("status") != "completed" or result.get("new_evaluations") != 9 or best.get("factor") != FACTOR or best.get("index") != 9:
        raise ValueError("The selected golden result changed")
    directory = Path(str(best["directory"]))
    if directory != golden.ROOT / "runs/eval_09":
        raise ValueError("Unexpected selected directory")
    request = golden.Request(int(number(best["index"])), number(best["factor"]), number(best["lower"]), number(best["upper"]), str(best["role"]))
    metadata = golden.verify_run(directory, request, manifest)
    if read_json(directory / "validation.json").get("status") != "passed":
        raise ValueError("Selected result lacks validation")
    for key, filename in (("metadata", "metadata.json"), ("surface", "surface.npz"), ("validation", "validation.json")):
        if best[key] != file_record(directory / filename):
            raise ValueError("Selected result evidence changed")
    return request, manifest, directory, metadata


def compare_surfaces(output: Path, reference: Path) -> dict[str, object]:
    with np.load(output / "surface.npz", allow_pickle=False) as actual, np.load(reference / "surface.npz", allow_pickle=False) as target:
        if set(actual.files) != set(target.files):
            raise ValueError("Surface archive members differ")
        for key in ("time_s", "coordinates_m", "node_ids", "velocity_m_s"):
            if actual[key].dtype != target[key].dtype or not np.array_equal(actual[key], target[key]):
                raise ValueError(f"Repeat is not exactly equal to selected surface: {key}")
    return {"exact_values": True, "exact_axes_and_node_ids": True, "max_absolute_difference_m_s": 0.,
        "reference_surface": file_record(reference / "surface.npz"), "repeat_surface": file_record(output / "surface.npz"),
        "method": "np.array_equal for every saved time/receiver/node-ID/velocity value; archive container hashes may differ"}


def verify(output: Path = OUTPUT) -> dict[str, object]:
    request, manifest, directory, target = selected()
    profile = sweep.Profile(request.name, "power", 2., FACTOR)
    metadata = sweep.verify(output, profile, 1., save_fields=True)
    for key in ("profile", "run_spec", "experiment_spec", "source_input_sha256", "source_node_id", "source_prefix",
                "nodes_sha256", "initial_state", "dt_s", "steps_count", "physical_model", "physics_ids", "abc_edge_count", "nodal_alpha_sha256"):
        if metadata[key] != target[key]:
            raise ValueError(f"Full-field repeat differs from selected experiment: {key}")
    for key in ("native", "mesh_evidence", "geometry_evidence"):
        if mapping(metadata[key])["sha256"] != mapping(target[key])["sha256"]:
            raise ValueError(f"Full-field repeat changed evidence: {key}")
    if metadata.get("golden_request") != asdict(request) or metadata.get("source_prefix") != manifest["source_prefix"]:
        raise ValueError("Wrong selected request or source prefix")
    if metadata["recorded_counts"] != {"surface": 1001, "fields": 101, "source_history": 0} or metadata["field_shape"] != [101, 589541, 2]:
        raise ValueError("Unexpected all-node recording shape/count")
    if set(mapping(metadata["field_files"])) != {"velocity_m_s.npy", "nodes.npy", "field_time_s.npy"}:
        raise ValueError("Missing field evidence")
    if (output / "velocity_m_s.partial.npy").exists() or (output / "surface.partial").exists():
        raise ValueError("Unfinished output remains")
    comparison = compare_surfaces(output, directory)
    if metadata.get("surface_match") != comparison:
        raise ValueError("Surface equality evidence mismatch")
    return metadata


def run() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError("Existing full-field output is never overwritten or retried automatically")
    request, manifest, directory, _ = selected()
    profile = sweep.Profile(request.name, "power", 2., FACTOR)
    spec = sweep.spec_for(profile, 1.)
    case, edges = sweep.homogeneous.load_case("padded", spec)
    mesh, geometry, certificate = sweep.homogeneous.mesh_inputs("padded")
    _check_write(ROOT, (101+1)*len(case.mesh.nodes_coords)*16+10_000_000, None)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "variant": "padded", "padding_m": 175.,
        "profile": sweep.profile_record(profile), "material_model": sweep.homogeneous.MATERIAL_MODEL,
        "physical_model": sweep.homogeneous.PHYSICAL_MODEL, "is_original_dataset_model": False,
        "run_spec": asdict(spec), "experiment_spec": asdict(ExperimentSpec(end_s=1.)), "golden_request": asdict(request),
        "source_prefix": manifest["source_prefix"], "boundary_mode": "outer_abc", "mass_damping_disabled": False,
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata, "mesh_evidence": file_record(mesh),
        "geometry_evidence": file_record(geometry), "reference_stability_evidence": file_record(certificate),
        "profile_stability": sweep.profile_stability(case, profile), "short_test": False, "components": ["Vx", "Vy"],
        "receiver_count": 70, "save_fields": True, "field_components": ["Vx", "Vy"], "field_cadence_s": .01,
        "field_shape": [len(inclusive_times(1., .01)), len(case.mesh.nodes_coords), 2],
        "field_node_order": "all global SEM node IDs in increasing order, coordinates in nodes.npy",
        "initial_state": golden.INITIAL_STATE, "reference_surface_directory": str(directory),
        "golden_result": file_record(golden.ROOT / "result.json"),
        "storage": "float64 NPY all-node Vx/Vy;101 frames at0.01s plus1001 surface frames at0.001s",
        "timing_note": "Animation repeat with all-node I/O; not the surface-only golden timing",
        "provenance": {"python": sys.executable, "numpy": np.__version__, "semgpu2": version("semgpu2"),
            "files": [file_record(Path(__file__)), file_record(Path(__file__).with_name("check_profile_golden_fields.py")), *golden.code_records()]}}
    try:
        metadata.update(sweep.execute(case, spec, OUTPUT, metadata, edges, profile, True))
        metadata.update(status="completed", wall_seconds=perf_counter()-started, surface_match=compare_surfaces(OUTPUT, directory))
        _write_metadata(OUTPUT / "metadata.json", metadata, None)
        verified = verify()
        golden.exclusive_json(OUTPUT / "validation.json", {"status": "passed", "metadata": file_record(OUTPUT / "metadata.json"),
            "surface_match": metadata["surface_match"], "recorded_counts": metadata["recorded_counts"],
            "scope": "Finite complete101 all-node frames, exact cross-recorder values and exact repeated selected surface; fixed source/native/mesh/material/profile"})
        return verified
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter()-started, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(OUTPUT / "metadata.failed.json", metadata, None)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "verify"))
    args = parser.parse_args()
    if args.command == "run":
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        for variable, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
            path = ROOT / "runtime_cache" / name
            path.mkdir(parents=True, exist_ok=True)
            os.environ[variable] = str(path)
        tempfile.tempdir = os.environ["TMPDIR"]
        result = run()
    else:
        result = verify()
    print(json.dumps({"status": result["status"], "factor": FACTOR, "wall_seconds": result["wall_seconds"], "surface_exact": mapping(result["surface_match"])["exact_values"]}), flush=True)


if __name__ == "__main__":
    main()
