"""Однородный контроль на прежней геометрии: Vp4087, Vs2431.62036, rho1000."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import traceback
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.mesh import FrozenMesh, build_sem, load_frozen
from scripts.dev_1_3_point_source.physics import ExperimentSpec, inclusive_times
from scripts.dev_1_3_point_source.run import abc_edges, file_record
from scripts.dev_1_3_point_source.stability import estimate
from scripts.dev_1_4_server_preparation.no_padding_run import CoreSpec, validate_core
from scripts.dev_1_4_server_preparation.surface_runner import (
    ALPHA_MAX_S_INV, SurfaceSpec, _check_write, _write_metadata, validate_stability,
)
from scripts.dev_1_4_server_preparation import velocity_fields_run as fields
from scripts.dev_1_4_server_preparation.velocity_fields_run import Variant
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json

ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation/homogeneous_3060_v1"
MATERIAL_MODEL = "homogeneous_reference_toplayer"
VP_M_S = 4087.0
VS_M_S = 2431.62036
DENSITY_KG_M3 = 1000.0
PHYSICAL_MODEL: dict[str, object] = {"vp_m_s": VP_M_S, "vs_m_s": VS_M_S,
    "density_kg_m3": DENSITY_KG_M3, "homogeneous": True}


def write_json(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def spec_for(variant: Variant, t_end: float = 2.) -> SurfaceSpec:
    return CoreSpec("EFVB", 0, 350., 96, t_end) if variant == "core" else SurfaceSpec("EFVB", 0, 350., 96, t_end)


def mesh_inputs(variant: Variant) -> tuple[Path, Path, Path]:
    directory = ROOT / "meshes" / variant
    return directory / "mesh_q2.npz", directory / "geometry.json", directory / "stability.json"


def check_material(frozen: FrozenMesh) -> None:
    if not np.all(frozen.vp == VP_M_S) or not np.all(frozen.vs == VS_M_S) or not np.all(frozen.regions == 0):
        raise ValueError("Mesh is not the requested single homogeneous material")
    if ExperimentSpec().density_kg_m3 != DENSITY_KG_M3:
        raise ValueError("Shared physics density differs from the homogeneous control")


def load_case(variant: Variant, spec: SurfaceSpec) -> tuple[MeshCase, NDArray[np.int64]]:
    mesh, geometry, stability = mesh_inputs(variant)
    frozen = load_frozen(mesh)
    check_material(frozen)
    record = read_json(geometry)
    if record["material_model"] != MATERIAL_MODEL or record["physical_model"] != PHYSICAL_MODEL or record["mesh_sha256"] != file_record(mesh)["sha256"]:
        raise ValueError("Homogeneous geometry evidence does not match material or mesh")
    with np.load(mesh, allow_pickle=False) as archive:
        if str(archive["material_model"]) != MATERIAL_MODEL or float(archive["density_kg_m3"]) != DENSITY_KG_M3:
            raise ValueError("Frozen mesh lacks its explicit homogeneous material identity")
    case = build_sem(frozen, 4)
    case.metadata.update(dataset="EFVB", material_model=MATERIAL_MODEL,
        physical_model=PHYSICAL_MODEL, is_original_dataset_model=False,
        identity_note="EFVB0 is the parent geometry/API identity only; material was replaced uniformly")
    if not np.all(case.boundary_vp == VP_M_S) or not np.all(case.boundary_vs == VS_M_S):
        raise ValueError("Boundary impedances do not use the homogeneous material")
    if variant == "core":
        edges = validate_core(case, cast(CoreSpec, spec), mesh, stability)
    else:
        validate_stability(case, spec, mesh, stability)
        edges = abc_edges(case)
    return case, edges


def prepare() -> dict[str, object]:
    """Save uniform-material copies without moving any coordinate or element."""
    output = ROOT / "meshes"
    if output.exists():
        raise FileExistsError(output)
    results: dict[str, object] = {}
    for variant in ("padded", "core"):
        selected = cast(Variant, variant)
        parent_mesh, _, _ = fields.default_inputs(selected)
        parent_geometry = parent_mesh.with_name("geometry.json")
        source = load_frozen(parent_mesh)
        source_geometry = read_json(parent_geometry)
        if source_geometry["mesh_sha256"] != file_record(parent_mesh)["sha256"]:
            raise ValueError("Parent geometry evidence does not match its frozen mesh")
        source_certificate = mapping(source_geometry["certificate"])
        if number(source_certificate["det_jacobian_certified_min_m2"]) <= 0:
            raise ValueError("Parent geometry has no positive whole-cell certificate")
        frozen = FrozenMesh(source.nodes.copy(), source.cells.copy(),
            np.full_like(source.vp, VP_M_S), np.full_like(source.vs, VS_M_S),
            np.zeros_like(source.regions), source.model_id, source.kind)
        check_material(frozen)
        mesh, geometry, stability = mesh_inputs(selected)
        mesh.parent.mkdir(parents=True, exist_ok=False)
        with mesh.open("xb") as stream:
            np.savez(stream, nodes=frozen.nodes, cells=frozen.cells, vp=frozen.vp, vs=frozen.vs,
                regions=frozen.regions, model_id=frozen.model_id, kind=frozen.kind, dataset="EFVB",
                material_model=MATERIAL_MODEL, density_kg_m3=DENSITY_KG_M3,
                is_original_dataset_model=False)
        reloaded = load_frozen(mesh)
        check_material(reloaded)
        if not np.array_equal(reloaded.nodes, source.nodes) or not np.array_equal(reloaded.cells, source.cells):
            raise ValueError("Uniform-material copy changed the original geometry")
        certificate = {key: value for key, value in source_certificate.items() if key != "area_by_region_m2"}
        certificate["area_by_region_m2"] = {"0": source_certificate["area_m2"]}
        report: dict[str, object] = {"schema": "chapter14.homogeneous_geometry.v1", "variant": variant,
            "material_model": MATERIAL_MODEL, "physical_model": PHYSICAL_MODEL,
            "is_original_dataset_model": False, "geometry_parent": {"dataset": "EFVB", "model_id": 0,
                "mesh": file_record(parent_mesh), "geometry": file_record(parent_geometry)},
            "mesh_sha256": file_record(mesh)["sha256"], "certificate": certificate,
            "geometry_identity": {"coordinates_exact": True, "connectivity_exact": True,
                "whole_cell_certificate_inheritance": "Identical Q2 nodes and connectivity preserve every polynomial Jacobian; only material arrays and region labels change"},
            "material_override": {"all_elements_vp_m_s": VP_M_S, "all_elements_vs_m_s": VS_M_S,
                "all_regions": 0, "material_interface_count": 0,
                "note": "Old inter-layer mesh edges remain conforming internal edges, with identical material on both sides"},
            "padding_m": 175. if variant == "padded" else 0., "planned_p": 4,
            "planned_substeps_per_ms": 96, "solver_run": False,
            "code": file_record(Path(__file__))}
        write_json(geometry, report)
        estimate_record = estimate(mesh, 4, ALPHA_MAX_S_INV if variant == "padded" else 0.)
        estimate_record.update(dataset="EFVB", material_model=MATERIAL_MODEL, physical_model=PHYSICAL_MODEL,
            selected_substeps_per_ms=96, selection_rule="Keep K96 identical to the preceding field comparison; verify against this new uniform-material certificate")
        if number(estimate_record["substeps_per_ms"]) > 96 or .001 / 96 > number(estimate_record["dt_start_s"]):
            raise ValueError("K96 is unsafe for the homogeneous control")
        write_json(stability, estimate_record)
        case, boundary = load_case(selected, spec_for(selected))
        results[variant] = {"geometry": file_record(geometry), "stability": file_record(stability),
            "nodes": len(case.mesh.nodes_coords), "elements": len(case.element_vp), "abc_edges": len(boundary),
            "minimum_substeps_per_ms": estimate_record["substeps_per_ms"], "selected_substeps_per_ms": 96}
    final: dict[str, object] = {"status": "prepared", "utc": datetime.now(timezone.utc).isoformat(),
        "material_model": MATERIAL_MODEL, "physical_model": PHYSICAL_MODEL,
        "variants": results, "source": "Same nodal downward Ricker30Hz/1N per m at x350; p4/K96", "solver_run": False}
    write_json(output / "preparation.json", final)
    return final


def verify(output: Path, spec: SurfaceSpec) -> dict[str, object]:
    metadata = fields.verify(output, spec)
    variant = cast(Variant, metadata["variant"])
    if variant not in ("padded", "core") or metadata.get("material_model") != MATERIAL_MODEL or metadata.get("physical_model") != PHYSICAL_MODEL or metadata.get("is_original_dataset_model") is not False:
        raise ValueError("Run does not declare the requested homogeneous override")
    case, edges = load_case(variant, spec)
    expected_physics = [11000, 32000, 30001, 20200] if variant == "padded" else [11000, 32000, 20200]
    if metadata["physics_ids"] != expected_physics or metadata["abc_edge_count"] != len(edges):
        raise ValueError("Wrong ABC or mass-damping selection")
    nodes = np.load(output / "nodes.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(nodes, case.mesh.nodes_coords):
        raise ValueError("Recorded coordinates differ from the homogeneous mesh")
    return metadata


def run(variant: Variant, output: Path, t_end: float = 2.) -> dict[str, object]:
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    spec = spec_for(variant, t_end)
    times = inclusive_times(t_end, .01)
    case, edges = load_case(variant, spec)
    mesh, geometry, stability = mesh_inputs(variant)
    _check_write(output.parent, (len(times) + 1) * len(case.mesh.nodes_coords) * 16 + 10_000_000, None)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "variant": variant,
        "material_model": MATERIAL_MODEL, "physical_model": PHYSICAL_MODEL, "is_original_dataset_model": False,
        "geometry_parent": read_json(geometry)["geometry_parent"],
        "model_identity_note": "Homogeneous synthetic control. Legacy EFVB0 in run_spec identifies only the reused geometry and API; this is not the original dataset model.",
        "run_spec": asdict(spec), "experiment_spec": {**asdict(ExperimentSpec()), "padding_m": 175. if variant == "padded" else 0.},
        "boundary_mode": "outer_abc" if variant == "padded" else "core_abc", "mass_damping_disabled": variant == "core",
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata,
        "mesh_evidence": file_record(mesh), "geometry_evidence": file_record(geometry), "stability_evidence": file_record(stability),
        "short_test": t_end != 2., "components": ["Vx", "Vy"], "receiver_count": 70,
        "field_components": ["Vx", "Vy"], "field_cadence_s": .01,
        "field_shape": [len(times), len(case.mesh.nodes_coords), 2],
        "field_node_order": "all global SEM node IDs in increasing order, coordinates in nodes.npy",
        "initial_state": "native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3",
        "storage": "float64 uncompressed NPY memmap all-node velocity plus surface NPZ; no displacement or energy",
        "timing_note": "Homogeneous animation control with all-node field I/O; not the earlier surface-only benchmark",
        "provenance": {"python": sys.executable, "python_version": sys.version, "numpy": np.__version__,
            "semgpu2": version("semgpu2"), "platform": platform.platform(),
            "files": [file_record(Path(__file__)), file_record(Path(fields.__file__)),
                file_record(Path(__file__).with_name("surface_runner.py")), file_record(Path(__file__).with_name("no_padding_run.py")),
                *[file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source" / name) for name in ("physics.py", "run.py", "mesh.py", "stability.py")]]}}
    try:
        metadata.update(fields.execute(case, spec, variant, output, metadata, edges))
        metadata.update(status="completed", wall_seconds=perf_counter() - started)
        _write_metadata(output / "metadata.json", metadata, None)
        verified = verify(output, spec)
        write_json(output / "validation.json", {"status": "passed", "material_model": MATERIAL_MODEL,
            "physical_model": PHYSICAL_MODEL, "all_node_surface_values_exact": True,
            "field_shape": metadata["field_shape"], "recorded_counts": metadata["recorded_counts"],
            "metadata": file_record(output / "metadata.json"), "mesh": file_record(mesh),
            "scope": "Finite complete field/surface recording and requested uniform material/boundary configuration; not independent solver accuracy validation"})
        return verified
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter() - started,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(output / "metadata.failed.json", metadata, None)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="CPU-only exact geometry copies and new material/time-step certificates")
    job = sub.add_parser("run", help="Explicit native/GPU calculation")
    job.add_argument("--variant", choices=("padded", "core"), required=True)
    job.add_argument("--out", type=Path, required=True)
    job.add_argument("--t-end", type=float, default=2.)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False), flush=True)
        return
    output = args.out.resolve()
    output.relative_to(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for variable, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
        cache = ROOT / "runtime_cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    tempfile.tempdir = os.environ["TMPDIR"]
    result = run(cast(Variant, args.variant), output, args.t_end)
    print(json.dumps({"status": result["status"], "variant": args.variant, "material_model": MATERIAL_MODEL,
        "wall_seconds": result["wall_seconds"], "out": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
