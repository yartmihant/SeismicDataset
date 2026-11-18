"""Local p4 timing experiment: crop away the sponge, keep ABC on the core boundary."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import version
import json
from math import isfinite
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

from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
from scripts.dev_1_3_point_source.physics import ExperimentSpec, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import TimedResult, file_record
from scripts.dev_1_4_server_preparation.surface_runner import (
    SurfaceRecorder, SurfaceSpec, _check_write, _write_metadata, make_cu_model,
    receiver_nodes, verify_surface,
)
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import number, read_json


@dataclass(frozen=True, slots=True)
class CoreSpec(SurfaceSpec):
    alpha_max_s_inv: float = field(default=0., init=False)
    padding_m: float = field(default=0., init=False)
    boundary_mode: str = field(default="core_abc", init=False)
    mass_damping_enabled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        SurfaceSpec.__post_init__(self)
        if self.dataset != "EFVB" or self.model_id != 0 or self.source_x_m != 350. or self.substeps != 96:
            raise ValueError("This scoped ablation keeps EFVB0/x350/p4/K96")


def core_abc_edges(case: MeshCase) -> NDArray[np.int64]:
    boundary = case.mesh.strata[2]
    groups: dict[str, list[int]] = {name: [] for name in ("top", "left", "right", "bottom")}
    for edge in range(boundary.p_node_offsets.size - 1):
        ids = boundary.p_nodes[boundary.p_node_offsets[edge]:boundary.p_node_offsets[edge + 1]]
        xy = case.mesh.nodes_coords[ids]
        matched = [name for name, axis, value in (("top", 1, 0.), ("left", 0, 0.),
                   ("right", 0, 700.), ("bottom", 1, -700.))
                   if np.all(np.abs(xy[:, axis] - value) < 1e-8)]
        if len(matched) != 1:
            raise ValueError("Boundary is not one edge of the 700x700 m core")
        groups[matched[0]].append(edge)
    if any(len(edges) != 140 for edges in groups.values()):
        raise ValueError("Expected 140 edges on each side of the core")
    return np.asarray(sorted(groups["left"] + groups["right"] + groups["bottom"]), dtype=np.int64)


def validate_core(case: MeshCase, spec: CoreSpec, mesh: Path, stability: Path) -> NDArray[np.int64]:
    record = read_json(stability)
    if record["mesh_sha256"] != file_record(mesh)["sha256"]:
        raise ValueError("Core mesh does not match its stability certificate")
    expected = {"order": 4, "nodes": 314721, "elements": 19600, "alpha_max_s_inv": 0., "abc_edge_count": 420}
    if any(number(record[key]) != value for key, value in expected.items()):
        raise ValueError("Certificate does not describe this no-padding p4 core")
    if case.metadata["model_id"] != 0 or case.metadata["kind"] != "curved" or case.metadata["dataset"] != "EFVB":
        raise ValueError("Wrong core model identity")
    if len(case.mesh.nodes_coords) != 314721 or len(case.element_vp) != 19600:
        raise ValueError("Wrong core size")
    coords = case.mesh.nodes_coords
    if not np.isfinite(coords).all() or not np.allclose(coords.min(axis=0), [0., -700.], rtol=0, atol=1e-8) or not np.allclose(coords.max(axis=0), [700., 0.], rtol=0, atol=1e-8):
        raise ValueError("Core does not have the intended 700x700 m bounds")
    dt, minimum = number(record["dt_start_s"]), number(record["substeps_per_ms"])
    if not isfinite(dt) or dt <= 0 or minimum < 1 or minimum != int(minimum) or spec.substeps < minimum or spec.dt_s > dt * (1 + 1e-12):
        raise ValueError("K96 does not satisfy the new boundary's stability certificate")
    receiver_nodes(case)
    nodal_point_force(coords, spec.source_x_m, 1.)
    return core_abc_edges(case)


def execute(case: MeshCase, spec: CoreSpec, output: Path, metadata: dict[str, object],
            edges: NDArray[np.int64]) -> dict[str, object]:
    """Same source/recorder/native as production; no mass-damping physics is added."""
    experiment = ExperimentSpec()
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    source = nodal_point_force(coords, spec.source_x_m, experiment.force_amplitude_n_per_m)
    receivers = receiver_nodes(case)
    model, native_path, success_code = make_cu_model(case, spec)
    times = np.linspace(0., spec.t_end_s, spec.steps + 1)
    values = np.zeros((times.size, 1, 2), dtype=np.float64)
    values[:, 0, 1] = source.force_y * ricker(times)
    model.add_physics(SGPhysicsModel.from_entity_values("material", 11000, case.mesh, 1,
        np.arange(case.element_vp.size, dtype=np.int64), plane_strain_parameters(case.element_vp, case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values("outer_abc", 32000, case.mesh, 2, edges,
        np.column_stack((experiment.density_kg_m3 * case.boundary_vp[edges], experiment.density_kg_m3 * case.boundary_vs[edges]))))
    model.add_physics(SGPhysicsModel("point_ricker", 20200, case.mesh, 0,
        np.asarray([source.node_id], dtype=np.int64), values,
        time=SGPhysicsTimeData(0., spec.t_end_s, times.size, 3)))
    metadata.update(source_node_id=source.node_id, source_input_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
        source_input_first_force_y_n_per_m=float(values[0, 0, 1]), native=file_record(native_path),
        nodes_sha256=hashlib.sha256(coords.tobytes()).hexdigest(),
        physics_ids=[11000, 32000, 20200], abc_edge_count=len(edges))
    recorder = SurfaceRecorder.create(spec)

    def handler(frame: NDArray[np.float64], event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        recorder.append(frame, event)
        if (recorder.count - 1) % 500 == 0:
            print(json.dumps({"stage": "surface", "time_s": float(recorder.times[recorder.count - 1])}), flush=True)

    model.add_receiver(SGReceiverModel.from_positions("surface_velocity", case.mesh, 0, receivers,
        np.asarray([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=spec.substeps, handler=handler))
    model.prepare()
    print(json.dumps({"stage": "prepared", "nodes": len(coords), "steps": spec.steps}), flush=True)
    result = cast(TimedResult, model.run(sync=True))
    if int(result.code) != success_code or result.steps_count != spec.steps or not np.isclose(result.current_time, spec.t_end_s, atol=2e-10, rtol=0):
        raise RuntimeError(f"Core integration did not reach its end: {result}")
    recorder.finish()
    expected = recorder.velocity.nbytes + recorder.times.nbytes + receivers.nbytes + 70 * 2 * 8 + 4096
    _check_write(output, expected, None)
    partial = output / "surface.partial"
    with partial.open("xb") as stream:
        np.savez(stream, time_s=recorder.times, coordinates_m=coords[receivers], node_ids=receivers,
                 velocity_m_s=recorder.velocity)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output / "surface.npz")
    return {"runtime": {"result_code": int(result.code), "time_end_code": success_code,
        "steps_count": result.steps_count, "current_time_s": result.current_time},
        "recorded_counts": {"surface": recorder.count, "source_history": 0, "fields": 0},
        "surface_file": file_record(output / "surface.npz"), "surface_file_bytes": (output / "surface.npz").stat().st_size}


def run_core(mesh: Path, stability: Path, out: Path, t_end: float = 2.) -> dict[str, object]:
    out = out.resolve()
    out.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if out.exists():
        raise FileExistsError(out)
    spec = CoreSpec("EFVB", 0, 350., 96, t_end)
    case = build_sem(load_frozen(mesh), 4)
    with np.load(mesh, allow_pickle=False) as archive:
        case.metadata["dataset"] = str(archive["dataset"])
    edges = validate_core(case, spec, mesh, stability)
    _check_write(out.parent, 65536, None)
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "run_spec": asdict(spec),
        "experiment_spec": {**asdict(ExperimentSpec()), "padding_m": 0.},
        "boundary_mode": "core_abc", "mass_damping_disabled": True,
        "boundary_description": "free top y=0; ABC on x=0, x=700, y=-700; no exterior elements or volumetric damping",
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata,
        "mesh_evidence": file_record(mesh), "stability_evidence": file_record(stability),
        "short_test": t_end != 2., "components": ["Vx", "Vy"], "receiver_count": 70,
        "initial_state": "native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3",
        "storage": "uncompressed NPZ; float64 velocity; no displacements, source history, energy or all-node fields",
        "provenance": {"python": sys.executable, "python_version": sys.version, "numpy": np.__version__,
            "semgpu2": version("semgpu2"), "platform": platform.platform(),
            "files": [file_record(Path(__file__)), file_record(PROJECT_ROOT / "scripts/dev_1_4_server_preparation/surface_runner.py"),
                file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source/physics.py"),
                file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source/run.py")]}}
    try:
        _write_metadata(out / "metadata.started.json", metadata, None)
        metadata.update(execute(case, spec, out, metadata, edges))
        metadata.update(status="completed", wall_seconds=perf_counter() - started)
        _write_metadata(out / "metadata.json", metadata, None)
        return verify_surface(out, spec)
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter() - started,
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(out / "metadata.failed.json", metadata, None)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--t-end", type=float, default=2.)
    args = parser.parse_args()
    output = args.out.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for variable, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"),
                           ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
        cache = output.parent / "runtime_cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    tempfile.tempdir = os.environ["TMPDIR"]
    result = run_core(args.mesh, args.stability, output, args.t_end)
    print(json.dumps({"status": result["status"], "wall_seconds": result["wall_seconds"], "out": str(output)}), flush=True)


if __name__ == "__main__":
    main()
