"""Local EFVB0 padded/core repeat with all-node velocity every 0.01s for movies."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import traceback
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
from scripts.dev_1_3_point_source.physics import ExperimentSpec, damping_alpha, inclusive_times, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import TimedResult, abc_edges, checked_time, file_record
from scripts.dev_1_4_server_preparation.no_padding_run import CoreSpec, validate_core
from scripts.dev_1_4_server_preparation.surface_runner import (
    SurfaceRecorder, SurfaceSpec, _check_write, _write_metadata, make_cu_model,
    receiver_nodes, validate_stability,
)
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json

FloatArray = NDArray[np.float64]
Variant = Literal["padded", "core"]
ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation"


def stream_record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return {"path": str(path.resolve()), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def default_inputs(variant: Variant) -> tuple[Path, Path, Path]:
    base = ROOT / ("benchmark_3060_v1/EFVB_00000" if variant == "padded" else "no_padding_3060_v1/core")
    reference = ROOT / ("benchmark_3060_v1/EFVB_00000/x350" if variant == "padded" else "no_padding_3060_v1/x350")
    return base / "mesh_q2.npz", base / "stability.json", reference


@dataclass
class FieldRecorder:
    times: FloatArray
    values: FloatArray
    path: Path
    count: int = 0

    @classmethod
    def create(cls, output: Path, spec: SurfaceSpec, nodes: int) -> FieldRecorder:
        times = inclusive_times(spec.t_end_s, .01)
        path = output / "velocity_m_s.partial.npy"
        _check_write(output, times.size * nodes * 16 + nodes * 16 + 10_000_000, None)
        if path.exists():
            raise FileExistsError(path)
        values = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(times.size, nodes, 2))
        return cls(times, values, path)

    def append(self, frame: FloatArray, event: SGBaseReceiverEventData) -> None:
        if self.count >= len(self.times) or frame.dtype != np.float64 or frame.size != self.values.shape[1] * 2:
            raise ValueError("Unexpected all-node frame count, dtype or size")
        checked_time(frame, event, float(self.times[self.count]))
        if self.count == 0 and np.any(frame != 0.):
            raise ValueError("Initial all-node velocity must be exactly zero")
        remaining = (len(self.times) - self.count) * self.values.shape[1] * 16
        _check_write(self.path.parent, remaining + 10_000_000, None)
        self.values[self.count] = frame.reshape(2, -1).T
        self.count += 1
        if self.count % 10 == 0:
            self.flush()

    def flush(self) -> None:
        if isinstance(self.values, np.memmap):
            self.values.flush()

    def finish(self) -> None:
        if self.count != len(self.times):
            raise ValueError(f"Incomplete all-node fields: {self.count}/{len(self.times)}")
        self.flush()
        with self.path.open("rb") as stream:
            os.fsync(stream.fileno())
        self.path.replace(self.path.with_name("velocity_m_s.npy"))


def execute(case: MeshCase, spec: SurfaceSpec, variant: Variant, output: Path,
            metadata: dict[str, object], edges: NDArray[np.int64]) -> dict[str, object]:
    experiment = ExperimentSpec()
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    ids = np.arange(len(coords), dtype=np.int64)
    receivers = receiver_nodes(case)
    source = nodal_point_force(coords, 350., experiment.force_amplitude_n_per_m)
    model, native, success_code = make_cu_model(case, spec)
    source_times = np.linspace(0., spec.t_end_s, spec.steps + 1)
    forces = np.zeros((len(source_times), 1, 2), dtype=np.float64)
    forces[:, 0, 1] = source.force_y * ricker(source_times)
    model.add_physics(SGPhysicsModel.from_entity_values("material", 11000, case.mesh, 1,
        np.arange(len(case.element_vp), dtype=np.int64), plane_strain_parameters(case.element_vp, case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values("outer_abc", 32000, case.mesh, 2, edges,
        np.column_stack((experiment.density_kg_m3 * case.boundary_vp[edges], experiment.density_kg_m3 * case.boundary_vs[edges]))))
    physics_ids = [11000, 32000]
    if variant == "padded":
        model.add_physics(SGPhysicsModel("mass_damping", 30001, case.mesh, 0, ids,
            damping_alpha(coords, spec.alpha_max_s_inv)[:, None]))
        physics_ids.append(30001)
    model.add_physics(SGPhysicsModel("point_ricker", 20200, case.mesh, 0,
        np.asarray([source.node_id], dtype=np.int64), forces,
        time=SGPhysicsTimeData(0., spec.t_end_s, len(source_times), 3)))
    physics_ids.append(20200)
    metadata.update(source_node_id=source.node_id, source_input_sha256=hashlib.sha256(forces.tobytes()).hexdigest(),
        source_input_first_force_y_n_per_m=float(forces[0, 0, 1]), native=file_record(native),
        nodes_sha256=hashlib.sha256(coords.tobytes()).hexdigest(), physics_ids=physics_ids, abc_edge_count=len(edges))
    surface = SurfaceRecorder.create(spec)
    fields = FieldRecorder.create(output, spec, len(coords))
    np.save(output / "nodes.npy", coords, allow_pickle=False)
    np.save(output / "field_time_s.npy", fields.times, allow_pickle=False)

    def surface_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        surface.append(frame, event)
        if (surface.count - 1) % 500 == 0:
            print(json.dumps({"stage": "surface", "variant": variant, "time_s": float(surface.times[surface.count - 1])}), flush=True)

    def field_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        fields.append(frame, event)

    model.add_receiver(SGReceiverModel.from_positions("surface_velocity", case.mesh, 0, receivers,
        np.asarray([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=spec.substeps, handler=surface_handler))
    model.add_receiver(SGReceiverModel.from_positions("all_node_velocity", case.mesh, 0, ids,
        np.asarray([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=10 * spec.substeps, handler=field_handler))
    _write_metadata(output / "metadata.started.json", metadata, None)
    model.prepare()
    print(json.dumps({"stage": "prepared", "variant": variant, "nodes": len(coords), "steps": spec.steps}), flush=True)
    try:
        result = cast(TimedResult, model.run(sync=True))
        if int(result.code) != success_code or result.steps_count != spec.steps or not np.isclose(result.current_time, spec.t_end_s, atol=2e-10, rtol=0):
            raise RuntimeError(f"Integration did not finish: {result}")
        surface.finish()
        if fields.count != len(fields.times):
            raise ValueError("Missing field frames")
        for index in range(len(fields.times)):
            if not np.array_equal(fields.values[index, receivers], surface.velocity[10 * index]):
                raise ValueError("All-node and surface recorders disagree at shared times/nodes")
        fields.finish()
    finally:
        fields.flush()
    partial = output / "surface.partial"
    _check_write(output, surface.velocity.nbytes + 100_000, None)
    with partial.open("xb") as stream:
        np.savez(stream, time_s=surface.times, coordinates_m=coords[receivers], node_ids=receivers,
                 velocity_m_s=surface.velocity)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output / "surface.npz")
    return {"runtime": {"result_code": int(result.code), "time_end_code": success_code,
        "steps_count": result.steps_count, "current_time_s": result.current_time},
        "recorded_counts": {"surface": surface.count, "fields": fields.count, "source_history": 0},
        "surface_file": stream_record(output / "surface.npz"),
        "surface_file_bytes": (output / "surface.npz").stat().st_size,
        "field_files": {name: stream_record(output / name) for name in ("velocity_m_s.npy", "nodes.npy", "field_time_s.npy")},
        "all_node_surface_values_exact": True}


def verify(output: Path, spec: SurfaceSpec) -> dict[str, object]:
    metadata = read_json(output / "metadata.json")
    if metadata.get("status") != "completed" or metadata["run_spec"] != asdict(spec) or (output / "metadata.failed.json").exists():
        raise ValueError("Run is not completed with this specification")
    runtime = mapping(metadata["runtime"])
    if runtime["result_code"] != runtime["time_end_code"] or runtime["steps_count"] != spec.steps or not np.isclose(number(runtime["current_time_s"]), spec.t_end_s, atol=2e-10, rtol=0):
        raise ValueError("Unexpected runtime end")
    expected_times = inclusive_times(spec.t_end_s, .01)
    expected_surface_times = inclusive_times(spec.t_end_s, .001)
    if metadata["recorded_counts"] != {"surface": len(expected_surface_times), "fields": len(expected_times), "source_history": 0}:
        raise ValueError("Unexpected recorded counts")
    for name, record in mapping(metadata["field_files"]).items():
        if stream_record(output / name)["sha256"] != mapping(record)["sha256"]:
            raise ValueError("Field file hash mismatch")
    if stream_record(output / "surface.npz")["sha256"] != mapping(metadata["surface_file"])["sha256"]:
        raise ValueError("Surface file hash mismatch")
    nodes = np.load(output / "nodes.npy", mmap_mode="r", allow_pickle=False)
    fields = np.load(output / "velocity_m_s.npy", mmap_mode="r", allow_pickle=False)
    times = np.load(output / "field_time_s.npy", allow_pickle=False)
    count = int(number(mapping(metadata["mesh"])["nodes"]))
    if nodes.shape != (count, 2) or fields.shape != (len(expected_times), count, 2) or nodes.dtype != np.float64 or fields.dtype != np.float64 or not np.array_equal(times, expected_times):
        raise ValueError("Unexpected field shape/dtype/time axis")
    if not np.isfinite(nodes).all() or np.any(fields[0] != 0.):
        raise ValueError("Invalid nodes or initial field")
    with np.load(output / "surface.npz", allow_pickle=False) as surface:
        ids = surface["node_ids"]
        velocities = surface["velocity_m_s"]
        if set(surface.files) != {"time_s", "coordinates_m", "node_ids", "velocity_m_s"} or velocities.shape != (len(expected_surface_times), 70, 2) or velocities.dtype != np.float64 or not np.isfinite(velocities).all():
            raise ValueError("Invalid surface")
        if not np.array_equal(surface["time_s"], expected_surface_times) or not np.array_equal(surface["coordinates_m"], nodes[ids]):
            raise ValueError("Surface axes mismatch")
        for index, frame in enumerate(fields):
            if not np.isfinite(frame).all() or not np.array_equal(frame[ids], velocities[10 * index]):
                raise ValueError("Invalid field frame or cross-recorder mismatch")
    return metadata


def run(variant: Variant, output: Path, t_end: float = 2.) -> dict[str, object]:
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    mesh, stability, reference = default_inputs(variant)
    spec = CoreSpec("EFVB", 0, 350., 96, t_end) if variant == "core" else SurfaceSpec("EFVB", 0, 350., 96, t_end)
    inclusive_times(t_end, .01)
    case = build_sem(load_frozen(mesh), 4)
    with np.load(mesh, allow_pickle=False) as archive:
        case.metadata["dataset"] = str(archive["dataset"])
    if isinstance(spec, CoreSpec):
        edges = validate_core(case, spec, mesh, stability)
    else:
        validate_stability(case, spec, mesh, stability)
        edges = abc_edges(case)
    _check_write(output.parent, (len(inclusive_times(t_end, .01)) + 1) * len(case.mesh.nodes_coords) * 16 + 10_000_000, None)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "variant": variant, "run_spec": asdict(spec),
        "experiment_spec": {**asdict(ExperimentSpec()), "padding_m": 175. if variant == "padded" else 0.},
        "boundary_mode": "outer_abc" if variant == "padded" else "core_abc", "mass_damping_disabled": variant == "core",
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata,
        "mesh_evidence": file_record(mesh), "stability_evidence": file_record(stability),
        "short_test": t_end != 2., "components": ["Vx", "Vy"], "receiver_count": 70,
        "field_components": ["Vx", "Vy"], "field_cadence_s": .01,
        "field_shape": [len(inclusive_times(t_end, .01)), len(case.mesh.nodes_coords), 2],
        "field_node_order": "all global SEM node IDs in increasing order, coordinates in nodes.npy",
        "initial_state": "native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3",
        "storage": "float64 uncompressed NPY memmap all-node velocity plus surface NPZ; no displacement or energy",
        "timing_note": "Animation repeat with field I/O; not the surface-only timing benchmark",
        "reference_surface_directory": str(reference),
        "provenance": {"python": sys.executable, "python_version": sys.version, "numpy": np.__version__,
            "semgpu2": version("semgpu2"), "platform": platform.platform(),
            "files": [file_record(Path(__file__)), file_record(Path(__file__).with_name("surface_runner.py")),
                file_record(Path(__file__).with_name("no_padding_run.py")),
                file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source/physics.py"),
                file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source/run.py")]}}
    try:
        metadata.update(execute(case, spec, variant, output, metadata, edges))
        metadata.update(status="completed", wall_seconds=perf_counter() - started)
        _write_metadata(output / "metadata.json", metadata, None)
        return verify(output, spec)
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter() - started,
                        error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(output / "metadata.failed.json", metadata, None)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("padded", "core"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--t-end", type=float, default=2.)
    args = parser.parse_args()
    output = args.out.resolve()
    output.relative_to(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for variable, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
        cache = output.parent / "runtime_cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    tempfile.tempdir = os.environ["TMPDIR"]
    result = run(cast(Variant, args.variant), output, args.t_end)
    print(json.dumps({"status": result["status"], "variant": args.variant, "wall_seconds": result["wall_seconds"], "out": str(output)}), flush=True)


if __name__ == "__main__":
    main()
