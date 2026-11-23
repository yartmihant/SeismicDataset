"""Сравнение профилей C=alpha*M на неизменной однородной сетке со слоем175м."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from math import isfinite
import os
from pathlib import Path
import platform
import re
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
from scripts.dev_1_3_point_source.physics import ExperimentSpec, damping_alpha, inclusive_times, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import TimedResult, file_record
from scripts.dev_1_4_server_preparation import homogeneous_run as homogeneous
from scripts.dev_1_4_server_preparation import velocity_fields_run as fields
from scripts.dev_1_4_server_preparation.surface_runner import (
    ALPHA_MAX_S_INV, SurfaceRecorder, SurfaceSpec, _check_write, _write_metadata,
    make_cu_model, receiver_nodes, verify_surface,
)
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json

ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation/profile_sweep_3060_v1"
REFERENCE = homogeneous.ROOT / "padded"
PADDING_M = 175.
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
Shape = Literal["power", "smoothstep"]


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    shape: Shape
    exponent: float
    amplitude_factor: float

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]+", self.name):
            raise ValueError("Profile name must be a safe explicit identifier")
        if self.shape not in ("power", "smoothstep") or not isfinite(self.exponent) or self.exponent < 2:
            raise ValueError("Need power>=2 or quintic smoothstep")
        if not isfinite(self.amplitude_factor) or not 0 < self.amplitude_factor <= 1:
            raise ValueError("Peak must be positive and no greater than the approved reference")

    @property
    def peak(self) -> float:
        return ALPHA_MAX_S_INV * self.amplitude_factor

    @property
    def integral_normalized(self) -> float:
        return self.amplitude_factor / (self.exponent + 1) if self.shape == "power" else self.amplitude_factor / 2


# The final explicit experiment selection is frozen before any numerical run.
PROFILES = {p.name: p for p in (
    Profile("q2_half", "power", 2., .5),
    Profile("q2_quarter", "power", 2., .25),
)}
BASELINE = Profile("quadratic_reference", "power", 2., 1.)


@dataclass(frozen=True, slots=True)
class SweepSpec(SurfaceSpec):
    alpha_max_s_inv: float = field(default=ALPHA_MAX_S_INV, init=True)

    def __post_init__(self) -> None:
        SurfaceSpec.__post_init__(self)
        if not isfinite(self.alpha_max_s_inv) or not 0 < self.alpha_max_s_inv <= ALPHA_MAX_S_INV:
            raise ValueError("Invalid explicit peak damping coefficient")


def spec_for(profile: Profile, t_end: float = 2.) -> SweepSpec:
    return SweepSpec("EFVB", 0, 350., 96, t_end, alpha_max_s_inv=profile.peak)


def profile_alpha(coords: FloatArray, profile: Profile) -> FloatArray:
    """Nodal sum of side/bottom contributions; no damping inside the core."""
    damping_alpha(coords, 0.)  # Coordinate validation uses the historical175m domain.
    x = np.clip(coords[:, 0], -175., 875.)
    y = np.clip(coords[:, 1], -875., 0.)
    sx = np.maximum.reduce((-x, x - 700., np.zeros_like(x))) / PADDING_M
    sy = np.maximum(-y - 700., 0.) / PADDING_M
    if profile.shape == "power":
        shape_x, shape_y = sx**profile.exponent, sy**profile.exponent
    else:
        shape_x = sx**3 * (10 + sx * (-15 + 6 * sx))
        shape_y = sy**3 * (10 + sy * (-15 + 6 * sy))
    values = np.asarray(profile.peak * (shape_x + shape_y), dtype=np.float64)
    if not np.isfinite(values).all() or np.min(values) < 0:
        raise ValueError("Invalid nodal damping values")
    return values


def profile_record(profile: Profile) -> dict[str, object]:
    distance = np.arange(176, dtype=np.float64)
    coords = np.column_stack((-distance, np.zeros_like(distance)))
    return {**asdict(profile), "padding_m": PADDING_M, "alpha_reference_s_inv": ALPHA_MAX_S_INV,
        "alpha_peak_per_direction_s_inv": profile.peak,
        "integral_normalized": profile.integral_normalized,
        "integral_alpha_ds_m_s": ALPHA_MAX_S_INV * PADDING_M * profile.integral_normalized,
        "integral_relative_to_reference_quadratic": 3 * profile.integral_normalized,
        "formula": "alpha_ref*factor*(sx^power+sy^power)" if profile.shape == "power" else "alpha_ref*factor*(S(sx)+S(sy)); S(s)=10s^3-15s^4+6s^5",
        "convention": "C=alpha*M; side and bottom contributions summed in lower corners; free top",
        "samples_1d": {"distance_m": distance.tolist(), "alpha_s_inv": profile_alpha(coords, profile).tolist()}}


def profile_stability(case: MeshCase, profile: Profile) -> dict[str, object]:
    """Bound new damping by old bound plus the largest positive nodal increment."""
    _, _, certificate = homogeneous.mesh_inputs("padded")
    old = read_json(certificate)
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    baseline = damping_alpha(coords, ALPHA_MAX_S_INV)
    actual = profile_alpha(coords, profile)
    increment = max(0., float(np.max(actual - baseline)))
    stiffness = number(old["stiffness_bound_s_minus_2"])
    damping = number(old["damping_bound_s_minus_1"]) + increment
    dt = float(2 / (np.sqrt(damping*damping + 4*stiffness) + damping))
    if .001 / 96 > dt:
        raise ValueError("K96 is outside the conservative starting-step bound")
    return {"reference": file_record(certificate), "stiffness_bound_s_minus_2": stiffness,
        "damping_bound_s_minus_1": damping, "positive_nodal_increment_over_reference_s_inv": increment,
        "dt_start_s": dt, "minimum_substeps_per_ms": int(np.ceil(.001 / dt)), "selected_substeps_per_ms": 96,
        "nodal_alpha_sha256": hashlib.sha256(actual.tobytes()).hexdigest(),
        "nodal_alpha_min_s_inv": float(actual.min()), "nodal_alpha_max_s_inv": float(actual.max()),
        "method": "Identical mass/stiffness/ABC; D_new <= D_old + max_i(max(alpha_new_i-alpha_old_i,0)); inherited half scalar damped central-difference bound",
        "limitation": "Conservative starting-step estimate, not an independent solver spectrum or convergence test"}


def prepare() -> dict[str, object]:
    case, edges = homogeneous.load_case("padded", spec_for(BASELINE))
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    if len(coords) != 589541 or len(case.element_vp) != 36750 or len(edges) != 560:
        raise ValueError("Reference175m mesh changed")
    np.testing.assert_array_equal(profile_alpha(coords, BASELINE), damping_alpha(coords, ALPHA_MAX_S_INV))
    records = {name: {"profile": profile_record(p), "stability": profile_stability(case, p)} for name, p in PROFILES.items()}
    mesh, geometry, certificate = homogeneous.mesh_inputs("padded")
    result: dict[str, object] = {"status": "prepared", "utc": datetime.now(timezone.utc).isoformat(),
        "material_model": homogeneous.MATERIAL_MODEL, "physical_model": homogeneous.PHYSICAL_MODEL,
        "padding_m": PADDING_M, "mesh": file_record(mesh), "geometry": file_record(geometry),
        "reference_stability": file_record(certificate), "nodes": len(coords), "elements": len(case.element_vp),
        "abc_edges": len(edges), "baseline_formula_exact_at_all_nodes": True, "profiles": records,
        "source": "Same downward nodal Ricker30Hz/1N per m at x350; p4/K96", "solver_run": False,
        "code": file_record(Path(__file__))}
    ROOT.mkdir(parents=True, exist_ok=True)
    homogeneous.write_json(ROOT / "preparation.json", result)
    return result


def execute(case: MeshCase, spec: SweepSpec, output: Path, metadata: dict[str, object],
            edges: IntArray, profile: Profile, save_fields: bool) -> dict[str, object]:
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    ids = np.arange(len(coords), dtype=np.int64)
    receivers = receiver_nodes(case)
    experiment = ExperimentSpec()
    source = nodal_point_force(coords, 350., experiment.force_amplitude_n_per_m)
    model, native, success_code = make_cu_model(case, spec)
    times = np.linspace(0., spec.t_end_s, spec.steps + 1)
    forces = np.zeros((len(times), 1, 2), dtype=np.float64)
    forces[:, 0, 1] = source.force_y * ricker(times)
    model.add_physics(SGPhysicsModel.from_entity_values("material", 11000, case.mesh, 1,
        np.arange(len(case.element_vp), dtype=np.int64), plane_strain_parameters(case.element_vp, case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values("outer_abc", 32000, case.mesh, 2, edges,
        np.column_stack((experiment.density_kg_m3 * case.boundary_vp[edges], experiment.density_kg_m3 * case.boundary_vs[edges]))))
    alpha = profile_alpha(coords, profile)
    model.add_physics(SGPhysicsModel("mass_damping", 30001, case.mesh, 0, ids, alpha[:, None]))
    model.add_physics(SGPhysicsModel("point_ricker", 20200, case.mesh, 0, np.array([source.node_id], dtype=np.int64),
        forces, time=SGPhysicsTimeData(0., spec.t_end_s, len(times), 3)))
    metadata.update(source_node_id=source.node_id, source_input_sha256=hashlib.sha256(forces.tobytes()).hexdigest(),
        source_input_first_force_y_n_per_m=float(forces[0, 0, 1]), native=file_record(native),
        nodes_sha256=hashlib.sha256(coords.tobytes()).hexdigest(), physics_ids=[11000, 32000, 30001, 20200],
        abc_edge_count=len(edges), nodal_alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest())
    surface = SurfaceRecorder.create(spec)
    recorder = fields.FieldRecorder.create(output, spec, len(coords)) if save_fields else None
    if recorder is not None:
        np.save(output / "nodes.npy", coords, allow_pickle=False)
        np.save(output / "field_time_s.npy", recorder.times, allow_pickle=False)

    def surface_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        surface.append(frame, event)
        if (surface.count - 1) % 500 == 0:
            print(json.dumps({"stage": "surface", "profile": profile.name, "time_s": float(surface.times[surface.count - 1])}), flush=True)

    def field_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        if recorder is None:
            raise AssertionError("Unexpected field event")
        recorder.append(frame, event)

    model.add_receiver(SGReceiverModel.from_positions("surface_velocity", case.mesh, 0, receivers,
        np.array([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=spec.substeps, handler=surface_handler))
    if recorder is not None:
        model.add_receiver(SGReceiverModel.from_positions("all_node_velocity", case.mesh, 0, ids,
            np.array([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=10*spec.substeps, handler=field_handler))
    _write_metadata(output / "metadata.started.json", metadata, None)
    model.prepare()
    print(json.dumps({"stage": "prepared", "profile": profile.name, "nodes": len(coords), "steps": spec.steps, "fields": save_fields}), flush=True)
    try:
        result = cast(TimedResult, model.run(sync=True))
        if int(result.code) != success_code or result.steps_count != spec.steps or not np.isclose(result.current_time, spec.t_end_s, atol=2e-10, rtol=0):
            raise RuntimeError(f"Integration did not finish: {result}")
        surface.finish()
        if recorder is not None:
            for index, frame in enumerate(recorder.values):
                if not np.array_equal(frame[receivers], surface.velocity[10*index]):
                    raise ValueError("All-node and surface values disagree")
            recorder.finish()
    finally:
        if recorder is not None:
            recorder.flush()
    partial = output / "surface.partial"
    _check_write(output, surface.velocity.nbytes + 100_000, None)
    with partial.open("xb") as stream:
        np.savez(stream, time_s=surface.times, coordinates_m=coords[receivers], node_ids=receivers, velocity_m_s=surface.velocity)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output / "surface.npz")
    return {"runtime": {"result_code": int(result.code), "time_end_code": success_code,
        "steps_count": result.steps_count, "current_time_s": result.current_time},
        "recorded_counts": {"surface": surface.count, "fields": recorder.count if recorder else 0, "source_history": 0},
        "surface_file": fields.stream_record(output / "surface.npz"), "surface_file_bytes": (output / "surface.npz").stat().st_size,
        "field_files": {name: fields.stream_record(output / name) for name in ("velocity_m_s.npy", "nodes.npy", "field_time_s.npy")} if recorder else {},
        "all_node_surface_values_exact": True if recorder else None}


def verify(output: Path, profile: Profile, t_end: float = 2., save_fields: bool = False) -> dict[str, object]:
    spec = spec_for(profile, t_end)
    metadata = fields.verify(output, spec) if save_fields else verify_surface(output, spec)
    if metadata.get("profile") != profile_record(profile) or metadata.get("physical_model") != homogeneous.PHYSICAL_MODEL:
        raise ValueError("Wrong profile or material")
    case, edges = homogeneous.load_case("padded", spec)
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    actual_alpha = profile_alpha(coords, profile)
    if metadata["nodal_alpha_sha256"] != hashlib.sha256(actual_alpha.tobytes()).hexdigest():
        raise ValueError("Wrong actual nodal profile")
    if metadata["abc_edge_count"] != len(edges) or metadata["physics_ids"] != [11000, 32000, 30001, 20200]:
        raise ValueError("Wrong boundary/physics selection")
    if metadata["nodes_sha256"] != hashlib.sha256(coords.tobytes()).hexdigest():
        raise ValueError("Changed global nodes")
    with np.load(output / "surface.npz", allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["node_ids"], receiver_nodes(case))
        np.testing.assert_array_equal(archive["coordinates_m"], coords[receiver_nodes(case)])
    if t_end == 2.:
        reference = read_json(REFERENCE / "metadata.json")
        for key in ("source_input_sha256", "nodes_sha256", "initial_state", "dt_s", "steps_count"):
            if metadata[key] != reference[key]:
                raise ValueError(f"Reference mismatch: {key}")
        if mapping(metadata["native"])["sha256"] != mapping(reference["native"])["sha256"]:
            raise ValueError("Native differs from reference")
    return metadata


def run(profile: Profile, output: Path, t_end: float = 2., save_fields: bool = False) -> dict[str, object]:
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    spec = spec_for(profile, t_end)
    case, edges = homogeneous.load_case("padded", spec)
    mesh, geometry, certificate = homogeneous.mesh_inputs("padded")
    stability = profile_stability(case, profile)
    field_times = inclusive_times(t_end, .01) if save_fields else np.array([], dtype=np.float64)
    _check_write(output.parent, len(field_times)*len(case.mesh.nodes_coords)*16+10_000_000, None)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "variant": "padded", "padding_m": PADDING_M,
        "profile": profile_record(profile), "material_model": homogeneous.MATERIAL_MODEL,
        "physical_model": homogeneous.PHYSICAL_MODEL, "is_original_dataset_model": False,
        "run_spec": asdict(spec), "experiment_spec": asdict(ExperimentSpec()),
        "boundary_mode": "outer_abc", "mass_damping_disabled": False, "dt_s": spec.dt_s, "steps_count": spec.steps,
        "mesh": case.metadata, "mesh_evidence": file_record(mesh), "geometry_evidence": file_record(geometry),
        "reference_stability_evidence": file_record(certificate), "profile_stability": stability,
        "short_test": t_end != 2., "components": ["Vx", "Vy"], "receiver_count": 70,
        "save_fields": save_fields, "field_components": ["Vx", "Vy"] if save_fields else [],
        "field_cadence_s": .01 if save_fields else None,
        "field_shape": [len(field_times), len(case.mesh.nodes_coords), 2] if save_fields else None,
        "field_node_order": "all global SEM node IDs in increasing order" if save_fields else None,
        "initial_state": "native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3",
        "timing_note": "Field I/O winner repeat" if save_fields else "Surface-only profile comparison; no all-node output",
        "provenance": {"python": sys.executable, "python_version": sys.version, "numpy": np.__version__,
            "semgpu2": version("semgpu2"), "platform": platform.platform(),
            "files": [file_record(Path(__file__)), file_record(Path(homogeneous.__file__)), file_record(Path(fields.__file__)),
                file_record(Path(__file__).with_name("surface_runner.py")),
                file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source/physics.py")]}}
    try:
        metadata.update(execute(case, spec, output, metadata, edges, profile, save_fields))
        metadata.update(status="completed", wall_seconds=perf_counter()-started)
        _write_metadata(output / "metadata.json", metadata, None)
        verified = verify(output, profile, t_end, save_fields)
        homogeneous.write_json(output / "validation.json", {"status": "passed", "metadata": file_record(output / "metadata.json"),
            "profile": profile_record(profile), "recorded_counts": metadata["recorded_counts"],
            "all_node_surface_values_exact": metadata["all_node_surface_values_exact"],
            "scope": "Finite complete records, exact reference mesh/material/force/native at full2s, actual damping and boundary configuration; not independent solver validation"})
        return verified
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter()-started, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(output / "metadata.failed.json", metadata, None)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="CPU-only immutable-mesh/profile/stability evidence")
    job = sub.add_parser("run", help="Explicit native/GPU experiment")
    job.add_argument("--profile", required=True, choices=PROFILES)
    job.add_argument("--out", type=Path, required=True)
    job.add_argument("--t-end", type=float, default=2.)
    job.add_argument("--fields", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False), flush=True)
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for variable, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
        path = ROOT / "runtime_cache" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    tempfile.tempdir = os.environ["TMPDIR"]
    result = run(PROFILES[args.profile], args.out, args.t_end, args.fields)
    print(json.dumps({"status": result["status"], "profile": args.profile, "wall_seconds": result["wall_seconds"], "out": str(args.out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
