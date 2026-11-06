"""Run or materialize the controlled raster-versus-curved SEM pilot.

The default mode is intentionally a native-free dry run.  It constructs the
mesh case and validates the complete physical prescription; ``--run`` is the
only mode that imports the native libCEED solver.
"""

from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
import sys
from time import perf_counter
from typing import Sequence

import numpy as np
from numpy.polynomial.legendre import leggauss
from numpy.typing import NDArray

from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase, build_case
from semgpu2.model.receiver import SGReceiverHandler
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

MAIN_SPACE = 1
BOUNDARY_SPACE = 2
MATERIAL_ISO_PLANE_STRAIN = 11000
ABC = 32000
N_FORCE_DYNAMIC = 20200
RECEIVER_UX = 2
RECEIVER_UY = 3
RHO_KG_M3 = 1000.0
SOURCE_X_M = 350.0
SOURCE_HALF_WIDTH_M = 5.0
SOURCE_SIGMA_M = SOURCE_HALF_WIDTH_M / 3.0
RICKER_HZ = 15.0
RICKER_CENTER_S = 0.1
OUTPUT_INTERVAL_S = 0.001
SNAPSHOT_TIMES_S = (0.2, 0.5, 1.0, 2.0)


@dataclass(frozen=True, slots=True)
class PilotSpec:
    kind: str
    order: int
    substeps: int
    t_end: float
    steps_count: int
    steps_per_tact: int
    output_count: int


@dataclass(frozen=True, slots=True)
class SourceLoad:
    node_ids: IntArray
    nodal_force_y: FloatArray
    integral_before_normalization: float


def ricker(time_s: float) -> float:
    """Unit-amplitude 15 Hz Ricker wavelet centred at 0.1 s."""
    arg = np.pi * RICKER_HZ * (time_s - RICKER_CENTER_S)
    return float((1.0 - 2.0 * arg * arg) * np.exp(-arg * arg))


def _lagrange_values(nodes_x: FloatArray, sample_x: FloatArray) -> FloatArray:
    result = np.ones((sample_x.size, nodes_x.size), dtype=np.float64)
    for i, x_i in enumerate(nodes_x):
        for j, x_j in enumerate(nodes_x):
            if i != j:
                result[:, i] *= (sample_x - x_j) / (x_i - x_j)
    return result


def _boundary_nodes(case: MeshCase, boundary_id: int) -> IntArray:
    stratum = case.mesh.strata[BOUNDARY_SPACE]
    begin = int(stratum.p_node_offsets[boundary_id])
    end = int(stratum.p_node_offsets[boundary_id + 1])
    return np.asarray(stratum.p_nodes[begin:end], dtype=np.int64)


def top_boundary_ids(case: MeshCase) -> IntArray:
    """Return BOUNDARY elements lying wholly on the horizontal free surface."""
    coords = case.mesh.nodes_coords
    y_top = float(np.max(coords[:, 1]))
    tolerance = 1.0e-9 * max(1.0, abs(y_top))
    count = int(case.mesh.strata[BOUNDARY_SPACE].p_node_offsets.size - 1)
    found = [
        boundary_id
        for boundary_id in range(count)
        if np.all(np.abs(coords[_boundary_nodes(case, boundary_id), 1] - y_top) <= tolerance)
    ]
    if not found:
        raise ValueError("No horizontal top BOUNDARY elements were found")
    return np.asarray(found, dtype=np.int64)


def consistent_surface_source(case: MeshCase) -> SourceLoad:
    """Integrate ``N_i g`` on top edges and normalize the N-space load to 1 N.

    This deliberately does not sample the Gaussian only at SEM nodes: for Q2
    the nodes at 340, 350 and 360 m would otherwise not represent its 345--355
    m support.  The top boundary is required to remain straight in this pilot.
    """
    coords = case.mesh.nodes_coords
    y_top = float(np.max(coords[:, 1]))
    force_by_node: dict[int, float] = {}
    gauss_x, gauss_w = leggauss(16)
    for boundary_id in top_boundary_ids(case):
        node_ids = _boundary_nodes(case, int(boundary_id))
        edge_coords = coords[node_ids]
        if not np.allclose(edge_coords[:, 1], y_top, rtol=0.0, atol=1.0e-9):
            raise ValueError("Surface source needs a straight horizontal top boundary")
        order = np.argsort(edge_coords[:, 0], kind="stable")
        sorted_ids = node_ids[order]
        nodes_x = np.asarray(edge_coords[order, 0], dtype=np.float64)
        x_lo, x_hi = float(nodes_x[0]), float(nodes_x[-1])
        overlap_lo = max(x_lo, SOURCE_X_M - SOURCE_HALF_WIDTH_M)
        overlap_hi = min(x_hi, SOURCE_X_M + SOURCE_HALF_WIDTH_M)
        if overlap_hi <= overlap_lo:
            continue
        sample_x = (overlap_hi + overlap_lo) / 2.0 + (overlap_hi - overlap_lo) * gauss_x / 2.0
        gaussian = np.exp(-0.5 * ((sample_x - SOURCE_X_M) / SOURCE_SIGMA_M) ** 2)
        weights = (overlap_hi - overlap_lo) / 2.0 * (_lagrange_values(nodes_x, sample_x).T @ (gauss_w * gaussian))
        for node_id, weight in zip(sorted_ids, weights, strict=True):
            force_by_node[int(node_id)] = force_by_node.get(int(node_id), 0.0) + float(weight)
    if not force_by_node:
        raise ValueError("The 345--355 m Gaussian support does not overlap the top boundary")
    node_ids = np.asarray(sorted(force_by_node), dtype=np.int64)
    raw = np.asarray([force_by_node[int(node_id)] for node_id in node_ids], dtype=np.float64)
    total = float(np.sum(raw))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Consistent source integration has nonpositive total force")
    return SourceLoad(node_ids, -raw / total, total)


def plane_strain_parameters(vp: FloatArray, vs: FloatArray) -> FloatArray:
    """Convert Vp/Vs/rho to CEED ``rho, E, nu`` plane-strain parameters."""
    mu = RHO_KG_M3 * vs * vs
    lame_lambda = RHO_KG_M3 * (vp * vp - 2.0 * vs * vs)
    denominator = lame_lambda + mu
    young = mu * (3.0 * lame_lambda + 2.0 * mu) / denominator
    poisson = lame_lambda / (2.0 * denominator)
    if np.any(~np.isfinite(young)) or np.any(~np.isfinite(poisson)) or np.any(poisson <= -1.0) or np.any(poisson >= 0.5):
        raise ValueError("Vp/Vs gives invalid plane-strain Young modulus or Poisson ratio")
    return np.column_stack((np.full(vp.size, RHO_KG_M3), young, poisson))


def build_spec(kind: str, order: int, substeps: int, t_end: float) -> PilotSpec:
    if kind not in {"raster", "curved"}:
        raise ValueError("kind must be 'raster' or 'curved'")
    if order < 2:
        raise ValueError("order must be at least 2 because the geometry is frozen Q2")
    if substeps < 1:
        raise ValueError("substeps must be positive")
    if not np.isfinite(t_end) or t_end <= 0:
        raise ValueError("t_end must be finite and positive")
    output_intervals = round(t_end / OUTPUT_INTERVAL_S)
    if not np.isclose(t_end, output_intervals * OUTPUT_INTERVAL_S, rtol=0.0, atol=1.0e-12):
        raise ValueError("t_end must be an integer number of 1 ms output intervals")
    return PilotSpec(kind, order, substeps, t_end, output_intervals * substeps, substeps, output_intervals + 1)


def _case_metadata(case: MeshCase, spec: PilotSpec, source: SourceLoad) -> dict[str, object]:
    return {
        "pilot": asdict(spec),
        "coordinate_convention": "x right; y upward; raster row 0 maps to y=700 m; negative Fy is downward",
        "solver_intent": {"id": "lin_elast_2d_tenzor_newmark_kim_ceed", "ceed_resource": "/cpu/self/gen"},
        "material": {"law": "plane_strain", "rho_kg_m3": RHO_KG_M3},
        "source": {"kind": "N_FORCE_DYNAMIC", "total_force_N_per_unit_thickness": -1.0, "x_m": SOURCE_X_M, "support_m": [345.0, 355.0], "sigma_m": SOURCE_SIGMA_M, "ricker_hz": RICKER_HZ, "ricker_center_s": RICKER_CENTER_S, "consistent_integral_before_normalization_m": source.integral_before_normalization, "node_count": int(source.node_ids.size)},
        "output": {"canonical_times_s": [0.0, spec.t_end], "canonical_count": spec.output_count, "compatibility_slice": "surface[:2000] represents t=0.000,...,1.999 s when t_end=2 s"},
        "mesh": case.metadata,
    }


def dynamic_source_values(source: SourceLoad, spec: PilotSpec) -> FloatArray:
    """Materialize the time-interpolated N-space force values without CEED."""
    times = np.linspace(0.0, spec.t_end, spec.output_count, dtype=np.float64)
    values = np.zeros((spec.output_count, source.node_ids.size, 2), dtype=np.float64)
    values[:, :, 1] = (
        np.asarray([ricker(time_s) for time_s in times], dtype=np.float64)[:, None]
        * source.nodal_force_y[None, :]
    )
    return values


def dry_run(case: MeshCase, spec: PilotSpec) -> dict[str, object]:
    """Validate all portable ingredients without importing the native solver."""
    source = consistent_surface_source(case)
    material = plane_strain_parameters(case.element_vp, case.element_vs)
    boundary_count = int(case.mesh.strata[BOUNDARY_SPACE].p_node_offsets.size - 1)
    if case.boundary_vp.shape != (boundary_count,) or case.boundary_vs.shape != (boundary_count,):
        raise ValueError("MeshCase boundary Vp/Vs must align with BOUNDARY element ids")
    if material.shape[0] != int(case.mesh.strata[MAIN_SPACE].p_node_offsets.size - 1):
        raise ValueError("MeshCase element Vp/Vs must align with MAIN element ids")
    source_values = dynamic_source_values(source, spec)
    if source_values.shape != (spec.output_count, source.node_ids.size, 2):
        raise AssertionError("Dynamic N-force shape is inconsistent with the requested time grid")
    if not np.isclose(float(np.sum(source.nodal_force_y)), -1.0, rtol=0.0, atol=1.0e-14):
        raise AssertionError("The spatially consistent source is not normalized to -1 N")
    return _case_metadata(case, spec, source)


def _make_handler(store: list[tuple[float, FloatArray]], *, progress: bool = False) -> SGReceiverHandler:
    def handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        current_time = float(getattr(event, "current_time"))
        if not np.isfinite(frame).all():
            raise FloatingPointError(f"Nonfinite receiver frame at t={current_time:g} s")
        store.append((current_time, np.asarray(frame, dtype=np.float64).copy()))
        if progress and round(current_time / OUTPUT_INTERVAL_S) % 100 == 0:
            print(json.dumps({"stage": "receiver", "time_s": current_time}), flush=True)
    return handler


def loaded_windows_ceed_libraries() -> list[Path]:
    """Resolve the actual loaded CEED DLL for per-run provenance."""
    if sys.platform != "win32":
        return []
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    kernel.GetModuleHandleW.restype = wintypes.HMODULE
    kernel.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    kernel.GetModuleFileNameW.restype = wintypes.DWORD
    libraries = []
    for name in ("ceed.dll", "libceed.dll"):
        handle = kernel.GetModuleHandleW(name)
        if handle:
            buffer = ctypes.create_unicode_buffer(32768)
            length = kernel.GetModuleFileNameW(handle, buffer, len(buffer))
            if not length or length >= len(buffer):
                raise OSError(f"Cannot resolve loaded {name} path")
            libraries.append(Path(buffer.value).resolve())
    if not libraries:
        raise RuntimeError("Native solver imported but no CEED DLL was found in this process")
    return libraries


def run_case(case: MeshCase, spec: PilotSpec, output: Path) -> dict[str, object]:
    """Import CEED lazily, run the experiment, and save receiver products."""
    from semgpu2.model.model import SGModel
    from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
    from semgpu2.model.receiver import SGReceiverModel
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed import _native
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed.constants import SGSolverConfig, SGRunCodeEnum, SGRunResultData
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed.solver import SGSolver

    source = consistent_surface_source(case)
    top_ids = top_boundary_ids(case)
    boundary_count = int(case.mesh.strata[BOUNDARY_SPACE].p_node_offsets.size - 1)
    abc_ids = np.setdiff1d(np.arange(boundary_count, dtype=np.int64), top_ids, assume_unique=True)
    materials = plane_strain_parameters(case.element_vp, case.element_vs)
    abc_values = np.column_stack((RHO_KG_M3 * case.boundary_vp[abc_ids], RHO_KG_M3 * case.boundary_vs[abc_ids]))
    times = np.linspace(0.0, spec.t_end, spec.output_count, dtype=np.float64)
    source_values = dynamic_source_values(source, spec)
    config = SGSolverConfig(t_begin=0.0, t_end=spec.t_end, steps_count=spec.steps_count, steps_per_tact=spec.steps_per_tact, ceed_resource="/cpu/self/gen")
    model = SGModel(case.mesh, SGSolver(config))
    main_count = int(case.mesh.strata[MAIN_SPACE].p_node_offsets.size - 1)
    model.add_physics(SGPhysicsModel.from_entity_values("material", MATERIAL_ISO_PLANE_STRAIN, case.mesh, MAIN_SPACE, np.arange(main_count, dtype=np.int64), materials))
    model.add_physics(SGPhysicsModel.from_entity_values("abc_bottom_sides", ABC, case.mesh, BOUNDARY_SPACE, abc_ids, abc_values))
    model.add_physics(SGPhysicsModel("surface_ricker_consistent", N_FORCE_DYNAMIC, case.mesh, 0, source.node_ids, source_values, time=SGPhysicsTimeData(0.0, spec.t_end, spec.output_count, 3)))
    surface_frames: list[tuple[float, FloatArray]] = []
    surface_points = np.column_stack((np.arange(71, dtype=np.float64) * 10.0, np.full(71, 700.0)))
    model.add_receiver(SGReceiverModel.from_points("surface_ux_uy", case.mesh, MAIN_SPACE, surface_points, np.asarray([RECEIVER_UX, RECEIVER_UY], dtype=np.int32), begin_tact=0, end_tact=spec.output_count - 1, stride=1, handler=_make_handler(surface_frames, progress=True), frame_shape=(71,)))
    field_frames: list[tuple[float, FloatArray]] = []
    grid_x = np.arange(70, dtype=np.float64) * 10.0 + 5.0
    grid_y = 700.0 - (np.arange(70, dtype=np.float64) * 10.0 + 5.0)
    grid_points = np.stack(np.meshgrid(grid_x, grid_y), axis=-1).reshape((-1, 2))
    for snapshot_s in SNAPSHOT_TIMES_S:
        if snapshot_s > spec.t_end + 1.0e-12:
            continue
        tact = round(snapshot_s / OUTPUT_INTERVAL_S)
        model.add_receiver(SGReceiverModel.from_points(f"field_{snapshot_s:g}s", case.mesh, MAIN_SPACE, grid_points, np.asarray([RECEIVER_UX, RECEIVER_UY], dtype=np.int32), begin_tact=tact, end_tact=tact, stride=1, handler=_make_handler(field_frames), frame_shape=(70, 70)))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metadata.json").exists():
        raise FileExistsError(f"Completed output already exists: {output}")
    metadata = _case_metadata(case, spec, source)
    native_path = Path(_native.__file__).resolve()
    provenance_files = [native_path, *loaded_windows_ceed_libraries(), Path(__file__).resolve(), Path(__file__).with_name("pilot_mesh.py"), (DATA_12 / "pilot_geometry") / "selected_model.npz"]
    metadata["provenance"] = {
        "python": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "platform": sys.platform,
        "geometry_nodes_sha256": hashlib.sha256(case.mesh.nodes_coords.tobytes()).hexdigest(),
        "files": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in provenance_files],
    }
    (output / "metadata.started.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    started = perf_counter()
    model.prepare()
    print(json.dumps({"stage": "prepared", "kind": spec.kind, "p": spec.order, "steps": spec.steps_count}), flush=True)
    result = model.run(sync=True)
    elapsed = perf_counter() - started
    if not isinstance(result, SGRunResultData) or result.code != SGRunCodeEnum.TIME_END or result.steps_count != spec.steps_count:
        raise RuntimeError(f"Incomplete run: {result}")
    surface_frames.sort(key=lambda item: item[0])
    field_frames.sort(key=lambda item: item[0])
    surface_time = np.asarray([item[0] for item in surface_frames], dtype=np.float64)
    surface = np.asarray([item[1].reshape((2, 71)) for item in surface_frames], dtype=np.float64)
    field_time = np.asarray([item[0] for item in field_frames], dtype=np.float64)
    field = np.asarray([item[1].reshape((2, 70, 70)) for item in field_frames], dtype=np.float64)
    if not np.isfinite(surface).all() or not np.isfinite(field).all():
        raise RuntimeError("Nonfinite receiver output")
    if surface_time.shape != (spec.output_count,) or not np.allclose(surface_time, times, rtol=0.0, atol=1.0e-12):
        raise RuntimeError("CEED did not emit the requested inclusive 1 ms surface time grid")
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "surface.npz", time_s=surface_time, displacement_ux_uy_m=surface, x_m=surface_points[:, 0], compatibility_2000_t0_to_1_999=surface[:2000])
    np.savez_compressed(output / "fields.npz", time_s=field_time, displacement_ux_uy_m=field, x_m=grid_x, y_m=grid_y)
    metadata["runtime"] = {"wall_seconds_prepare_and_run": elapsed, "result_code": int(result.code)}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("raster", "curved"), required=True)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--substeps", type=int, default=18, help="explicit CEED steps per 1 ms output tact")
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run", action="store_true", help="import native CEED and execute; otherwise only validate and write metadata")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    spec = build_spec(args.kind, args.order, args.substeps, args.t_end)
    case = build_case(spec.kind, spec.order)
    if args.run:
        metadata = run_case(case, spec, args.out)
    else:
        metadata = dry_run(case, spec)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "metadata.dry_run.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
