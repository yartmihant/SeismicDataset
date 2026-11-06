"""Windows-only research sanity comparison for 2D tenzor CEED CPU gen/ref.

This is a small dynamic elasticity probe, not a solver acceptance test.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
import sys
from time import perf_counter
from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray
from semgpu2.model.mesh import SGMeshModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl

from scripts.dev_1_2_smoothing.pilot_run import loaded_windows_ceed_libraries


FloatArray = NDArray[np.float64]
EXPECTED_NATIVE_SHA256 = "ec09ec361a5b26fa1934e99dedf525b24bdd58a01c17d6eda9f2a3496c252a81"
EXPECTED_CEED_DLL_SHA256 = "84f50925453828644871feec9a81c768e98843039a70c3e36b04524d50f4d216"
MATERIAL_ISO_PLANE_STRAIN = 11000
N_FORCE_DYNAMIC = 20200
MAIN_SPACE = 1
RECEIVER_UX = 2
RECEIVER_UY = 3


@dataclass(frozen=True, slots=True)
class ProbeResult:
    resource: str
    times_s: FloatArray
    frames: FloatArray
    wall_seconds: float
    native_module: str
    native_sha256: str
    ceed_dlls: tuple[tuple[str, str], ...]


class _DllDirectoryCookie(Protocol):
    def close(self) -> None: ...


def _activate_windows_dll_directory(directory: Path) -> _DllDirectoryCookie | None:
    if os.name != "nt":
        return None
    if not directory.is_dir():
        raise FileNotFoundError(f"libCEED DLL directory does not exist: {directory}")
    return getattr(os, "add_dll_directory")(str(directory))


def _build_mesh() -> SGMeshModel:
    from semgpu2.model import SGPETALineModel
    from semgpu2.tools.mesh.box import rectangle_quad_fast

    line = SGPETALineModel("GLL", order=2)
    mesh = rectangle_quad_fast(sizes=(2.0, 2.0), elements=(2, 2), pets=[line])
    coords = mesh.nodes_coords
    x = coords[:, 0]
    y = coords[:, 1]
    # x is affine. y = eta + .06(1-x^2)(eta+1) is Q2 and has dy/deta >= 1.
    mesh.nodes_coords[:] = np.column_stack((x, y + 0.06 * (1.0 - x * x) * (y + 1.0)))
    mesh.centers = mesh.build_centers()
    return mesh


def _run(resource: str, dll_dir: Path) -> ProbeResult:
    dll_cookie = _activate_windows_dll_directory(dll_dir)
    try:
        from semgpu2.model.model import SGModel
        from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
        from semgpu2.model.receiver import SGReceiverModel
        from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed.constants import SGSolverConfig, SGRunCodeEnum, SGRunResultData
        from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed.solver import SGSolver
        from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed import _native

        native_path = Path(_native.__file__).resolve()
        native_hash = hashlib.sha256(native_path.read_bytes()).hexdigest()
        if native_hash != EXPECTED_NATIVE_SHA256:
            raise RuntimeError(f"Unexpected native module {native_path} hash {native_hash}; expected {EXPECTED_NATIVE_SHA256}")
        ceed_dlls = tuple((str(path), hashlib.sha256(path.read_bytes()).hexdigest()) for path in loaded_windows_ceed_libraries())
        if EXPECTED_CEED_DLL_SHA256 not in {digest for _, digest in ceed_dlls}:
            raise RuntimeError(f"Loaded CEED DLL hashes {ceed_dlls} omit expected {EXPECTED_CEED_DLL_SHA256}")
        mesh = _build_mesh()
        config = SGSolverConfig(t_begin=0.0, t_end=0.1, steps_count=40, steps_per_tact=4, ceed_resource=resource)
        model = SGModel(mesh, SGSolver(config))
        element_count = int(mesh.strata[MAIN_SPACE].p_node_offsets.size - 1)
        material = np.tile(np.asarray([1.0, 20.0, 0.25], dtype=np.float64), (element_count, 1))
        model.add_physics(SGPhysicsModel.from_entity_values("plane_strain", MATERIAL_ISO_PLANE_STRAIN, mesh, MAIN_SPACE, np.arange(element_count, dtype=np.int64), material))
        force_node = mesh.nearest_nodes(np.asarray([[0.0, 0.0]], dtype=np.float64))
        force_values = np.zeros((5, 1, 2), dtype=np.float64)
        force_values[:, 0, 1] = np.asarray((0.0, -1.0, 0.0, 0.0, 0.0), dtype=np.float64)
        model.add_physics(SGPhysicsModel("short_vertical_nodal_force", N_FORCE_DYNAMIC, mesh, 0, force_node, force_values, time=SGPhysicsTimeData(0.0, 0.1, 5, 3)))
        frames: list[tuple[float, FloatArray]] = []

        def receiver(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
            frames.append((float(getattr(event, "current_time")), np.asarray(frame, dtype=np.float64).copy()))

        points = np.asarray(((-0.75, -0.70), (0.0, -0.10), (0.55, 0.48)), dtype=np.float64)
        model.add_receiver(SGReceiverModel.from_points("ux_uy", mesh, MAIN_SPACE, points, np.asarray([RECEIVER_UX, RECEIVER_UY], dtype=np.int32), begin_tact=0, end_tact=10, stride=1, handler=receiver, frame_shape=(3,)))
        started = perf_counter()
        model.prepare()
        result = model.run(sync=True)
        elapsed = perf_counter() - started
        if not isinstance(result, SGRunResultData) or result.code != SGRunCodeEnum.TIME_END or result.steps_count != 40:
            raise RuntimeError(f"Incomplete run for {resource}: {result}")
        frames.sort(key=lambda item: item[0])
        times = np.asarray([item[0] for item in frames], dtype=np.float64)
        values = np.asarray([item[1].reshape((2, 3)) for item in frames], dtype=np.float64)
        if times.shape != (11,) or values.shape != (11, 2, 3) or not np.isfinite(values).all():
            raise RuntimeError(f"Invalid receiver result for {resource}")
        if float(np.max(np.abs(values))) == 0.0:
            raise RuntimeError(f"No dynamic response for {resource}; stiffness/source probe is inactive")
        return ProbeResult(resource, times, values, elapsed, str(native_path), native_hash, ceed_dlls)
    finally:
        if dll_cookie is not None:
            dll_cookie.close()


def compare(gen: ProbeResult, ref: ProbeResult) -> dict[str, object]:
    if gen.times_s.shape != ref.times_s.shape or not np.allclose(gen.times_s, ref.times_s, rtol=0.0, atol=1.0e-14):
        raise RuntimeError("gen/ref receiver time grids differ")
    difference = gen.frames - ref.frames
    reference_norm = float(np.linalg.norm(ref.frames))
    return {
        "relative_l2": float(np.linalg.norm(difference) / reference_norm),
        "max_abs": float(np.max(np.abs(difference))),
        "reference_l2": reference_norm,
        "response_max_abs": float(np.max(np.abs(ref.frames))),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = PROJECT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll-dir", type=Path, required=True, help="Explicit Windows libCEED bin directory")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    gen = _run("/cpu/self/gen", args.dll_dir)
    ref = _run("/cpu/self/ref", args.dll_dir)
    document: dict[str, object] = {
        "status": "research_sanity_not_solver_acceptance",
        "case": {"mesh": "2x2 Q2 curved; y=eta+.06(1-x^2)(eta+1), detJ>=1", "material": "plane strain rho=1,E=20,nu=.25", "source": "dynamic N_FORCE vertical five-sample pulse", "time": "0..0.1, 40 steps, 4 per tact", "receivers": "three physical points, Ux/Uy"},
        "gen": {"resource": gen.resource, "wall_seconds": gen.wall_seconds, "native_module": gen.native_module, "native_sha256": gen.native_sha256, "loaded_ceed_dlls": gen.ceed_dlls},
        "ref": {"resource": ref.resource, "wall_seconds": ref.wall_seconds, "native_module": ref.native_module, "native_sha256": ref.native_sha256, "loaded_ceed_dlls": ref.ceed_dlls},
        "comparison": compare(gen, ref),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
