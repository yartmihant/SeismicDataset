"""Reproducible one-source p4 smoke, including geometry and timestep certificates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
import shutil
import sys
import tempfile
from time import perf_counter
import traceback
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
    from scripts.dev_1_4_server_preparation.production_surface import SurfaceSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation"
FREE_FLOOR_BYTES = 40 * 1024**3
Dataset = Literal["EFVB", "ECVB"]


@dataclass(frozen=True)
class SmokeSpec:
    dataset: Dataset
    model_id: int
    gpu: int
    out: Path
    t_end_s: float = .06

    def __post_init__(self) -> None:
        if self.dataset not in ("EFVB", "ECVB") or not 0 <= self.model_id < 30000:
            raise ValueError("Expected EFVB/ECVB model ID 00000..29999")
        if self.gpu not in (0, 1):
            raise ValueError("Expected physical GPU index 0 or 1")
        if not isfinite(self.t_end_s) or not 0 < self.t_end_s <= 1:
            raise ValueError("t-end must lie in (0, 1] seconds")
        if abs(self.t_end_s / .001 - round(self.t_end_s / .001)) > 1e-8:
            raise ValueError("t-end must be an integer number of milliseconds")
        self.out.resolve().relative_to(OUTPUT_ROOT)
        if self.out.resolve() == OUTPUT_ROOT:
            raise ValueError("Use a new dedicated smoke subdirectory")


@dataclass(frozen=True)
class PreparedSmoke:
    case: MeshCase
    source_spec: SurfaceSpec
    mesh_path: Path
    stability_path: Path
    preparation_seconds: float


def check_space(path: Path, additional_bytes: int = 0) -> int:
    """Keep 40 GiB available, including headroom for the next explicit write."""
    probe = path.resolve()
    while not probe.exists():
        probe = probe.parent
    free = shutil.disk_usage(probe).free
    if additional_bytes < 0 or free < FREE_FLOOR_BYTES + additional_bytes:
        raise RuntimeError(f"Insufficient disk space: free={free}, required={FREE_FLOOR_BYTES + additional_bytes}")
    return free


def write_json(path: Path, record: dict[str, object]) -> None:
    payload = (json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    check_space(path.parent, len(payload) + 4096)
    with path.open("xb") as stream:
        stream.write(payload)


def configure_environment(spec: SmokeSpec) -> dict[str, str]:
    """Run before importing SEMGPU/native modules; isolate all caches under this smoke."""
    variables = {"CUDA_VISIBLE_DEVICES": str(spec.gpu), "OMP_NUM_THREADS": "2",
                 "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
                 "PYTHONDONTWRITEBYTECODE": "1"}
    for variable, name in (("TMPDIR", "tmp"), ("TMP", "tmp"), ("TEMP", "tmp"),
                           ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"),
                           ("CUDA_CACHE_PATH", "cuda"), ("NUMBA_CACHE_DIR", "numba")):
        cache = spec.out.resolve() / "runtime_cache" / name
        cache.resolve().relative_to(OUTPUT_ROOT)
        cache.mkdir(parents=True, exist_ok=True)
        variables[variable] = str(cache)
    os.environ.update(variables)
    sys.dont_write_bytecode = True
    tempfile.tempdir = variables["TMPDIR"]
    return variables


def prepare(spec: SmokeSpec) -> PreparedSmoke:
    """CPU preparation only; caller may inspect the result without invoking a solver."""
    output = spec.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    free = check_space(output)
    output.mkdir(parents=True, exist_ok=False)
    environment = configure_environment(spec)
    started = perf_counter()
    specification: dict[str, object] = {**asdict(spec), "out": str(output)}
    write_json(output / "smoke.started.json", {"status": "preparing", "spec": specification,
        "utc": datetime.now(timezone.utc).isoformat(), "environment": environment,
        "free_bytes_at_start": free, "minimum_free_bytes": FREE_FLOOR_BYTES,
        "smoke_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})

    # All imports which can reach SEMGPU/native code follow environment selection.
    from scripts.dev_1_3_point_source.mesh import build_sem
    from scripts.dev_1_3_point_source.stability import estimate
    from scripts.dev_1_4_server_preparation.curved_mesh import build_curved
    from scripts.dev_1_4_server_preparation.model_input import read_model
    from scripts.dev_1_4_server_preparation.production_surface import ALPHA_MAX_S_INV, SurfaceSpec
    from scripts.dev_1_4_server_preparation.production_worker import save_dataset_mesh, select_substeps
    from scripts.reporting import number

    model = read_model(spec.dataset, spec.model_id)
    geometry = build_curved(model)
    mesh = geometry.mesh
    mesh_path = output / "mesh_q2.npz"
    mesh_bytes = sum(array.nbytes for array in (mesh.nodes, mesh.cells, mesh.vp, mesh.vs, mesh.regions)) + 65_536
    check_space(output, mesh_bytes)
    save_dataset_mesh(mesh, spec.dataset, mesh_path)
    geometry.report["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    write_json(output / "geometry.json", geometry.report)
    certificate = estimate(mesh_path, 4, ALPHA_MAX_S_INV)
    certificate["dataset"] = spec.dataset
    substeps = select_substeps(spec.dataset, spec.model_id, int(number(certificate["substeps_per_ms"])))
    certificate["selected_substeps_per_ms"] = substeps
    certificate["selection_rule"] = "ECVB12404=64, ECVB65=224; otherwise ceil(1.6*Kmin/8)*8"
    stability_path = output / "stability.json"
    write_json(stability_path, certificate)
    case = build_sem(mesh, 4)
    case.metadata["dataset"] = spec.dataset
    source_spec = SurfaceSpec(spec.dataset, spec.model_id, 350., substeps, spec.t_end_s)
    elapsed = perf_counter() - started
    print(json.dumps({"stage": "cpu_prepared", "dataset": spec.dataset, "model_id": spec.model_id,
        "gpu": spec.gpu, "substeps_per_ms": substeps, "t_end_s": spec.t_end_s,
        "preparation_seconds": elapsed, "out": str(output)}), flush=True)
    return PreparedSmoke(case, source_spec, mesh_path, stability_path, elapsed)


def run(spec: SmokeSpec) -> dict[str, object]:
    output = spec.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    started = perf_counter()
    try:
        prepared = prepare(spec)
        from scripts.dev_1_4_server_preparation.production_surface import run_case

        def before_write() -> None:
            check_space(output, (round(spec.t_end_s / .001) + 1) * 70 * 2 * 8 + 131_072)

        before_write()
        result = run_case(prepared.case, prepared.source_spec, output / "x350", prepared.mesh_path,
                          prepared.stability_path, before_write=before_write)
        report: dict[str, object] = {"status": result["status"], "dataset": spec.dataset,
            "model_id": spec.model_id, "gpu": spec.gpu, "source_x_m": 350., "t_end_s": spec.t_end_s,
            "substeps_per_ms": prepared.source_spec.substeps, "order": 4, "geometry": "curved",
            "preparation_seconds": prepared.preparation_seconds, "wall_seconds": perf_counter() - started,
            "surface": str(output / "x350/surface.npz"), "surface_metadata": str(output / "x350/metadata.json"),
            "free_bytes_at_end": check_space(output), "scope": "single central source; short integration unless t-end=1"}
        write_json(output / "smoke.json", report)
        return report
    except BaseException as error:
        if output.is_dir():
            try:
                write_json(output / "smoke.failed.json", {"status": "failed",
                    "error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc(),
                    "wall_seconds": perf_counter() - started})
            except BaseException as diagnostic_error:
                print(json.dumps({"failure_metadata_not_written": str(diagnostic_error)}), file=sys.stderr, flush=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("EFVB", "ECVB"), required=True)
    parser.add_argument("--model", type=int, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--t-end", type=float, default=.06)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = SmokeSpec(cast(Dataset, args.dataset), args.model, args.gpu, args.out, args.t_end)
    print(json.dumps(run(spec), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
