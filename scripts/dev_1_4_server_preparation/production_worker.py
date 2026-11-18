"""One GPU worker for the explicitly enabled curved Vx/Vy70 chunk queue.

Only a successfully verified model's temporary work directory is removed.
Dataset outputs, failed work and disk reservations are never auto-deleted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Callable

import numpy as np

from scripts.dev_1_3_point_source.mesh import FrozenMesh, build_sem
from scripts.dev_1_3_point_source.stability import estimate
from scripts.dev_1_4_server_preparation.production_plan import (
    BudgetUnavailable, DiskGuard, GuardConfig, JobReservation, build_chunks,
)
from scripts.dev_1_4_server_preparation.curved_mesh import build_curved
from scripts.dev_1_4_server_preparation.model_input import read_model
from scripts.project_paths import PROJECT_ROOT, SOURCE
from scripts.reporting import mapping, number, sequence


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def directory_bytes(root: Path) -> int:
    """Actual allocated blocks, not apparent sizes or optimistic compression."""
    size = 0
    for path in [root, *root.rglob("*")]:
        try:
            size += path.lstat().st_blocks * 512
        except FileNotFoundError:
            # The other GPU may just have removed its verified temporary work directory.
            continue
    return size


def select_substeps(dataset: str, model_id: int, minimum: int) -> int:
    """Keep reference K64/224; elsewhere use 1.6×the bound, rounded up to eight."""
    if minimum < 1:
        raise ValueError("Nonpositive temporal certificate")
    reference = {12404: 64, 65: 224}.get(model_id) if dataset == "ECVB" else None
    if reference is not None:
        if reference < minimum:
            raise ValueError("Reference temporal step no longer satisfies this geometry certificate")
        return reference
    return 8 * ((minimum + 4) // 5)


def implementation_provenance() -> dict[str, object]:
    """Identify the algorithms and environment needed to reconstruct removed meshes."""
    paths = (
        "scripts/dev_1_4_server_preparation/curved_mesh.py",
        "scripts/dev_1_4_server_preparation/model_input.py",
        "scripts/dev_1_4_server_preparation/production_surface.py",
        "scripts/dev_1_4_server_preparation/production_worker.py",
        "scripts/dev_1_4_server_preparation/production_plan.py",
        "scripts/dev_1_4_server_preparation/production_launcher.py",
        "scripts/dev_1_3_point_source/mesh.py",
        "scripts/dev_1_3_point_source/stability.py",
        "scripts/dev_1_3_point_source/physics.py",
        "scripts/dev_1_2_smoothing/model65_geometry.py",
        "scripts/dev_1_2_smoothing/pilot_mesh.py",
        "scripts/dev_1_1_preprocessing/prepare_data.py",
    )
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                              check=True, capture_output=True, text=True, timeout=10).stdout.strip()
    return {"project_git_head": revision,
            "code_sha256": {path: hashlib.sha256((PROJECT_ROOT / path).read_bytes()).hexdigest() for path in paths},
            "package_versions": {package: version(package) for package in ("numpy", "scipy", "gmsh", "semgpu2")},
            "rebuild": "Use original VP/VS and these exact algorithm files/package versions; no raster fallback"}


def progress_counts(status: dict[str, object]) -> tuple[int, int]:
    """Return active and failed model counts for a strict inter-chunk barrier."""
    active = failed = 0
    for item in sequence(status["progress"]):
        row = mapping(item)
        count = int(number(row["models"]))
        if row["status"] == "reserved":
            active += count
        if row["status"] == "failed":
            failed += count
    return active, failed


def compact_geometry(report: dict[str, object], mesh_sha256: str, provenance: dict[str, object]) -> dict[str, object]:
    """Retain certificates and provenance without every sampled CAD curve."""
    construction = mapping(report["construction"])
    keys = ("construction", "source_sample_bound_m", "source_topology", "source_regions",
            "rows_topdown_before_refinement", "curve_intersection_check", "settings", "method")
    compact = {key: construction[key] for key in keys if key in construction}
    method = construction.get("construction")
    if method == "layered_pilot_Q2_exact_2x2":
        compact["settings"] = {"fit": "cubic least squares, locked endpoints", "degree": 3,
            "control_counts": [10, 14, 20, 28, 40], "source_centres_x_m": "5:10:695",
            "source_bound_m": 7.5, "frozen_Q2_x_m": "0:5:700", "row_rule": "max(1, round(mean_layer_thickness/10m))",
            "refinement": "exact 2x2 subdivision of frozen Q2", "absorbing_extension_m": 175.}
    elif method == "shared_CAD_model65_rules_plus_periodic_closed_chains":
        compact["settings"] = {"fit": "cubic arclength least squares; straight interfaces degree1",
            "control_counts": [4, 5, 6, 8, 10, 14, 20, 28, 40, 52, 64, "source_count"],
            "periodic_fit": "cyclic cubic, C2 seam, two exact clamped CAD arcs", "source_bound_m": 7.5,
            "gmsh_seed_min_max_m": 12.5, "outer_transfinite_spacing_m": 10., "gmsh_algorithm": 6,
            "gmsh_subdivision_algorithm": 1, "gmsh_recombine_all": 0, "gmsh_second_order_incomplete": 0,
            "gmsh_second_order_linear": 0, "gmsh_threads": 2, "Q2_centre": "serendipity",
            "final_top_edge_m": 5., "absorbing_extension_m": 175.}
    if "curves" in construction:
        compact["curves_count"] = len(sequence(construction["curves"]))
    if "closed_chain_fits" in construction:
        compact["closed_chain_fits_count"] = len(sequence(construction["closed_chain_fits"]))
    for collection, metric in (("splines", "max_sample_depth_error_m"),
                               ("curves", "sampled_source_point_max_error_m"),
                               ("closed_chain_fits", "sampled_source_point_max_error_m"),
                               ("closed_chain_fits", "cad_arc_restriction_max_error_m")):
        if collection in construction:
            records = [mapping(item) for item in sequence(construction[collection])]
            values = [number(record[metric]) for record in records if metric in record]
            if values:
                compact[f"{collection}_{metric}"] = max(values)
    if "closed_chain_fits" in construction:
        periodic = [mapping(item) for item in sequence(construction["closed_chain_fits"])]
        compact["periodic_fit_methods"] = sorted({str(record["fit"]) for record in periodic if "fit" in record})
        errors = [number(error) for record in periodic for error in sequence(record.get("periodic_seam_errors_derivatives_0_1_2", []))]
        if errors:
            compact["periodic_seam_max_error_derivatives_0_1_2"] = max(errors)
    encoded = json.dumps(report, sort_keys=True, allow_nan=False).encode()
    return {"schema": report["schema"], "dataset": report["dataset"], "model_id": report["model_id"],
            "input": report["input"], "certificate": report["certificate"], "construction": compact,
            "fallback": report["fallback"], "algorithm_evidence": report["algorithm_evidence"],
            "implementation": provenance,
            "mesh_sha256": mesh_sha256, "full_geometry_report_sha256": hashlib.sha256(encoded).hexdigest(),
            "full_geometry_report_sha256_format": "json.dumps(report, sort_keys=True, allow_nan=False).encode()",
            "full_geometry_report_retention": "temporary work only; deleted after five verified sources"}


def write_json(path: Path, value: dict[str, object], before_write: Callable[[int], None]) -> None:
    data = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    before_write(len(data) + 4096)
    with path.open("xb") as stream:
        stream.write(data)


def save_dataset_mesh(mesh: FrozenMesh, dataset: str, path: Path) -> None:
    """Keep the group explicit while preserving the chapter 1.3 frozen mesh schema."""
    with path.open("xb") as stream:
        np.savez(stream, nodes=mesh.nodes, cells=mesh.cells, vp=mesh.vp, vs=mesh.vs,
                 regions=mesh.regions, model_id=mesh.model_id, kind=mesh.kind, dataset=dataset)


def run_reserved_model(root: Path, source: Path, guard: DiskGuard, job: JobReservation) -> dict[str, object]:
    """Generate one certified geometry, reuse it for five independent sources."""
    from scripts.dev_1_4_server_preparation.production_surface import ALPHA_MAX_S_INV, SurfaceSpec, run_case, verify_surface

    chunk = build_chunks()[job.chunk_number - 1]
    output = root / "chunks" / chunk.name / "models" / f"{job.model_id:05d}"
    work = root / "work" / job.dataset / f"{job.model_id:05d}"
    if output.exists() or work.exists():
        raise FileExistsError("Model output/work already exists; automatic overwrite/resume is forbidden")
    guard.check_before_write(job.job_id, 8192)
    output.mkdir(parents=True, exist_ok=False)
    work.mkdir(parents=True, exist_ok=False)
    started = utc()
    provenance = implementation_provenance()

    def check_write(extra: int = 0) -> None:
        _, failed = progress_counts(guard.status())
        if failed:
            raise BudgetUnavailable("Another worker failed; stop before further model writes")
        guard.check_before_write(job.job_id, directory_bytes(output) + extra)

    def check_work_write(extra: int) -> None:
        # At most two active jobs each receive half W, so concurrent writes cannot overbook W.
        allowance = guard.config.working_floor_bytes // guard.config.max_in_flight
        if directory_bytes(work) + extra > allowance or directory_bytes(root / "work") + extra > guard.config.working_floor_bytes:
            raise BudgetUnavailable("Temporary geometry exceeds its share of the working allowance")
        check_write()

    model = read_model(job.dataset, job.model_id, source)
    check_write(80_000)
    with (output / "input.npz").open("xb") as stream:
        np.savez_compressed(stream, vp=model.vp, vs=model.vs)
    geometry = build_curved(model)
    mesh_path = work / "mesh_q2.npz"
    mesh = geometry.mesh
    mesh_upper_bound = sum(array.nbytes for array in (mesh.nodes, mesh.cells, mesh.vp, mesh.vs, mesh.regions)) + 65_536
    check_work_write(mesh_upper_bound)
    save_dataset_mesh(mesh, job.dataset, mesh_path)
    mesh_hash = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    full_report = (json.dumps(geometry.report, ensure_ascii=False, allow_nan=False) + "\n").encode()
    check_work_write(len(full_report) + 4096)
    with (work / "geometry_full.json").open("xb") as stream:
        stream.write(full_report)
    geometry_compact = compact_geometry(geometry.report, mesh_hash, provenance)
    write_json(output / "geometry.json", geometry_compact, check_write)
    certificate = estimate(mesh_path, 4, ALPHA_MAX_S_INV)
    certificate["dataset"] = job.dataset
    minimum = int(number(certificate["substeps_per_ms"]))
    substeps = select_substeps(job.dataset, job.model_id, minimum)
    certificate["selected_substeps_per_ms"] = substeps
    certificate["selection_rule"] = "ECVB12404=64, ECVB65=224; otherwise ceil(1.6*Kmin/8)*8"
    stability_path = output / "stability.json"
    write_json(stability_path, certificate, check_write)
    case = build_sem(geometry.mesh, 4)
    case.metadata["dataset"] = job.dataset
    sources = (0.0, 175.0, 350.0, 525.0, 700.0)
    source_evidence: list[dict[str, object]] = []
    for source_x in sources:
        check_write()
        spec = SurfaceSpec(job.dataset, job.model_id, source_x, substeps)
        destination = output / f"x{int(source_x):03d}"
        expected_after_source = directory_bytes(output) + 1_130_808 + 32_768

        def before_surface_write() -> None:
            # Repeated callback calls reserve the same complete source once, including metadata rewrites.
            _, failed = progress_counts(guard.status())
            if failed:
                raise BudgetUnavailable("Another worker failed; stop before further source writes")
            guard.check_before_write(job.job_id, max(directory_bytes(output), expected_after_source))

        run_case(case, spec, destination, mesh_path, stability_path, before_write=before_surface_write)
        verification = verify_surface(destination, spec)
        source_evidence.append({"source_x_m": source_x, "verification": verification})
    # Verify the five retained outputs again before removal of working geometry.
    for source_x in sources:
        verify_surface(output / f"x{int(source_x):03d}", SurfaceSpec(job.dataset, job.model_id, source_x, substeps))
    check_write(16_384)
    if work.is_symlink() or work.resolve() != root.resolve() / "work" / job.dataset / f"{job.model_id:05d}":
        raise ValueError("Unexpected work path before authorized temporary cleanup")
    shutil.rmtree(work)
    summary: dict[str, object] = {"status": "completed", "dataset": job.dataset, "model_id": job.model_id,
        "chunk_number": job.chunk_number, "gpu": job.gpu, "started_utc": started, "finished_utc": utc(),
        "order": 4, "geometry": "curved", "mesh_sha256": mesh_hash, "substeps_per_ms": substeps,
        "sources": source_evidence, "implementation": provenance,
        "temporary_mesh_removed_after_verification": True}
    write_json(output / "model.json", summary, check_write)
    size = directory_bytes(output)
    guard.complete(job.job_id, size)
    return {"job_id": job.job_id, "status": "completed", "allocated_bytes": size,
            "output": str(output), "substeps_per_ms": substeps}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--max-models", type=int, help="Limit newly completed models for an explicitly scoped worker trial")
    args = parser.parse_args()
    root: Path = args.root.resolve()
    root.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if args.max_models is not None and args.max_models < 1:
        raise ValueError("max-models must be positive")
    root.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for variable, relative in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"),
                               ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
        cache = PROJECT_ROOT / ".runtime_cache" / f"gpu{args.gpu}" / relative
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    guard = DiskGuard(root / "queue.sqlite", root, GuardConfig())
    completed = 0
    while args.max_models is None or completed < args.max_models:
        try:
            job = guard.reserve_next(args.gpu)
        except BudgetUnavailable as exc:
            print(json.dumps({"event": "queue_blocked", "gpu": args.gpu, "utc": utc(), "reason": str(exc)}), flush=True)
            raise SystemExit(2) from exc
        if job is None:
            active, failed = progress_counts(guard.status())
            if failed:
                raise SystemExit("Queue failed; no automatic skip or reservation release")
            if active:
                time.sleep(10)
                continue
            print(json.dumps({"event": "authorized_chunk_limit_complete", "chunk_limit": guard.config.execution_chunk_limit,
                              "gpu": args.gpu, "utc": utc()}), flush=True)
            break
        print(json.dumps({"event": "model_started", "job_id": job.job_id, "gpu": args.gpu, "utc": utc()}), flush=True)
        try:
            result = run_reserved_model(root, args.source.resolve(), guard, job)
        except Exception as exc:
            guard.fail(job.job_id, f"{type(exc).__name__}: {exc}")
            print(json.dumps({"event": "model_failed", "job_id": job.job_id, "gpu": args.gpu,
                              "utc": utc(), "reason": str(exc), "traceback": traceback.format_exc()}), flush=True)
            raise
        completed += 1
        print(json.dumps({"event": "model_completed", "utc": utc(), **result}), flush=True)


if __name__ == "__main__":
    main()
