"""CPU-only EFVB00000 ablation: remove padding by retaining exact existing Q2 cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.dev_1_3_point_source.mesh import FrozenMesh, audit, build_sem, edges, jacobians, load_frozen
from scripts.dev_1_3_point_source.stability import estimate
from scripts.dev_1_4_server_preparation.worker import save_dataset_mesh
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import number

IntArray = NDArray[np.int64]
OUTPUT_ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation"
TOLERANCE_M = 1e-8


def index_ranges(indices: IntArray) -> list[list[int]]:
    """Compact half-open source index ranges, sufficient to repeat the exact selection."""
    cuts = np.r_[0, np.flatnonzero(np.diff(indices) != 1) + 1, len(indices)]
    return [[int(indices[begin]), int(indices[end - 1]) + 1] for begin, end in zip(cuts[:-1], cuts[1:], strict=True)]


def crop(source: FrozenMesh) -> tuple[FrozenMesh, IntArray, IntArray]:
    """Select complete cells in [0,700]×[-700,0]; only renumber their used nodes."""
    local = source.nodes[source.cells]
    low, high = local.min(axis=1), local.max(axis=1)
    box_low, box_high = np.array([0., -700.]), np.array([700., 0.])
    inside = np.all(low >= box_low - TOLERANCE_M, axis=1) & np.all(high <= box_high + TOLERANCE_M, axis=1)
    overlap = np.all(np.minimum(high, box_high) - np.maximum(low, box_low) > TOLERANCE_M, axis=1)
    if np.any(overlap & ~inside):
        raise ValueError("An existing Q2 cell straddles the requested crop boundary")
    retained = np.flatnonzero(inside).astype(np.int64)
    used = np.unique(source.cells[retained]).astype(np.int64)
    remap = np.full(len(source.nodes), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    core = FrozenMesh(source.nodes[used], remap[source.cells[retained]], source.vp[retained],
                      source.vs[retained], source.regions[retained], source.model_id, source.kind)
    if len(core.cells) != 19600 or len(core.nodes) != 78961:
        raise ValueError("Expected EFVB00000 core with 19600 full Q2 cells and 78961 Q2 nodes")
    return core, retained, used


def exact_subset(core: FrozenMesh, source: FrozenMesh, retained: IntArray, used: IntArray) -> dict[str, object]:
    checks = {"used_node_coordinates": np.array_equal(core.nodes, source.nodes[used]),
              "cell_local_coordinates": np.array_equal(core.nodes[core.cells], source.nodes[source.cells[retained]]),
              "vp": np.array_equal(core.vp, source.vp[retained]),
              "vs": np.array_equal(core.vs, source.vs[retained]),
              "regions": np.array_equal(core.regions, source.regions[retained])}
    if not all(checks.values()):
        raise ValueError("Cropped geometry/materials differ from the exact retained source cells")
    return {name: bool(value) for name, value in checks.items()}


def audit_core(core: FrozenMesh, source_certificate: dict[str, object]) -> dict[str, object]:
    """Check the cropped disk; inherit a fresh whole-cell proof through exact Q2 identity."""
    directed, inverse, counts, exterior = edges(core)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold edge")
    midpoints = np.full(len(counts), -1, dtype=np.int64)
    midpoints[inverse] = directed[:, 1]
    if not np.array_equal(midpoints[inverse], directed[:, 1]):
        raise ValueError("Nonconforming Q2 midpoint")
    signs = np.where(directed[:, 0] < directed[:, 2], 1, -1)
    if np.any(np.bincount(inverse, weights=signs)[counts == 2] != 0):
        raise ValueError("Adjacent Q2 cells have inconsistent orientation")
    boundary = core.nodes[directed[exterior]]
    side = np.stack([np.all(np.abs(boundary[:, :, axis] - value) < TOLERANCE_M, axis=1)
                     for axis, value in ((0, 0.), (0, 700.), (1, -700.), (1, 0.))])
    if not np.all(np.sum(side, axis=0) == 1) or not np.all(np.sum(side, axis=1) == 140):
        raise ValueError("Boundary is not exactly four complete 140-edge rectangle sides")
    euler = len(np.unique(core.cells[:, [0, 2, 6, 8]])) - len(counts) + len(core.cells)
    if euler != 1:
        raise ValueError("Cropped mesh is not a disk")
    top = boundary[side[-1]]
    if not np.allclose(np.abs(top[:, -1, 0] - top[:, 0, 0]), 5., rtol=0., atol=TOLERANCE_M):
        raise ValueError("Top boundary is not 140×5m")
    points, weights = np.polynomial.legendre.leggauss(3)
    areas = np.einsum("eij,i,j->e", jacobians(core, points), weights, weights)
    if abs(float(areas.sum()) - 490000.) > 1e-5:
        raise ValueError("Cropped area is not 490000m²")
    certified_minimum = number(source_certificate["det_jacobian_certified_min_m2"])
    if certified_minimum <= 0:
        raise ValueError("The source does not provide a positive whole-cell Jacobian bound")
    return {"elements": len(core.cells), "q2_nodes": len(core.nodes), "area_m2": float(areas.sum()),
        "conforming": True, "euler_characteristic": euler, "straddling_cells": 0,
        "core_top_edges": 140, "core_top_edge_m": 5., "external_edges": len(exterior),
        "abc_edge_count": int(np.count_nonzero(~side[-1])), "free_surface_edges": 140,
        "det_jacobian_certified_min_m2": certified_minimum,
        "whole_cell_jacobian_proof": "Fresh Bernstein certificate for every source Q2 cell; exact retained cell-local coordinates preserve the complete polynomial map and its bound",
        "source_bernstein_max_depth": source_certificate["bernstein_max_depth"],
        "area_by_region_m2": {str(int(region)): float(areas[core.regions == region].sum()) for region in np.unique(core.regions)}}


def write_json(path: Path, record: dict[str, object]) -> None:
    with path.open("x") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def build(input_path: Path, output: Path) -> dict[str, object]:
    input_path, output = input_path.resolve(), output.resolve()
    input_path.relative_to(OUTPUT_ROOT)
    output.relative_to(OUTPUT_ROOT)
    if output.exists():
        raise FileExistsError(output)
    with np.load(input_path, allow_pickle=False) as archive:
        if "dataset" not in archive or str(archive["dataset"]) != "EFVB":
            raise ValueError("This ablation requires an explicitly labelled EFVB mesh")
    source = load_frozen(input_path)
    if source.model_id != 0 or source.kind != "curved":
        raise ValueError("This ablation is restricted to the existing curved EFVB00000 mesh")
    source_certificate = audit(source)
    core, retained, used = crop(source)
    equality = exact_subset(core, source, retained, used)
    certificate = audit_core(core, source_certificate)
    case = build_sem(core, 4)
    if len(case.mesh.nodes_coords) != 314721 or len(case.element_vp) != 19600:
        raise ValueError("Unexpected p4 mesh size")
    coords = case.mesh.nodes_coords
    expected_x = np.r_[np.arange(5., 700., 10.), 0., 175., 350., 525., 700.]
    if any(np.count_nonzero(np.linalg.norm(coords - [x, 0.], axis=1) < 1e-9) != 1 for x in expected_x):
        raise ValueError("A receiver/source is not a unique exact surface node")
    output.mkdir(parents=True, exist_ok=False)
    mesh_path = output / "mesh_q2.npz"
    save_dataset_mesh(core, "EFVB", mesh_path)
    # Check the serialized/reloaded artifact against the original, not only in-memory views.
    exact_subset(load_frozen(mesh_path), source, retained, used)
    stability = estimate(mesh_path, 4, 0.)
    if int(number(stability["substeps_per_ms"])) > 96 or .001 / 96 > number(stability["dt_start_s"]):
        raise ValueError("Requested K96 is unsafe for the no-padding geometry")
    if int(number(stability["abc_edge_count"])) != 420:
        raise ValueError("Stability calculation did not use all 420 side/bottom ABC edges")
    stability.update({"dataset": "EFVB", "selected_substeps_per_ms": 96,
        "selection_rule": "Fixed K96 to match the padded EFVB00000 ablation reference; verified against this alpha=0 certificate"})
    write_json(output / "stability.json", stability)
    report: dict[str, object] = {"schema": "chapter14.no_padding_exact_crop.v1", "dataset": "EFVB", "model_id": 0,
        "source_mesh": str(input_path), "source_mesh_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "source_fresh_geometry_certificate": source_certificate,
        "mesh_sha256": hashlib.sha256(mesh_path.read_bytes()).hexdigest(),
        "method": "Exact cell subset and used-node renumbering; no spline fitting, coordinate movement or remeshing",
        "bounds_m": {"x": [0., 700.], "y": [-700., 0.]}, "padding_m": 0., "alpha_max_s_inv": 0.,
        "retained_source_cell_index_ranges_half_open": index_ranges(retained),
        "used_source_node_index_ranges_half_open": index_ranges(used),
        "exact_subset_checks_in_memory_and_reloaded": equality, "certificate": certificate,
        "p4_nodes": len(coords), "p4_elements": len(case.element_vp), "receiver_count": 70,
        "source_x_m": 350., "all_five_source_positions_are_nodes": True,
        "planned_substeps_per_ms": 96, "solver_run": False,
        "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    write_json(output / "geometry.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.input, args.out)
    print(json.dumps({"dataset": report["dataset"], "model_id": report["model_id"],
        "certificate": report["certificate"], "p4_nodes": report["p4_nodes"],
        "selected_substeps_per_ms": 96, "solver_run": False}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
