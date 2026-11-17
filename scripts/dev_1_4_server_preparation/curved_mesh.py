"""Generalize the two established curved constructions; refuse unsupported geometry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline

from scripts.dev_1_2_smoothing import model65_geometry as graph
from scripts.dev_1_2_smoothing.pilot_mesh import quadratic_range, quadratic_values
from scripts.dev_1_3_point_source.mesh import FrozenMesh, audit, edges, extend, jacobians, save_frozen, structured_cells
from scripts.dev_1_4_server_preparation.model_input import ModelInput, read_model
from scripts.project_paths import PROJECT_ROOT, SOURCE
from scripts.reporting import mapping, number, sequence

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class GeometryResult:
    mesh: FrozenMesh
    report: dict[str, object]


def layer_sequence(vp: FloatArray, vs: FloatArray) -> tuple[IntArray, FloatArray, FloatArray] | None:
    """Detect exactly the nonvanishing material sequence used by the layered pilot."""
    labels, values = graph._material_labels(vp, vs)
    changes = [np.r_[0, np.flatnonzero(np.diff(column)) + 1, 70] for column in labels.T]
    sequences = [column[index[:-1]] for column, index in zip(labels.T, changes, strict=True)]
    if any(not np.array_equal(sequences[0], candidate) for candidate in sequences[1:]):
        return None
    return sequences[0], np.asarray(values, dtype=np.float64), np.asarray([change[1:-1] * 10.0 for change in changes], dtype=np.float64).T


def layered_core(model: ModelInput, layers: tuple[IntArray, FloatArray, FloatArray]) -> GeometryResult:
    """Same locked-end cubic fit, frozen Q2 and exact 2×2 refinement as model12404."""
    material_sequence, materials, interfaces = layers
    x_centres = np.arange(70, dtype=np.float64) * 10.0 + 5.0
    x = np.r_[0.0, x_centres, 700.0]
    source = np.column_stack((interfaces[:, 0], interfaces, interfaces[:, -1])) if interfaces.size else np.empty((0, 72))
    depths = np.empty((len(material_sequence) + 1, 141), dtype=np.float64)
    depths[0], depths[-1] = 0.0, 700.0
    records: list[dict[str, object]] = []
    for row, values in enumerate(source):
        for count in (10, 14, 20, 28, 40):
            knots = np.r_[np.zeros(4), np.linspace(0, 700, count - 2)[1:-1], np.full(4, 700.0)]
            basis = BSpline.design_matrix(x, knots, 3).toarray()
            rhs = values - basis[:, 0] * values[0] - basis[:, -1] * values[-1]
            coefficients = np.r_[values[0], np.linalg.lstsq(basis[:, 1:-1], rhs, rcond=None)[0], values[-1]]
            spline = BSpline(knots, coefficients, 3)
            deviation = float(np.max(np.abs(np.asarray(spline(x_centres)) - interfaces[row])))
            if deviation <= 7.5:
                depths[row + 1] = spline(np.linspace(0, 700, 141))
                records.append({"knots": knots.tolist(), "coefficients": coefficients.tolist(), "max_sample_depth_error_m": deviation})
                break
        else:
            raise ValueError("Layer fit cannot satisfy the existing 7.5m source-point bound")
    thickness = np.diff(depths, axis=0)
    low, _ = quadratic_range(thickness)
    if low.min() <= 0:
        raise ValueError("Frozen Q2 layer interfaces intersect")
    means = np.sum((thickness[:, :-1:2] + 4 * thickness[:, 1::2] + thickness[:, 2::2]) * 10 / 6, axis=1) / 700
    rows = np.maximum(1, np.rint(means / 10).astype(np.int64))
    xs = np.linspace(0, 700, 281)
    values = quadratic_values(depths, xs)
    offsets = np.r_[0, np.cumsum(rows)]
    logical = np.linspace(0, rows.sum(), int(rows.sum()) * 4 + 1)
    layer = np.minimum(np.searchsorted(offsets, logical, side="right") - 1, len(rows) - 1)
    fraction = (logical - offsets[layer]) / rows[layer]
    y = -((1 - fraction[:, None]) * values[layer] + fraction[:, None] * values[layer + 1])
    coords = np.stack((np.broadcast_to(xs, y.shape), y), axis=-1)[::-1].reshape((-1, 2))
    region = np.repeat(np.arange(len(rows)), 2 * rows)[::-1]
    properties = np.repeat(materials[material_sequence][region], 140, axis=0)
    core = FrozenMesh(coords, structured_cells(140, int(2 * rows.sum())), properties[:, 0], properties[:, 1], np.repeat(region, 140), model.model_id, "curved")
    return GeometryResult(core, {"construction": "layered_pilot_Q2_exact_2x2", "splines": records,
        "rows_topdown_before_refinement": rows.tolist(), "source_sample_bound_m": 7.5,
        "source_topology": "same nonvanishing layer sequence in every column"})


def shared_curves(model: ModelInput) -> dict[str, object]:
    """Reuse model65's topology, locked endpoints and 7.5m fit; no repair/fallback."""
    materials, values = graph._material_labels(model.vp, model.vs)
    owners, regions = graph._regions(materials)
    chains = graph._chains(graph._boundary_edges(owners))
    curves: list[dict[str, object]] = []
    samples: list[FloatArray] = []
    periodic_records: list[dict[str, object]] = []
    for chain in chains:
        source = graph._physical(chain.source)
        if np.array_equal(source[0], source[-1]):
            pieces, periodic_record = periodic_curve_pieces(source)
            periodic_records.append(periodic_record)
        else:
            degree, controls, knots, error = graph._fit_curve(source)
            pieces = [(source, degree, controls, knots, error)]
        for piece_source, degree, controls, knots, error in pieces:
            sampled = graph._sample(degree, controls, knots)
            curves.append({"id": len(curves) + 1, "adjacent_regions": [chain.left, chain.right], "degree": degree,
                "knots": knots.tolist(), "control_points_m": controls.tolist(), "source_points_m": piece_source.tolist(),
                "sampled_points_m": sampled.tolist(), "sampled_source_point_max_error_m": error})
            samples.append(sampled)
    graph._check_crossings(samples)
    loops = graph._region_loops(curves, regions)
    by_id = {int(number(curve["id"])): curve for curve in curves}
    for region_id, region_loops in loops.items():
        def area(loop: list[int]) -> float:
            polygon: list[tuple[float, float]] = []
            for signed_id in loop:
                points = np.asarray(by_id[abs(signed_id)]["source_points_m"], dtype=np.float64)
                if signed_id < 0:
                    points = points[::-1]
                polygon.extend((float(point[0]), float(point[1])) for point in points[:-1])
            return graph._signed_area(polygon)
        ordered = sorted(region_loops, key=lambda loop: abs(area(loop)), reverse=True)
        if area(ordered[0]) <= 0 or any(area(loop) >= 0 for loop in ordered[1:]):
            raise ValueError("Region must have one positive outer loop and negative hole loops")
        loops[region_id] = ordered
    return {"curves": curves, "regions": [{"id": r.identifier, "vp": values[r.material][0], "vs": values[r.material][1],
        "source_cells": r.cells, "loops": loops[r.identifier]} for r in regions],
        "source_regions": len(regions), "construction": "shared_CAD_model65_rules_plus_periodic_closed_chains",
        "closed_chain_fits": periodic_records,
        "source_sample_bound_m": 7.5, "curve_intersection_check": "101 sampled points per curve; not an exact spline intersection proof"}


def periodic_curve_pieces(source: FloatArray) -> tuple[list[tuple[FloatArray, int, FloatArray, FloatArray, float]], dict[str, object]]:
    """Fit a C2 periodic cubic, then represent it by two exact clamped CAD arcs.

    The two arcs introduce no gap, kink or independent fitting. Splitting uses
    exact knot insertion; it only avoids ambiguous closed-edge CAD endpoints.
    There is no fixed artificial endpoint in the periodic least-squares fit.
    """
    if not np.array_equal(source[0], source[-1]) or len(source) < 5:
        raise ValueError("Expected a closed grid loop with at least four edges")
    parameters = graph._parameters(source)
    count_points = len(source) - 1
    candidates = sorted({4, 5, 6, 8, 10, 14, 20, 28, 40, 52, 64, count_points})
    for control_count in candidates:
        if control_count > count_points:
            continue
        knots = np.arange(-3, control_count + 4, dtype=np.float64) / control_count
        design = BSpline.design_matrix(parameters[:-1], knots, 3).toarray()
        periodic_design = design[:, :control_count].copy()
        periodic_design[:, :3] += design[:, control_count:]
        controls, _, _, _ = np.linalg.lstsq(periodic_design, source[:-1], rcond=None)
        periodic = BSpline(knots, np.vstack((controls, controls[:3])), 3)
        error = float(np.max(np.linalg.norm(periodic(parameters) - source, axis=1)))
        if error <= 7.5:
            break
    else:
        raise ValueError("Periodic closed fit cannot satisfy the 7.5m source-point bound")
    seam_errors = [float(np.linalg.norm(periodic(0., nu=derivative) - periodic(1., nu=derivative))) for derivative in range(3)]
    if max(seam_errors) > 1e-6:
        raise ValueError("Periodic spline does not have a verified C2 seam")
    # Split at an existing source vertex nearest half-perimeter for exact source-loop bookkeeping.
    split = int(np.argmin(np.abs(parameters - .5)))
    middle = float(parameters[split])
    pieces: list[tuple[FloatArray, int, FloatArray, FloatArray, float]] = []
    max_restriction_error = 0.0
    for begin, end, original in ((0., middle, source[:split + 1]), (middle, 1., source[split:])):
        restricted = periodic
        for value in (begin, end):
            multiplicity = int(np.count_nonzero(restricted.t == value))
            restricted = restricted.insert_knot(value, 4 - multiplicity)
        first = int(np.flatnonzero(restricted.t == begin)[0])
        last = int(np.flatnonzero(restricted.t == end)[-1] + 1)
        arc_knots = np.asarray(restricted.t[first:last], dtype=np.float64)
        arc_controls = np.asarray(restricted.c[first:first + len(arc_knots) - 4], dtype=np.float64)
        normalized_knots = (arc_knots - begin) / (end - begin)
        arc = BSpline(normalized_knots, arc_controls, 3)
        sample = np.linspace(0., 1., 201)
        max_restriction_error = max(max_restriction_error, float(np.max(np.linalg.norm(arc(sample) - periodic(begin + (end - begin) * sample), axis=1))))
        pieces.append((original, 3, arc_controls, normalized_knots, error))
    if max_restriction_error > 1e-9:
        raise ValueError("Clamped CAD arcs changed the periodic spline")
    return pieces, {"control_count": control_count, "degree": 3, "fit": "cyclic least squares at source arclength parameters",
        "sampled_source_point_max_error_m": error, "periodic_seam_errors_derivatives_0_1_2": seam_errors,
        "cad_arc_restriction_max_error_m": max_restriction_error, "cad_arcs": 2,
        "source_points_m": source.tolist(), "periodic_knots": knots.tolist(), "independent_controls_m": controls.tolist(),
        "topology": "one unchanged closed interface represented by two shared CAD arcs; no gap or region merge"}


def shared_core(model: ModelInput, record: dict[str, object]) -> FrozenMesh:
    """Apply the same seed12.5/boundary10 subdivision and full Q2 as chapter1.3/65."""
    import gmsh
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add(f"chapter14_{model.dataset}_{model.model_id}")
        endpoints: dict[tuple[float, float], int] = {}
        boundary_tags: list[tuple[int, int]] = []
        for item in sequence(record["curves"]):
            curve = mapping(item)
            points = np.asarray(curve["control_points_m"], dtype=np.float64)
            tags: list[int] = []
            for index, point in enumerate(points):
                key = (round(float(point[0]), 9), round(float(point[1]), 9))
                endpoint = index in (0, len(points) - 1)
                if endpoint and key in endpoints:
                    tag = endpoints[key]
                else:
                    tag = int(gmsh.model.occ.addPoint(float(point[0]), float(point[1]), 0, 12.5))
                    if endpoint:
                        endpoints[key] = tag
                tags.append(tag)
            tag = int(number(curve["id"]))
            degree = int(number(curve["degree"]))
            if degree == 1:
                gmsh.model.occ.addLine(tags[0], tags[-1], tag=tag)
                if any(np.all(np.abs(points[:, axis] - value) < 1e-10) for axis in (0, 1) for value in (0., 700.)):
                    boundary_tags.append((tag, round(float(np.linalg.norm(points[-1] - points[0])) / 10) + 1))
            else:
                knots, counts = np.unique(np.asarray(curve["knots"], dtype=np.float64), return_counts=True)
                gmsh.model.occ.addBSpline(tags, tag=tag, degree=degree, knots=knots.tolist(), multiplicities=counts.tolist())
        surfaces: dict[int, tuple[int, float, float]] = {}
        for item in sequence(record["regions"]):
            region = mapping(item)
            loops = [gmsh.model.occ.addCurveLoop([int(number(x)) for x in sequence(loop)]) for loop in sequence(region["loops"])]
            surface = int(gmsh.model.occ.addPlaneSurface(loops))
            surfaces[surface] = (int(number(region["id"])), number(region["vp"]), number(region["vs"]))
        gmsh.model.occ.synchronize()
        cad_error = 0.0
        for item in sequence(record["curves"]):
            curve = mapping(item)
            sample = np.asarray(curve["sampled_points_m"], dtype=np.float64)
            curve_id = int(number(curve["id"]))
            low, high = gmsh.model.getParametrizationBounds(1, curve_id)
            parameters = np.linspace(float(low[0]), float(high[0]), len(sample))
            xyz = np.asarray(gmsh.model.getValue(1, curve_id, parameters.tolist())).reshape((-1, 3))
            cad_error = max(cad_error, float(np.max(np.linalg.norm(xyz[:, :2] - sample, axis=1))))
        if cad_error > 2e-8:
            raise ValueError(f"CAD curve deviates from the fitted spline: {cad_error}")
        record["cad_sample_max_difference_m"] = cad_error
        for tag, count in boundary_tags:
            gmsh.model.mesh.setTransfiniteCurve(tag, count)
        for name, value in {"General.NumThreads": 2, "Mesh.MeshSizeMin": 12.5, "Mesh.MeshSizeMax": 12.5,
            "Mesh.MeshSizeFromCurvature": 0, "Mesh.Algorithm": 6, "Mesh.RecombineAll": 0, "Mesh.SubdivisionAlgorithm": 1,
            "Mesh.SecondOrderIncomplete": 0, "Mesh.SecondOrderLinear": 0}.items():
            gmsh.option.setNumber(name, value)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        lookup = {int(tag): np.asarray(point[:2], dtype=np.float64) for tag, point in zip(all_tags, np.asarray(all_coords).reshape((-1, 3)), strict=True)}
        cell_rows: list[list[int]] = []
        metadata: list[tuple[int, float, float]] = []
        for surface, params in surfaces.items():
            types, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
            for cell_type, connectivity in zip(types, element_nodes, strict=True):
                if int(cell_type) != 10:
                    raise ValueError("Gmsh did not produce exclusively full Q2 quadrilaterals")
                for row in np.asarray(connectivity).reshape((-1, 9)):
                    cell_rows.append([int(tag) for tag in row[[0, 4, 1, 7, 8, 5, 3, 6, 2]]])
                    metadata.append(params)
        used = sorted({tag for row in cell_rows for tag in row})
        ids = {tag: index for index, tag in enumerate(used)}
        nodes = np.asarray([lookup[tag] for tag in used], dtype=np.float64)
        nodes[:, 1] -= 700
        cells = np.asarray([[ids[tag] for tag in row] for row in cell_rows], dtype=np.int64)
        properties = np.asarray(metadata, dtype=np.float64)
        core = FrozenMesh(nodes, cells, properties[:, 1], properties[:, 2], properties[:, 0].astype(np.int64), model.model_id, "curved")
        negative = jacobians(core, np.array([0.]))[:, 0, 0] < 0
        cells[negative] = cells[negative].reshape((-1, 3, 3))[:, :, ::-1].reshape((-1, 9))
        nodes[cells[:, 4]] = .5 * nodes[cells[:, [1, 3, 5, 7]]].sum(axis=1) - .25 * nodes[cells[:, [0, 2, 6, 8]]].sum(axis=1)
        actual_pairs = region_adjacency(core)
        expected_pairs = {tuple(sorted(int(number(v)) for v in sequence(mapping(item)["adjacent_regions"]))) for item in sequence(record["curves"])}
        expected_pairs = {pair for pair in expected_pairs if -1 not in pair}
        if actual_pairs != expected_pairs:
            raise ValueError(f"Material-region adjacency differs after meshing: missing={sorted(expected_pairs - actual_pairs)}, extra={sorted(actual_pairs - expected_pairs)}")
        if set(int(v) for v in np.unique(core.regions)) != {int(number(mapping(item)["id"])) for item in sequence(record["regions"])}:
            raise ValueError("A source region disappeared in meshing")
        return core
    finally:
        gmsh.finalize()


def region_adjacency(mesh: FrozenMesh) -> set[tuple[int, int]]:
    """Compare topological contacts, retaining short contacts and region identities."""
    _, inverse, counts, _ = edges(mesh)
    order = np.argsort(inverse)
    offsets = np.r_[0, np.cumsum(counts)]
    result: set[tuple[int, int]] = set()
    for edge in np.flatnonzero(counts == 2):
        owners = order[offsets[edge]:offsets[edge + 1]] // 4
        first, second = sorted((int(mesh.regions[owners[0]]), int(mesh.regions[owners[1]])))
        pair = (first, second)
        if pair[0] != pair[1]:
            result.add(pair)
    return result


def build_curved(model: ModelInput) -> GeometryResult:
    """Build one model, with explicit failure rather than topology repair or raster fallback."""
    layers = layer_sequence(model.vp, model.vs)
    if layers is not None:
        core_result = layered_core(model, layers)
    else:
        curves = shared_curves(model)
        core_result = GeometryResult(shared_core(model, curves), curves)
    mesh = extend(core_result.mesh)
    certificate = audit(mesh)
    return GeometryResult(mesh, {"schema": "chapter14.curved.v1", "dataset": model.dataset, "model_id": model.model_id,
        "input": model.provenance, "construction": core_result.report, "certificate": certificate,
        "fallback": "none; errors stop the model and must be reported", "algorithm_evidence": "chapter1.3: layered12404 and shared_CAD65"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ECVB", "EFVB"), required=True)
    parser.add_argument("--model", type=int, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.out.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    output.mkdir(parents=True, exist_ok=False)
    result = build_curved(read_model(args.dataset, args.model, args.source))
    mesh_path = output / "mesh_q2.npz"
    save_frozen(result.mesh, mesh_path)
    result.report["mesh_sha256"] = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
    (output / "geometry.json").write_text(json.dumps(result.report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result.report["certificate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
