"""Build topology-preserving all-quad Q2 meshes for ECVB 00065 and sample p4."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
from math import comb
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
import sys
from typing import Sequence

import gmsh  # type: ignore[import-untyped]  # Official Gmsh Python API has no typing metadata.
import numpy as np
from numpy.typing import NDArray

from scripts.dev_1_2_smoothing.pilot_geometry_plot import lagrange
from semgpu2.model import SGPETALineModel
from semgpu2.tools.mesh.box import rectangle_quad_fast

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ROOT = (DATA_12 / "model65") / "geometry"
Q2_REF = np.array([-1., 0., 1.])


@dataclass
class Q2Mesh:
    nodes: FloatArray
    cells: IntArray
    materials: FloatArray
    regions: IntArray


def jacobians(local: FloatArray, xi: FloatArray) -> FloatArray:
    w = lagrange(Q2_REF, xi)
    d = np.stack((xi-.5, -2*xi, xi+.5), axis=1)
    x = local.reshape((-1, 3, 3, 2))
    dx = np.einsum("ai,bj,eijc->eabc", w, d, x)
    dy = np.einsum("ai,bj,eijc->eabc", d, w, x)
    return dx[..., 0]*dy[..., 1] - dx[..., 1]*dy[..., 0]


def certify_jacobians(mesh: Q2Mesh) -> dict[str, object]:
    """Certify detJ>0 on every whole reference square via cubic Bernstein bounds."""
    t = np.linspace(0., 1., 4)
    basis = np.stack([comb(3, i)*t**i*(1-t)**(3-i) for i in range(4)], axis=1)
    inv = np.linalg.inv(basis)
    values = jacobians(mesh.nodes[mesh.cells], 2*t-1)
    control = np.einsum("ai,eij,bj->eab", inv, values, inv)
    transforms = []
    for offset in (0., .5):
        tt = offset+t/2
        transforms.append(inv @ np.stack([comb(3, i)*tt**i*(1-tt)**(3-i) for i in range(4)], axis=1))
    minimum = float("inf")
    max_depth = 0
    for element, c in enumerate(control):
        stack = [(c, 0)]
        while stack:
            patch, depth = stack.pop()
            lo = float(patch.min())
            if lo > 1e-9:
                minimum = min(minimum, lo)
                max_depth = max(max_depth, depth)
                continue
            if float(patch.max()) <= 0 or depth == 9:
                raise ValueError(f"Cannot certify positive full-Q2 detJ: element {element}, depth {depth}, bound {lo}")
            for left in transforms:
                for right in transforms:
                    stack.append((left @ patch @ right.T, depth+1))
    g, w = np.polynomial.legendre.leggauss(3)
    areas = np.einsum("eij,i,j->e", jacobians(mesh.nodes[mesh.cells], g), w, w)
    if abs(float(areas.sum())-490000.) > 1e-5:
        raise ValueError(f"Mesh does not cover the square: area={areas.sum()}")
    corners = mesh.nodes[mesh.cells[:, [0, 2, 8, 6]]]
    lengths = np.linalg.norm(corners-np.roll(corners, -1, axis=1), axis=2)
    central_shift = np.linalg.norm(mesh.nodes[mesh.cells[:, 4]]-corners.mean(axis=1), axis=1)
    return {"positive_jacobian_certificate": "tensor cubic Bernstein coefficients, recursively subdivided where necessary",
            "det_jacobian_certified_lower_bound": minimum, "bernstein_max_subdivision_depth": max_depth,
            "area_m2": float(areas.sum()), "area_by_region_m2": {str(int(r)): float(areas[mesh.regions == r].sum()) for r in np.unique(mesh.regions)},
            "sqrt_area_quantiles_m": np.quantile(np.sqrt(areas), [0, .01, .05, .5, .95, .99, 1]).tolist(),
            "edge_length_quantiles_m": np.quantile(lengths, [0, .01, .05, .5, .95, .99, 1]).tolist(),
            "q2_centre_shift_from_corner_average_max_m": float(central_shift.max())}


def raster_mesh(root: Path) -> Q2Mesh:
    with np.load(root / "model65.npz", allow_pickle=False) as data:
        vp, vs, regions = data["vp"], data["vs"], data["region_labels"]
    mesh = rectangle_quad_fast(sizes=(700., 700.), elements=(70, 70), pets=[SGPETALineModel("GLL", order=2)])
    nodes = np.asarray(mesh.nodes_coords+350., dtype=np.float64)
    cells = np.asarray(mesh.strata[1].p_nodes.reshape((-1, 9)), dtype=np.int64)
    centres = nodes[cells[:, 4]]
    rr = np.floor((700-centres[:, 1])/10).astype(np.int64)
    cc = np.floor(centres[:, 0]/10).astype(np.int64)
    return Q2Mesh(nodes, cells, np.column_stack((vp[rr, cc], vs[rr, cc])), regions[rr, cc])


def curved_mesh(root: Path, size: float, algorithm: int, recombination: int, subdivide: bool, triangle_subdivide: bool = False) -> Q2Mesh:
    record = json.loads((root / "curves.json").read_text(encoding="utf-8"))
    gmsh.initialize()
    try:
        gmsh.model.add("ECVB_00065_shared_interfaces")
        endpoint_tags: dict[tuple[float, float], int] = {}
        for curve in record["curves"]:
            points = np.asarray(curve["control_points_m"], dtype=np.float64)
            tags = []
            for index, point in enumerate(points):
                key = (float(point[0]), float(point[1]))
                endpoint = index in (0, len(points)-1)
                if endpoint and key in endpoint_tags:
                    tag = endpoint_tags[key]
                else:
                    tag = int(gmsh.model.occ.addPoint(*key, 0., size))
                    if endpoint:
                        endpoint_tags[key] = tag
                tags.append(tag)
            if curve["degree"] == 1:
                if len(tags) != 2:
                    raise ValueError("Expected simplified straight segment, not polyline fallback")
                gmsh.model.occ.addLine(tags[0], tags[-1], tag=curve["id"])
            else:
                knots, counts = np.unique(curve["knots"], return_counts=True)
                gmsh.model.occ.addBSpline(tags, tag=curve["id"], degree=curve["degree"], knots=knots.tolist(), multiplicities=counts.tolist())
        surfaces: dict[int, tuple[int, float, float]] = {}
        for region in record["regions"]:
            loops = [gmsh.model.occ.addCurveLoop(loop) for loop in region["loops"]]
            surface = int(gmsh.model.occ.addPlaneSurface(loops))
            surfaces[surface] = (int(region["id"]), float(region["vp"]), float(region["vs"]))
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("General.NumThreads", 2)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm", algorithm)
        gmsh.option.setNumber("Mesh.RecombinationAlgorithm", recombination)
        gmsh.option.setNumber("Mesh.RecombineAll", 0 if triangle_subdivide else 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1 if subdivide or triangle_subdivide else 0)
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 0)
        gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
        gmsh.model.mesh.generate(2)
        gmsh.model.mesh.setOrder(2)
        # Keep the CAD boundaries fixed, and preserve all nine Q2 nodes.
        all_tags, all_coords, _ = gmsh.model.mesh.getNodes()
        node_lookup = {int(tag): np.asarray(point[:2], dtype=np.float64) for tag, point in zip(all_tags, np.asarray(all_coords).reshape((-1, 3)), strict=True)}
        cells_list: list[list[int]] = []
        material_list: list[tuple[float, float]] = []
        region_list: list[int] = []
        for surface, (region_id, vp, vs) in surfaces.items():
            types, _, element_nodes = gmsh.model.mesh.getElements(2, surface)
            for element_type, connectivity in zip(types, element_nodes, strict=True):
                if int(element_type) != 10:
                    raise ValueError(f"Non-Q2-quad element type {element_type} in region {region_id}")
                for row in np.asarray(connectivity).reshape((-1, 9)):
                    cells_list.append([int(tag) for tag in row[[0, 4, 1, 7, 8, 5, 3, 6, 2]]])
                    material_list.append((vp, vs))
                    region_list.append(region_id)
        used = sorted({tag for row in cells_list for tag in row})
        ids = {tag: i for i, tag in enumerate(used)}
        nodes = np.asarray([node_lookup[tag] for tag in used], dtype=np.float64)
        cells = np.asarray([[ids[tag] for tag in row] for row in cells_list], dtype=np.int64)
        # Orient every reference element positively in physical (x,y).
        signed = jacobians(nodes[cells], np.array([0.]))[:, 0, 0]
        cells[signed < 0] = cells[signed < 0].reshape((-1, 3, 3))[:, :, ::-1].reshape((-1, 9))
        # Q2 Coons centre follows curved edge midpoints, rather than averaging corners.
        nodes[cells[:, 4]] = .5*nodes[cells[:, [1, 3, 5, 7]]].sum(axis=1) - .25*nodes[cells[:, [0, 2, 6, 8]]].sum(axis=1)
        for node in np.unique(cells[:, 4]):
            gmsh.model.mesh.setNode(used[int(node)], [float(nodes[node, 0]), float(nodes[node, 1]), 0.], [])
        gmsh.write(str(root / "curved_gmsh_q2.msh"))
        return Q2Mesh(nodes, cells, np.asarray(material_list), np.asarray(region_list, dtype=np.int64))
    finally:
        gmsh.finalize()


def write_spectral(mesh: Q2Mesh, root: Path, kind: str) -> dict[str, object]:
    ref = np.asarray(SGPETALineModel("GLL", order=4).p_nodes.coords[:, 0], dtype=np.float64)
    weights = lagrange(Q2_REF, ref)
    mapped = np.einsum("ai,bj,eijc->eabc", weights, weights, mesh.nodes[mesh.cells].reshape((-1, 3, 3, 2)))
    node_ids: dict[tuple[str, int, int, int], int] = {}
    coords: list[FloatArray] = []
    cells = np.empty((len(mesh.cells), 5, 5), dtype=np.int64)
    for e, q2 in enumerate(mesh.cells.reshape((-1, 3, 3))):
        for j in range(5):
            for i in range(5):
                if j in (0, 4) and i in (0, 4):
                    key = ("v", int(q2[j//2, i//2]), 0, 0)
                elif j in (0, 4) or i in (0, 4):
                    a, b, k = (int(q2[j//2, 0]), int(q2[j//2, 2]), i) if j in (0, 4) else (int(q2[0, i//2]), int(q2[2, i//2]), j)
                    key = ("e", min(a, b), max(a, b), k if a < b else 4-k)
                else:
                    key = ("c", e, j, i)
                point = mapped[e, j, i]
                if key not in node_ids:
                    node_ids[key] = len(coords)
                    coords.append(point)
                elif np.max(np.abs(coords[node_ids[key]]-point)) > 1e-9:
                    raise ValueError("Nonconforming Q2 edge: adjacent elements disagree")
                cells[e, j, i] = node_ids[key]
    edges: dict[tuple[int, int], list[tuple[int, IntArray]]] = defaultdict(list)
    for e, cell in enumerate(cells):
        for edge in (cell[0, :], cell[:, -1], cell[-1, ::-1], cell[::-1, 0]):
            edges[(min(int(edge[0]), int(edge[-1])), max(int(edge[0]), int(edge[-1])))].append((e, edge))
    boundary, owners = [], []
    for uses in edges.values():
        if len(uses) == 1:
            owner, edge = uses[0]
            boundary.append(edge)
            owners.append(owner)
        elif len(uses) != 2 or not np.array_equal(uses[0][1], uses[1][1][::-1]):
            raise ValueError("Non-manifold or misoriented shared edge")
    nodes = np.asarray(coords)
    boundary_array = np.asarray(boundary, dtype=np.int64)
    exterior = nodes[boundary_array]
    on_side = np.any(np.stack([np.all(np.abs(exterior[:, :, axis]-value) < 1e-8, axis=1) for axis in (0, 1) for value in (0., 700.)]), axis=0)
    if not np.all(on_side):
        raise ValueError("Crack: single-owner edge inside the physical square")
    vertex_count = len({int(v) for v in mesh.cells[:, [0, 2, 6, 8]].ravel()})
    if vertex_count-len(edges)+len(cells) != 1:
        raise ValueError("Mesh topology is not a disk")
    audit = certify_jacobians(mesh)
    material_b = mesh.materials[np.asarray(owners)]
    np.savez_compressed(root / f"{kind}_p4_mesh.npz", nodes=nodes, cells=cells.reshape((-1, 25)), boundary=boundary_array,
                        vp=mesh.materials[:, 0], vs=mesh.materials[:, 1], boundary_vp=material_b[:, 0], boundary_vs=material_b[:, 1], region_ids=mesh.regions)
    np.savez_compressed(root / f"{kind}_q2_mesh.npz", nodes=mesh.nodes, cells=mesh.cells, materials=mesh.materials, region_ids=mesh.regions)
    audit.update({"kind": kind, "elements": len(cells), "nodes_p4": len(nodes), "nodes_q2": len(mesh.nodes), "external_edges": len(boundary),
                  "euler_characteristic": 1, "conforming_edges": True, "geometry": "full Q2 with Coons central node, frozen and evaluated at p4 GLL nodes"})
    return audit


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-root", type=Path, default=ROOT)
    parser.add_argument("--size", type=float, default=25., help="Gmsh seed size in metres; 25 m gives final median sqrt(area) about 9 m after triangle subdivision")
    parser.add_argument("--algorithm", type=int, default=6)
    parser.add_argument("--recombination", type=int, default=3)
    parser.add_argument("--subdivide", action="store_true")
    parser.add_argument("--triangle-subdivide", action=argparse.BooleanOptionalAction, default=True, help="split a triangular seed mesh into three quads per triangle; size describes the seed")
    args = parser.parse_args(argv)
    root = args.case_root.resolve()
    raster = write_spectral(raster_mesh(root), root, "raster")
    curved = write_spectral(curved_mesh(root, args.size, args.algorithm, args.recombination, args.subdivide, args.triangle_subdivide), root, "curved")
    record = {"model_id": 65, "domain_m": [0., 0., 700., 700.], "order": 4, "geometry_order": 2, "target_h_m": 10., "gmsh_seed_size_m": args.size,
              "gmsh_version": gmsh.__version__, "gmsh_algorithm": args.algorithm, "gmsh_recombination": args.recombination,
              "triangle_subdivision": args.triangle_subdivide,
              "meshes": {"raster": raster, "curved": curved},
              "boundary_audit": {"raster": {"external_edges": raster["external_edges"], "conforming": True, "all_on_domain_square": True},
                                 "curved": {"external_edges": curved["external_edges"], "conforming": True, "all_on_domain_square": True}}}
    (root / "case.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
