"""Расширенные замороженные Q2-сетки главы 1.3 и их вложение в GLL p3/p4/p5."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from math import comb
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from semgpu2.model import SGPETALineModel
from semgpu2.model.mesh import SGMeshModel, SGMeshStratumData
from scripts.project_paths import DATA_12, PROJECT_ROOT
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase, quadratic_values
from scripts.reporting import mapping, sequence, number

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ROOT = PROJECT_ROOT / "data/dev_1_3_point_source/meshes_v1"
EDGE_LOCAL = np.array([[0, 1, 2], [2, 5, 8], [8, 7, 6], [6, 3, 0]], dtype=np.int64)


@dataclass(frozen=True)
class FrozenMesh:
    nodes: FloatArray
    cells: IntArray
    vp: FloatArray
    vs: FloatArray
    regions: IntArray
    model_id: int
    kind: str


def basis(x: FloatArray) -> FloatArray:
    return np.column_stack((x * (x - 1) / 2, 1 - x*x, x * (x + 1) / 2))


def jacobians(mesh: FrozenMesh, x: FloatArray) -> FloatArray:
    weights = basis(x)
    derivative = np.column_stack((x - .5, -2*x, x + .5))
    local = mesh.nodes[mesh.cells].reshape((-1, 3, 3, 2))
    dx = np.einsum("ai,bj,eijc->eabc", weights, derivative, local)
    dy = np.einsum("ai,bj,eijc->eabc", derivative, weights, local)
    return dx[..., 0]*dy[..., 1] - dx[..., 1]*dy[..., 0]


def structured_cells(nx: int, ny: int) -> IntArray:
    start = (2*np.arange(ny)[:, None]*(2*nx+1) + 2*np.arange(nx)[None, :]).ravel()
    offsets = (np.arange(3)[:, None]*(2*nx+1)+np.arange(3)[None, :]).ravel()
    return np.asarray(start[:, None]+offsets, dtype=np.int64)


def raster_core(model_id: int) -> FrozenMesh:
    path = DATA_12 / ("pilot_geometry/selected_model.npz" if model_id == 12404 else "model65/geometry/model65.npz")
    with np.load(path, allow_pickle=False) as archive:
        vp = np.asarray(archive["vp"], dtype=np.float64)
        vs = np.asarray(archive["vs"], dtype=np.float64)
        region_name = "material_labels" if model_id == 12404 else "region_labels"
        regions = np.asarray(archive[region_name], dtype=np.int64)
    x, y = np.meshgrid(np.linspace(0, 700, 281), np.linspace(-700, 0, 281))
    def refined(values: FloatArray) -> FloatArray:
        return np.repeat(np.repeat(values[::-1], 2, axis=0), 2, axis=1).ravel()
    return FrozenMesh(np.column_stack((x.ravel(), y.ravel())), structured_cells(140, 140),
                      refined(vp), refined(vs), refined(regions.astype(np.float64)).astype(np.int64), model_id, "raster")


def layered_core() -> FrozenMesh:
    with np.load(DATA_12 / "pilot_geometry/frozen_q2_geometry.npz", allow_pickle=False) as archive:
        depths = np.asarray(archive["depth_q2"], dtype=np.float64)
        rows = np.asarray(archive["rows_topdown"], dtype=np.int64)
        materials = np.asarray(archive["materials_topdown"], dtype=np.float64)
    xs = np.linspace(0, 700, 281)
    values = quadratic_values(depths, xs)
    offsets = np.r_[0, np.cumsum(rows)]
    logical = np.linspace(0, rows.sum(), int(rows.sum())*4+1)
    layer = np.minimum(np.searchsorted(offsets, logical, side="right")-1, len(rows)-1)
    fraction = (logical-offsets[layer])/rows[layer]
    y = -((1-fraction[:, None])*values[layer]+fraction[:, None]*values[layer+1])
    coords = np.stack((np.broadcast_to(xs, y.shape), y), axis=-1)[::-1].reshape((-1, 2))
    region = np.repeat(np.arange(len(rows)), 2*rows)[::-1]
    material = np.repeat(materials[region], 140, axis=0)
    return FrozenMesh(coords, structured_cells(140, int(2*rows.sum())), material[:, 0], material[:, 1], np.repeat(region, 140), 12404, "curved")


def curved65_core() -> FrozenMesh:
    """Сохранить CAD-интерфейсы №65; seed10м на внешних гранях делится пополам."""
    import gmsh
    record = mapping(json.loads((DATA_12 / "model65/geometry/curves.json").read_text()))
    gmsh.initialize()
    try:
        gmsh.model.add("chapter13_65")
        endpoints: dict[tuple[float, float], int] = {}
        boundary_tags: list[tuple[int, int]] = []
        for item in sequence(record["curves"]):
            curve = mapping(item)
            points = np.asarray(curve["control_points_m"], dtype=np.float64)
            tags: list[int] = []
            for i, point in enumerate(points):
                key = (float(point[0]), float(point[1]))
                endpoint = i in (0, len(points)-1)
                if endpoint and key in endpoints:
                    tag = endpoints[key]
                else:
                    tag = int(gmsh.model.occ.addPoint(key[0], key[1], 0, 12.5))
                    if endpoint:
                        endpoints[key] = tag
                tags.append(tag)
            tag = int(number(curve["id"]))
            degree = int(number(curve["degree"]))
            if degree == 1:
                gmsh.model.occ.addLine(tags[0], tags[-1], tag=tag)
                if any(np.all(np.abs(points[:, axis]-value) < 1e-10) for axis in (0, 1) for value in (0., 700.)):
                    length = float(np.linalg.norm(points[-1]-points[0]))
                    boundary_tags.append((tag, round(length/10)+1))
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
        for tag, count in boundary_tags:
            gmsh.model.mesh.setTransfiniteCurve(tag, count)
        for option_name, value in {"General.NumThreads": 2, "Mesh.MeshSizeMin": 12.5, "Mesh.MeshSizeMax": 12.5,
                           "Mesh.MeshSizeFromCurvature": 0, "Mesh.Algorithm": 6, "Mesh.RecombineAll": 0,
                           "Mesh.SubdivisionAlgorithm": 1, "Mesh.SecondOrderIncomplete": 0,
                           "Mesh.SecondOrderLinear": 0}.items():
            gmsh.option.setNumber(option_name, value)
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
        ids = {tag: i for i, tag in enumerate(used)}
        nodes = np.asarray([lookup[tag] for tag in used], dtype=np.float64)
        nodes[:, 1] -= 700
        cells = np.asarray([[ids[tag] for tag in row] for row in cell_rows], dtype=np.int64)
        props = np.asarray(metadata, dtype=np.float64)
        result = FrozenMesh(nodes, cells, props[:, 1], props[:, 2], props[:, 0].astype(np.int64), 65, "curved")
        negative = jacobians(result, np.array([0.]))[:, 0, 0] < 0
        cells[negative] = cells[negative].reshape((-1, 3, 3))[:, :, ::-1].reshape((-1, 9))
        nodes[cells[:, 4]] = .5*nodes[cells[:, [1, 3, 5, 7]]].sum(axis=1)-.25*nodes[cells[:, [0, 2, 6, 8]]].sum(axis=1)
        return result
    finally:
        gmsh.finalize()


def edges(mesh: FrozenMesh) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """Return directed Q2 edges, unique inverse, multiplicities and exterior owners."""
    directed = mesh.cells[:, EDGE_LOCAL].reshape((-1, 3))
    _, inverse, counts = np.unique(np.sort(directed[:, [0, 2]], axis=1), axis=0, return_inverse=True, return_counts=True)
    exterior = np.flatnonzero(counts[inverse] == 1)
    return directed, inverse, counts, exterior


def extend(core: FrozenMesh) -> FrozenMesh:
    """Экструзия каждого внешнего Q2-ребра и четыре угловых/нижних блока."""
    directed, _, _, external = edges(core)
    nodes = core.nodes.tolist()
    lookup = {(round(x, 9), round(y, 9)): i for i, (x, y) in enumerate(nodes)}
    cells = core.cells.tolist()
    vp, vs, regions = core.vp.tolist(), core.vs.tolist(), core.regions.tolist()
    def add(local: FloatArray, owner: int) -> None:
        row: list[int] = []
        for point in local.reshape((-1, 2)):
            key = (round(float(point[0]), 9), round(float(point[1]), 9))
            if key not in lookup:
                lookup[key] = len(nodes)
                nodes.append([float(point[0]), float(point[1])])
            row.append(lookup[key])
        cells.append(row)
        vp.append(float(core.vp[owner])); vs.append(float(core.vs[owner])); regions.append(int(core.regions[owner]))
    corner_owners: dict[int, int] = {}
    for index in external:
        owner = int(index//4)
        edge = core.nodes[directed[index]]
        if np.allclose(edge[:, 1], 0, atol=1e-8, rtol=0):
            continue
        if np.allclose(edge[:, 0], 0, atol=1e-8, rtol=0) or np.allclose(edge[:, 0], 700, atol=1e-8, rtol=0):
            x0 = float(edge[0, 0]); sign = -1 if abs(x0) < 1e-8 else 1
            yy = np.sort(edge[:, 1])
            for k in range(35):
                xx = np.sort(x0 + sign*(5*k + np.array([0, 2.5, 5])))
                x, y = np.meshgrid(xx, yy)
                add(np.stack((x, y), axis=-1), owner)
            if yy[0] < -700+1e-8:
                corner_owners[sign] = owner
        elif np.allclose(edge[:, 1], -700, atol=1e-8, rtol=0):
            xx = np.sort(edge[:, 0])
            for k in range(35):
                yy = -700-5*k-np.array([5, 2.5, 0])
                x, y = np.meshgrid(xx, yy)
                add(np.stack((x, y), axis=-1), owner)
        else:
            raise ValueError("Core contains an internal crack")
    for sign, owner in corner_owners.items():
        for j in range(35):
            for i in range(35):
                xx = (0 if sign == -1 else 700) + sign*(5*i+np.array([0, 2.5, 5]))
                yy = -700-5*j-np.array([5, 2.5, 0])
                x, y = np.meshgrid(np.sort(xx), yy)
                add(np.stack((x, y), axis=-1), owner)
    if set(corner_owners) != {-1, 1}:
        raise ValueError("Missing bottom corner material")
    return FrozenMesh(np.asarray(nodes), np.asarray(cells, dtype=np.int64), np.asarray(vp), np.asarray(vs), np.asarray(regions, dtype=np.int64), core.model_id, core.kind)


def audit(mesh: FrozenMesh) -> dict[str, object]:
    """Сертификат detJ на всей Q2-карте, конформность, площадь и верхняя разбивка."""
    t = np.linspace(0, 1, 4)
    bernstein = np.stack([comb(3, i)*t**i*(1-t)**(3-i) for i in range(4)], axis=1)
    inv = np.linalg.inv(bernstein)
    control = np.einsum("ai,eij,bj->eab", inv, jacobians(mesh, 2*t-1), inv)
    transforms = []
    for offset in (0., .5):
        tt = offset+t/2
        transforms.append(inv@np.stack([comb(3, i)*tt**i*(1-tt)**(3-i) for i in range(4)], axis=1))
    minimum = float(control.min()); max_depth = 0
    if minimum <= 1e-10:
        minimum = float("inf")
        for element, coefficient in enumerate(control):
            stack = [(coefficient, 0)]
            while stack:
                patch, depth = stack.pop()
                if float(patch.min()) > 1e-10:
                    minimum = min(minimum, float(patch.min())); max_depth = max(max_depth, depth)
                elif float(patch.max()) <= 0 or depth == 9:
                    raise ValueError(f"Nonpositive/unverified Jacobian element {element}, bound {patch.min()}")
                else:
                    stack.extend((a@patch@b.T, depth+1) for a in transforms for b in transforms)
    directed, inverse, counts, exterior = edges(mesh)
    if np.any(counts > 2):
        raise ValueError("Nonmanifold edge")
    middle_by_edge = np.full(len(counts), -1, dtype=np.int64)
    middle_by_edge[inverse] = directed[:, 1]
    if not np.array_equal(middle_by_edge[inverse], directed[:, 1]):
        raise ValueError("Nonconforming Q2 edge midpoint")
    # Two uses must carry opposite orientation, hence identical ordered IDs cannot repeat.
    signs = np.where(directed[:, 0] < directed[:, 2], 1, -1)
    orientation = np.bincount(inverse, weights=signs)
    if np.any(orientation[counts == 2] != 0):
        raise ValueError("Misoriented neighbouring cells")
    boundary = mesh.nodes[directed[exterior]]
    side = np.stack([np.all(np.abs(boundary[:, :, axis]-value)<1e-8, axis=1) for axis, value in ((0, -175), (0, 875), (1, -875), (1, 0))])
    if not side.any(axis=0).all():
        raise ValueError("Internal crack after extension")
    vertex_count = len(np.unique(mesh.cells[:, [0, 2, 6, 8]]))
    if vertex_count-len(counts)+len(mesh.cells) != 1:
        raise ValueError("Mesh is not a disk")
    top = boundary[side[-1]]
    core_top = top[(top[:, :, 0].min(axis=1)>=-1e-8)&(top[:, :, 0].max(axis=1)<=700+1e-8)]
    lengths = np.abs(core_top[:, -1, 0]-core_top[:, 0, 0])
    if len(core_top) != 140 or not np.allclose(lengths, 5, atol=1e-8, rtol=0):
        raise ValueError("Top is not exactly 140x5m")
    g, w = np.polynomial.legendre.leggauss(3)
    area = np.einsum("eij,i,j->e", jacobians(mesh, g), w, w)
    if abs(float(area.sum())-1050*875)>1e-5:
        raise ValueError("Incorrect total area")
    return {"model_id": mesh.model_id, "kind": mesh.kind, "elements": len(mesh.cells), "q2_nodes": len(mesh.nodes),
            "area_m2": float(area.sum()), "det_jacobian_certified_min_m2": minimum, "bernstein_max_depth": max_depth,
            "euler_characteristic": 1, "conforming": True, "core_top_edges": 140, "core_top_edge_m": 5,
            "external_edges": len(exterior), "area_by_region_m2": {str(int(r)): float(area[mesh.regions == r].sum()) for r in np.unique(mesh.regions)},
            "sqrt_area_quantiles_m": np.quantile(np.sqrt(area), [0, .01, .5, .99, 1]).tolist()}


def save_frozen(mesh: FrozenMesh, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, nodes=mesh.nodes, cells=mesh.cells, vp=mesh.vp, vs=mesh.vs,
                            regions=mesh.regions, model_id=mesh.model_id, kind=mesh.kind)


def load_frozen(path: Path) -> FrozenMesh:
    with np.load(path, allow_pickle=False) as data:
        return FrozenMesh(np.asarray(data["nodes"], dtype=np.float64), np.asarray(data["cells"], dtype=np.int64),
                          np.asarray(data["vp"], dtype=np.float64), np.asarray(data["vs"], dtype=np.float64),
                          np.asarray(data["regions"], dtype=np.int64), int(data["model_id"]), str(data["kind"]))


def build_sem(frozen: FrozenMesh, order: int) -> MeshCase:
    """Вложить одну Q2-карту в GLL; общие узлы объединяются по топологии рёбер."""
    if order not in (3, 4, 5):
        raise ValueError("Chapter 1.3 supports p3/p4/p5")
    line = SGPETALineModel("GLL", order=order)
    ref = np.asarray(line.p_nodes.coords[:, 0], dtype=np.float64)
    mapped = np.einsum("ai,bj,eijc->eabc", basis(ref), basis(ref), frozen.nodes[frozen.cells].reshape((-1, 3, 3, 2)))
    directed, inverse, counts, exterior = edges(frozen)
    vertices = np.unique(frozen.cells[:, [0, 2, 6, 8]])
    vertex_map = np.full(len(frozen.nodes), -1, dtype=np.int64)
    vertex_map[vertices] = np.arange(len(vertices))
    total = len(vertices)+len(counts)*(order-1)+len(frozen.cells)*(order-1)**2
    cells = np.empty((len(frozen.cells), order+1, order+1), dtype=np.int64)
    for j, qj in ((0, 0), (order, 2)):
        for i, qi in ((0, 0), (order, 2)):
            cells[:, j, i] = vertex_map[frozen.cells.reshape((-1, 3, 3))[:, qj, qi]]
    edge_ids = len(vertices)+inverse[:, None]*(order-1)+np.arange(order-1)[None, :]
    reverse = directed[:, 0] > directed[:, 2]
    edge_ids[reverse] = edge_ids[reverse, ::-1]
    edge_ids = edge_ids.reshape((-1, 4, order-1))
    cells[:, 0, 1:-1] = edge_ids[:, 0]
    cells[:, 1:-1, -1] = edge_ids[:, 1]
    cells[:, -1, -2:0:-1] = edge_ids[:, 2]
    cells[:, -2:0:-1, 0] = edge_ids[:, 3]
    cells[:, 1:-1, 1:-1] = (len(vertices)+len(counts)*(order-1)+np.arange(len(frozen.cells)*(order-1)**2)).reshape((-1, order-1, order-1))
    coords = np.empty((total, 2), dtype=np.float64)
    coords[cells.ravel()] = mapped.reshape((-1, 2))
    if not np.allclose(coords[cells], mapped, atol=1e-8, rtol=0):
        raise ValueError("Adjacent Q2 geometries disagree")
    all_edges = np.stack((cells[:, 0, :], cells[:, :, -1], cells[:, -1, ::-1], cells[:, ::-1, 0]), axis=1).reshape((-1, order+1))
    boundary = all_edges[exterior]
    owners = exterior//4
    tags = np.zeros((total, 3), dtype=np.int64)
    tags[:, 0] = 2
    tags[:len(vertices), 0] = 0
    tags[len(vertices):len(vertices)+len(counts)*(order-1), 0] = 1
    incidence = np.bincount(boundary.ravel(), minlength=total)
    tags[:, 1] = 2-np.minimum(incidence, 2)
    count, boundary_count = len(cells), len(boundary)
    strata = {
        1: SGMeshStratumData((2, 2, 2), cells.ravel(), np.arange(count+1, dtype=np.int64)*(order+1)**2,
                            np.zeros(2*count, dtype=np.int32), np.arange(count+1, dtype=np.int64)*2, np.ones(count, dtype=np.int64)),
        2: SGMeshStratumData((2, 1, 1), boundary.ravel(), np.arange(boundary_count+1, dtype=np.int64)*(order+1),
                            np.zeros(boundary_count, dtype=np.int32), np.arange(boundary_count+1, dtype=np.int64), np.zeros(boundary_count, dtype=np.int64)),
    }
    mesh = SGMeshModel(coords, tags, [line], strata)
    meta: dict[str, object] = {"model_id": frozen.model_id, "kind": frozen.kind, "order": order,
                               "nodes": total, "elements": count, "geometry": "frozen_full_Q2", "coordinate_convention": "y_up_surface_0"}
    return MeshCase(mesh, frozen.vp, frozen.vs, frozen.vp[owners], frozen.vs[owners], meta)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=int, choices=(65, 12404), required=True)
    parser.add_argument("--kind", choices=("raster", "curved"), required=True)
    parser.add_argument("--out", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    path = args.out / f"{args.model:05d}_{args.kind}_q2.npz"
    if path.exists():
        raise FileExistsError(path)
    core = raster_core(args.model) if args.kind == "raster" else layered_core() if args.model == 12404 else curved65_core()
    mesh = extend(core)
    record = audit(mesh)
    save_frozen(mesh, path)
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    record["construction"] = "exact_2x2_refinement_of_frozen_Q2" if args.model == 12404 else "raster_2x2" if args.kind == "raster" else "new_shared_CAD_triangle_seed12.5_boundary10_quad_subdivision_Q2"
    path.with_suffix(".json").write_text(json.dumps(record, ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
