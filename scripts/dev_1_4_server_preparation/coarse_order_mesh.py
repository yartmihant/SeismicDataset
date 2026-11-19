"""Independent 10m Q2 meshes and exact 2x2 reference refinement for p6/p7 tests."""
from __future__ import annotations
import argparse
from math import comb
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from semgpu2.model import SGPETALineModel
from semgpu2.model.mesh import SGMeshModel, SGMeshStratumData
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.mesh import FrozenMesh, basis, jacobians, edges, structured_cells, save_frozen
from scripts.dev_1_3_point_source.run import file_record, write_json
from scripts.dev_1_4_server_preparation.curved_mesh import layered_core, layer_sequence, shared_curves, region_adjacency
from scripts.dev_1_4_server_preparation.model_input import ModelInput, read_model
from scripts.reporting import sequence, mapping, number
from scripts.project_paths import PROJECT_ROOT
FloatArray=NDArray[np.float64]
IntArray=NDArray[np.int64]


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
                    tag = int(gmsh.model.occ.addPoint(float(point[0]), float(point[1]), 0, 25.))
                    if endpoint:
                        endpoints[key] = tag
                tags.append(tag)
            tag = int(number(curve["id"]))
            degree = int(number(curve["degree"]))
            if degree == 1:
                gmsh.model.occ.addLine(tags[0], tags[-1], tag=tag)
                if any(np.all(np.abs(points[:, axis] - value) < 1e-10) for axis in (0, 1) for value in (0., 700.)):
                    boundary_tags.append((tag, round(float(np.linalg.norm(points[-1] - points[0])) / 20) + 1))
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
        for name, value in {"General.NumThreads": 2, "Mesh.MeshSizeMin": 25., "Mesh.MeshSizeMax": 25.,
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
            for k in range(18):
                xx = np.sort(x0 + sign*((175/18)*(k + np.array([0, .5, 1]))))
                x, y = np.meshgrid(xx, yy)
                add(np.stack((x, y), axis=-1), owner)
            if yy[0] < -700+1e-8:
                corner_owners[sign] = owner
        elif np.allclose(edge[:, 1], -700, atol=1e-8, rtol=0):
            xx = np.sort(edge[:, 0])
            for k in range(18):
                yy = -700-(175/18)*(k+np.array([1, .5, 0]))
                x, y = np.meshgrid(xx, yy)
                add(np.stack((x, y), axis=-1), owner)
        else:
            raise ValueError("Core contains an internal crack")
    for sign, owner in corner_owners.items():
        for j in range(18):
            for i in range(18):
                xx = (0 if sign == -1 else 700) + sign*((175/18)*(i+np.array([0, .5, 1])))
                yy = -700-(175/18)*(j+np.array([1, .5, 0]))
                x, y = np.meshgrid(np.sort(xx), yy)
                add(np.stack((x, y), axis=-1), owner)
    if set(corner_owners) != {-1, 1}:
        raise ValueError("Missing bottom corner material")
    return FrozenMesh(np.asarray(nodes), np.asarray(cells, dtype=np.int64), np.asarray(vp), np.asarray(vs), np.asarray(regions, dtype=np.int64), core.model_id, core.kind)

def audit(mesh: FrozenMesh, top_size: float = 10.) -> dict[str, object]:
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
    if len(core_top) != round(700/top_size) or not np.allclose(lengths, top_size, atol=1e-8, rtol=0):
        raise ValueError("Unexpected top partition")
    g, w = np.polynomial.legendre.leggauss(3)
    area = np.einsum("eij,i,j->e", jacobians(mesh, g), w, w)
    if abs(float(area.sum())-1050*875)>1e-5:
        raise ValueError("Incorrect total area")
    return {"model_id": mesh.model_id, "kind": mesh.kind, "elements": len(mesh.cells), "q2_nodes": len(mesh.nodes),
            "area_m2": float(area.sum()), "det_jacobian_certified_min_m2": minimum, "bernstein_max_depth": max_depth,
            "euler_characteristic": 1, "conforming": True, "core_top_edges": len(core_top), "core_top_edge_m": top_size,
            "external_edges": len(exterior), "area_by_region_m2": {str(int(r)): float(area[mesh.regions == r].sum()) for r in np.unique(mesh.regions)},
            "sqrt_area_quantiles_m": np.quantile(np.sqrt(area), [0, .01, .5, .99, 1]).tolist()}

def build_sem(frozen: FrozenMesh, order: int) -> MeshCase:
    """Вложить одну Q2-карту в GLL; общие узлы объединяются по топологии рёбер."""
    if order not in (3, 4, 5, 6, 7):
        raise ValueError("Coarse experiment supports p3..p7")
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



def coarse_core(model: ModelInput) -> FrozenMesh:
    layers=layer_sequence(model.vp,model.vs)
    if layers is None:
        return shared_core(model,shared_curves(model))
    fine=layered_core(model,layers).mesh
    nx=140;ny=len(fine.cells)//nx
    assert ny%2==0
    coords=fine.nodes.reshape(2*ny+1,2*nx+1,2)[::2,::2].reshape(-1,2)
    props=[]
    for values in (fine.vp,fine.vs,fine.regions):
        a=values.reshape(ny,nx)
        assert np.array_equal(a,np.repeat(np.repeat(a[::2,::2],2,axis=0),2,axis=1))
        props.append(a[::2,::2].ravel())
    return FrozenMesh(coords,structured_cells(nx//2,ny//2),props[0].astype(np.float64),props[1].astype(np.float64),props[2].astype(np.int64),model.model_id,'curved')


def refine(mesh: FrozenMesh) -> FrozenMesh:
    # Evaluate the SAME Q2 map at quarter positions, so geometry is exactly nested.
    t=np.linspace(-1,1,5)
    mapped=np.einsum('ai,bj,eijc->eabc',basis(t),basis(t),mesh.nodes[mesh.cells].reshape(-1,3,3,2))
    rows=np.stack([mapped[:,j:j+3,i:i+3].reshape(-1,9,2) for j in (0,2) for i in (0,2)],axis=1).reshape(-1,9,2)
    flat=rows.reshape(-1,2);_,first,inverse=np.unique(np.round(flat,9),axis=0,return_index=True,return_inverse=True)
    coords=flat[first]
    assert np.max(np.abs(coords[inverse]-flat))<1e-8
    return FrozenMesh(coords,inverse.reshape(-1,9),np.repeat(mesh.vp,4),np.repeat(mesh.vs,4),np.repeat(mesh.regions,4),mesh.model_id,mesh.kind)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',type=int,choices=(65,12404),required=True);p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    root=args.root.resolve();root.relative_to(PROJECT_ROOT/'data/dev_1_4_server_preparation')
    base=root/f'{args.model:05d}';base.mkdir(parents=True,exist_ok=False)
    coarse=extend(coarse_core(read_model('ECVB',args.model)))
    fine=refine(coarse)
    for name,mesh,h in [('coarse',coarse,10.),('refined',fine,5.)]:
        dest=base/name;dest.mkdir();r=audit(mesh,h);save_frozen(mesh,dest/'mesh_q2.npz');r['mesh']=file_record(dest/'mesh_q2.npz');r['construction']='coarse CAD, 18 padding elements of 175/18m; refined is exact 2x2 restriction of coarse Q2';write_json(dest/'geometry.json',r)
        print(args.model,name,len(mesh.cells),r['sqrt_area_quantiles_m'],flush=True)


if __name__=='__main__':
    main()
