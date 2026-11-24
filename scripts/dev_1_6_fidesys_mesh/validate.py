"""Проверки реального FC-экспорта по связности, геометрии QUAD8 и CAD."""
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from fc_model import FCModel
from scripts.dev_1_3_point_source.mesh import FrozenMesh, IntArray, edges, jacobians
from scripts.dev_1_4_server_preparation.curved_mesh import region_adjacency
from scripts.dev_1_6_p67s_showcase.mesh import InvalidJacobian, certify_jacobians
from scripts.dev_1_6_p67s_showcase.fc_mesh_quality import Sample, inspect


@dataclass
class RefineRegions(Exception):
    region_ids: list[int]
    element_ids: list[int]
    reason: str

    def __str__(self) -> str:
        return f"{self.reason}; FC elements {self.element_ids}, CAD regions {self.region_ids}"


def read_core(path: Path, cad: dict, *, require_fc_materials: bool = False) -> tuple[FrozenMesh, IntArray]:
    """Точное тензорное представление FC QUAD8 и исходные ID элементов в том же порядке."""
    document = FCModel.load(str(path))
    m = document.mesh
    if set(m.elements) != {"QUAD8"}:
        raise ValueError(f"Expected planar QUAD8, got {list(m.elements)}")
    coordinates = m.nodes_xyz
    if not np.isfinite(coordinates).all() or np.any(coordinates[:,2]!=0):
        raise ValueError("Invalid planar node coordinates")
    ids = {int(n):i for i,n in enumerate(m.nodes_ids)}
    elements = list(m.elements["QUAD8"].values())
    connectivity = np.array([[ids[n] for n in e.nodes] for e in elements])
    regions = np.array([e.block for e in elements])
    if require_fc_materials:
        from scripts.dev_1_6_fidesys_mesh.materials import read_materials
        materials = read_materials(document, cad)
    else:
        materials = {r["id"]:(r["vp"],r["vs"]) for r in cad["regions"]}
    if set(regions)!=set(materials) or len(np.unique(connectivity))!=len(coordinates):
        raise ValueError("Missing region or unused nodes")
    props = np.array([materials[i] for i in regions])
    # The serendipity QUAD8 map has this exact value at the ninth tensor node.
    corners = connectivity[:,:4]
    mids = connectivity[:,4:]
    center = .5*coordinates[mids,:2].sum(axis=1) - .25*coordinates[corners,:2].sum(axis=1)
    nodes = np.vstack((coordinates[:,:2],center))
    cells = np.column_stack((connectivity,np.arange(len(center))+len(coordinates)))[:,[0,4,1,7,8,5,3,6,2]]
    frozen = FrozenMesh(nodes,cells,props[:,0],props[:,1],regions,cad["model"],"fidesys_quad8")
    return frozen, np.asarray([e.id for e in elements], dtype=np.int64)


def validate(path: Path, cad: dict, min_scaled_jacobian: float = 0.05) -> dict:
    frozen, element_ids = read_core(path, cad)
    nodes, cells, regions = frozen.nodes, frozen.cells, frozen.regions
    try:
        det,depth = certify_jacobians(frozen)
    except InvalidJacobian as exc:
        # Also collect other inverted cells before retrying. The final acceptance
        # always uses the certificate over the complete reference square.
        bad = np.flatnonzero(jacobians(frozen,np.linspace(-1,1,7)).min(axis=(1,2))<=0)
        indices = np.union1d(bad,[exc.element])
        blocks = set(regions[indices])
        raise RefineRegions(sorted(map(int,blocks)),element_ids[indices].tolist(),str(exc)) from exc
    expected = {tuple(sorted(c["adjacent_regions"])) for c in cad["curves"] if -1 not in c["adjacent_regions"]}
    if region_adjacency(frozen)!=expected:
        raise ValueError("Changed block adjacency")
    directed, inverse, _, exterior = edges(frozen)
    mid_ids = np.empty(int(inverse.max())+1,dtype=np.int64)
    mid_ids[inverse] = directed[:,1]
    if not np.array_equal(mid_ids[inverse],directed[:,1]):
        raise ValueError("Nonconforming QUAD8 midpoint IDs")
    boundary = nodes[directed[exterior]]
    top = boundary[np.all(np.abs(boundary[:,:,1])<1e-8,axis=1)]
    if len(top)!=140 or np.max(np.abs(top[:,1,0]-(top[:,0,0]+top[:,2,0])/2))>1e-8:
        raise ValueError("Invalid quadratic top partition")
    sample = Sample(nodes[cells],cells[:,[0,2,8,6]],regions,element_ids,{})
    aligned = [dict(c,control_points_m=(np.array(c["control_points_m"])+[0.,700.]).tolist()) for c in cad["curves"]]
    report,scaled = inspect(sample,aligned)
    worst = int(np.argmin(scaled))
    if scaled[worst] < min_scaled_jacobian:
        raise RefineRegions([int(regions[worst])],[int(element_ids[worst])],
                            f"Minimum sampled scaled Jacobian {scaled[worst]:.8g} below {min_scaled_jacobian}")
    # A misplaced midpoint may leave a curved interface effectively first-order.
    if report["interface_midpoint_max_error_m"] > 1e-4:
        raise ValueError(f'Interface midpoint misses CAD: {report["interface_midpoint_max_error_m"]} m')
    return dict(report,nodes=len(nodes)-len(cells),blocks=len(np.unique(regions)),element_type="QUAD8",
                certified_min_det_J_m2=det,bernstein_subdivision_depth=depth)
