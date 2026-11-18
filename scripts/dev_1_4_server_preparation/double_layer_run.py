"""Удвоенный слой: растянутый квадратный профиль и контроль переноса ABC."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
from math import comb
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter
import traceback
from typing import Literal, cast

import numpy as np
from scipy.spatial import cKDTree
from numpy.typing import NDArray

from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.mesh import FrozenMesh, build_sem, load_frozen, edges, jacobians
from scripts.dev_1_3_point_source.physics import ExperimentSpec, inclusive_times, damping_alpha, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import file_record, TimedResult
from scripts.dev_1_2_smoothing.pilot_stability import tensor_basis
from scripts.dev_1_4_server_preparation.no_padding_run import CoreSpec, validate_core
from scripts.dev_1_4_server_preparation.surface_runner import (
    ALPHA_MAX_S_INV, SurfaceSpec, SurfaceRecorder, make_cu_model, receiver_nodes, _check_write, _write_metadata, validate_stability,
)
from scripts.dev_1_4_server_preparation import velocity_fields_run as fields
from scripts.dev_1_4_server_preparation.velocity_fields_run import Variant, FieldRecorder, stream_record
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json

ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation/double_layer_3060_v1"
MATERIAL_MODEL = "homogeneous_reference_toplayer"
VP_M_S = 4087.0
VS_M_S = 2431.62036
DENSITY_KG_M3 = 1000.0
PHYSICAL_MODEL: dict[str, object] = {"vp_m_s": VP_M_S, "vs_m_s": VS_M_S,
    "density_kg_m3": DENSITY_KG_M3, "homogeneous": True}


PADDING_M = 350.0
Profile = Literal["stretched350", "fixed175plateau"]
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def profile_record(profile: Profile) -> dict[str, object]:
    return {"name": profile, "ramp_width_m": 350. if profile == "stretched350" else 175.,
        "plateau_after_ramp": profile == "fixed175plateau", "alpha_max_s_inv": ALPHA_MAX_S_INV,
        "formula": "alpha_max*(min(dx/L,1)^2+min(dy/L,1)^2)",
        "convention": "C=alpha*M; sum x/y contributions", "alpha_peak_selection": "Same peak as the175m reference; not rescaled by350m thickness"}


def profile_alpha(coords: FloatArray, profile: Profile) -> FloatArray:
    if profile not in ("stretched350", "fixed175plateau"):
        raise ValueError("Unknown damping profile")
    damping_alpha(coords, 0., ExperimentSpec(padding_m=PADDING_M))
    width = 350. if profile == "stretched350" else 175.
    x, y = coords[:, 0], coords[:, 1]
    sx = np.minimum(np.maximum.reduce((-x, x - 700., np.zeros_like(x))) / width, 1.)
    sy = np.minimum(np.maximum(-y - 700., 0.) / width, 1.)
    return np.asarray(ALPHA_MAX_S_INV * (sx**2 + sy**2), dtype=np.float64)


def abc_edges(case: MeshCase) -> IntArray:
    boundary = case.mesh.strata[2]
    selected: list[int] = []
    for edge in range(boundary.p_node_offsets.size - 1):
        ids = boundary.p_nodes[boundary.p_node_offsets[edge]:boundary.p_node_offsets[edge + 1]]
        xy = case.mesh.nodes_coords[ids]
        if np.all(np.abs(xy[:, 1]) < 1e-8):
            continue
        if not any(np.all(np.abs(xy[:, axis] - value) < 1e-8) for axis,value in ((0,-350.),(0,1050.),(1,-1050.))):
            raise ValueError("Internal boundary or wrong350m rectangle")
        selected.append(edge)
    return np.asarray(selected, dtype=np.int64)


def write_json(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def spec_for(variant: Variant, t_end: float = 2.) -> SurfaceSpec:
    return CoreSpec("EFVB", 0, 350., 96, t_end) if variant == "core" else SurfaceSpec("EFVB", 0, 350., 96, t_end)


def mesh_inputs(variant: Variant) -> tuple[Path, Path, Path]:
    directory = ROOT / "meshes" / variant
    return directory / "mesh_q2.npz", directory / "geometry.json", directory / "stability.json"


def check_material(frozen: FrozenMesh) -> None:
    if not np.all(frozen.vp == VP_M_S) or not np.all(frozen.vs == VS_M_S) or not np.all(frozen.regions == 0):
        raise ValueError("Mesh is not the requested single homogeneous material")
    if ExperimentSpec().density_kg_m3 != DENSITY_KG_M3:
        raise ValueError("Shared physics density differs from the homogeneous control")


def load_case(variant: Variant, spec: SurfaceSpec) -> tuple[MeshCase, NDArray[np.int64]]:
    mesh, geometry, stability = mesh_inputs(variant)
    frozen = load_frozen(mesh)
    check_material(frozen)
    record = read_json(geometry)
    if record["material_model"] != MATERIAL_MODEL or record["physical_model"] != PHYSICAL_MODEL or record["mesh_sha256"] != file_record(mesh)["sha256"]:
        raise ValueError("Homogeneous geometry evidence does not match material or mesh")
    with np.load(mesh, allow_pickle=False) as archive:
        if str(archive["material_model"]) != MATERIAL_MODEL or float(archive["density_kg_m3"]) != DENSITY_KG_M3:
            raise ValueError("Frozen mesh lacks its explicit homogeneous material identity")
    case = build_sem(frozen, 4)
    case.metadata.update(dataset="EFVB", material_model=MATERIAL_MODEL,
        physical_model=PHYSICAL_MODEL, is_original_dataset_model=False,
        identity_note="EFVB0 is the parent geometry/API identity only; material was replaced uniformly")
    if not np.all(case.boundary_vp == VP_M_S) or not np.all(case.boundary_vs == VS_M_S):
        raise ValueError("Boundary impedances do not use the homogeneous material")
    if variant == "core":
        edges = validate_core(case, cast(CoreSpec, spec), mesh, stability)
    else:
        validate_stability(case, spec, mesh, stability)
        edges = abc_edges(case)
    return case, edges


def extend350(core: FrozenMesh) -> FrozenMesh:
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
            for k in range(70):
                xx = np.sort(x0 + sign*(5*k + np.array([0, 2.5, 5])))
                x, y = np.meshgrid(xx, yy)
                add(np.stack((x, y), axis=-1), owner)
            if yy[0] < -700+1e-8:
                corner_owners[sign] = owner
        elif np.allclose(edge[:, 1], -700, atol=1e-8, rtol=0):
            xx = np.sort(edge[:, 0])
            for k in range(70):
                yy = -700-5*k-np.array([5, 2.5, 0])
                x, y = np.meshgrid(xx, yy)
                add(np.stack((x, y), axis=-1), owner)
        else:
            raise ValueError("Core contains an internal crack")
    for sign, owner in corner_owners.items():
        for j in range(70):
            for i in range(70):
                xx = (0 if sign == -1 else 700) + sign*(5*i+np.array([0, 2.5, 5]))
                yy = -700-5*j-np.array([5, 2.5, 0])
                x, y = np.meshgrid(np.sort(xx), yy)
                add(np.stack((x, y), axis=-1), owner)
    if set(corner_owners) != {-1, 1}:
        raise ValueError("Missing bottom corner material")
    return FrozenMesh(np.asarray(nodes), np.asarray(cells, dtype=np.int64), np.asarray(vp), np.asarray(vs), np.asarray(regions, dtype=np.int64), core.model_id, core.kind)


def audit350(mesh: FrozenMesh) -> dict[str, object]:
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
    side = np.stack([np.all(np.abs(boundary[:, :, axis]-value)<1e-8, axis=1) for axis, value in ((0, -350), (0, 1050), (1, -1050), (1, 0))])
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
    if abs(float(area.sum())-1400*1050)>1e-5:
        raise ValueError("Incorrect total area")
    return {"model_id": mesh.model_id, "kind": mesh.kind, "elements": len(mesh.cells), "q2_nodes": len(mesh.nodes),
            "area_m2": float(area.sum()), "det_jacobian_certified_min_m2": minimum, "bernstein_max_depth": max_depth,
            "euler_characteristic": 1, "conforming": True, "core_top_edges": 140, "core_top_edge_m": 5,
            "external_edges": len(exterior), "area_by_region_m2": {str(int(r)): float(area[mesh.regions == r].sum()) for r in np.unique(mesh.regions)},
            "sqrt_area_quantiles_m": np.quantile(np.sqrt(area), [0, .01, .5, .99, 1]).tolist()}


def estimate(path: Path, order: int, alpha_max: float) -> dict[str, object]:
    """Ограничить M⁻¹K по абсолютным множителям BᵀDB, не вычисляя спектр."""
    case = build_sem(load_frozen(path), order)
    coords = case.mesh.nodes_coords
    cells = np.asarray(case.mesh.strata[1].p_nodes, dtype=np.int64).reshape(-1, (order + 1)**2)
    _, derivatives, weights, derivative_line = tensor_basis(order)
    node_count = len(coords)
    mass = np.zeros(node_count, dtype=np.float64)
    rows = np.zeros((node_count, 2), dtype=np.float64)
    jacobian_min = float("inf")
    for begin in range(0, len(cells), 48):
        nodes = cells[begin:begin + 48]
        jac = np.einsum("epc,qpr->eqcr", coords[nodes], derivatives, optimize=True)
        det = np.linalg.det(jac)
        if not np.isfinite(det).all() or np.any(det <= 0):
            raise ValueError("Неположительный якобиан в квадратуре")
        jacobian_min = min(jacobian_min, float(det.min()))
        grad = np.einsum("qpr,eqrc->eqpc", derivatives, np.linalg.inv(jac), optimize=True)
        n = nodes.shape[1]
        b = np.zeros((len(nodes), n, 3, 2*n), dtype=np.float64)
        b[:, :, 0, 0::2] = grad[:, :, :, 0]
        b[:, :, 1, 1::2] = grad[:, :, :, 1]
        b[:, :, 2, 0::2] = grad[:, :, :, 1]
        b[:, :, 2, 1::2] = grad[:, :, :, 0]
        vp, vs = case.element_vp[begin:begin+48], case.element_vs[begin:begin+48]
        mu = 1000.0 * vs**2
        lam = 1000.0 * (vp**2 - 2*vs**2)
        material = np.zeros((len(nodes), 3, 3), dtype=np.float64)
        material[:, 0, 0] = material[:, 1, 1] = lam + 2*mu
        material[:, 0, 1] = material[:, 1, 0] = lam
        material[:, 2, 2] = mu
        abs_b = np.abs(b)
        weighted = det * weights[None, :]
        bound = np.einsum("eqai,eab,eqb,eq->ei", abs_b, np.abs(material), abs_b.sum(axis=3), weighted, optimize=True)
        np.add.at(mass, nodes.ravel(), (1000*weighted).ravel())
        np.add.at(rows, nodes.ravel(), bound.reshape(-1, 2))
    if np.any(mass <= 0):
        raise ValueError("Неположительная узловая масса")
    edge_nodes = np.asarray(case.mesh.strata[2].p_nodes, dtype=np.int64).reshape(-1, order+1)
    line_weights = np.asarray(case.mesh.pets[0].build_data(use_symmetry=False).weights, dtype=np.float64)
    abc = np.zeros(node_count, dtype=np.float64)
    abc_count = 0
    for edge_id, ids in enumerate(edge_nodes):
        if np.all(np.abs(coords[ids, 1]) < 1e-9):
            continue
        tangent = derivative_line @ coords[ids]
        values = line_weights * np.linalg.norm(tangent, axis=1) * 1000*case.boundary_vp[edge_id]
        np.add.at(abc, ids, values)
        abc_count += 1
    stiffness = float(np.max(rows / mass[:, None]))
    damping = float(np.max(abc/mass + profile_alpha(np.asarray(coords,dtype=np.float64), "fixed175plateau")))
    dt = float(2/(np.sqrt(damping*damping + 4*stiffness) + damping))
    return {
        "mesh": str(path.resolve()), "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "order": order, "nodes": node_count, "elements": len(cells),
        "alpha_max_s_inv": alpha_max, "stiffness_bound_s_minus_2": stiffness,
        "damping_bound_s_minus_1": damping, "dt_start_s": dt,
        "substeps_per_ms": int(np.ceil(.001/dt)), "min_mass_kg_per_m": float(mass.min()),
        "quadrature_jacobian_min_m2": jacobian_min, "abc_edge_count": abc_count,
        "profile_bound": "fixed175plateau >= stretched350 pointwise; same outer ABC",
        "method": "GLL mass; absolute-factor bound sum_q |B|^T |D| |B| wJ; ABC rhoVp edge quadrature and nodal alpha; half scalar damped central-difference limit",
        "limitation": "Conservative starting bound, not the solver spectrum or a convergence result; full-cell geometry certified separately",
    }



def prepare() -> dict[str, object]:
    output = ROOT / "meshes"
    if output.exists():
        raise FileExistsError(output)
    parent = PROJECT_ROOT / "data/dev_1_4_server_preparation/homogeneous_3060_v1/meshes/core/mesh_q2.npz"
    source = load_frozen(parent)
    check_material(source)
    frozen = extend350(source)
    check_material(frozen)
    if not np.array_equal(frozen.nodes[:len(source.nodes)],source.nodes) or not np.array_equal(frozen.cells[:len(source.cells)],source.cells):
        raise ValueError("Extension changed the original core")
    old175_path=PROJECT_ROOT / "data/dev_1_4_server_preparation/homogeneous_3060_v1/meshes/padded/mesh_q2.npz"
    old175=load_frozen(old175_path)
    distances,matching=cKDTree(frozen.nodes[frozen.cells[:,4]]).query(old175.nodes[old175.cells[:,4]])
    if np.max(distances)>1e-9 or not np.array_equal(frozen.nodes[frozen.cells[matching]],old175.nodes[old175.cells]):
        raise ValueError("Old175m Q2 geometry changed")
    np.testing.assert_allclose(profile_alpha(old175.nodes,"fixed175plateau"),damping_alpha(old175.nodes,ALPHA_MAX_S_INV),rtol=0.,atol=1e-12)
    certificate = audit350(frozen)
    mesh,geometry,stability = mesh_inputs("padded")
    mesh.parent.mkdir(parents=True,exist_ok=False)
    with mesh.open("xb") as stream:
        np.savez(stream,nodes=frozen.nodes,cells=frozen.cells,vp=frozen.vp,vs=frozen.vs,regions=frozen.regions,
            model_id=frozen.model_id,kind=frozen.kind,dataset="EFVB",material_model=MATERIAL_MODEL,
            density_kg_m3=DENSITY_KG_M3,is_original_dataset_model=False)
    write_json(geometry,{"schema":"chapter14.double_layer_geometry.v1", "padding_m":PADDING_M,
        "material_model":MATERIAL_MODEL,"physical_model":PHYSICAL_MODEL,"is_original_dataset_model":False,
        "geometry_parent":{"mesh":file_record(parent),"core_coordinates_exact":True,"core_connectivity_exact":True,"old175_mesh":file_record(old175_path),"old175_q2_cell_coordinates_exact":True},
        "mesh_sha256":file_record(mesh)["sha256"],"certificate":certificate,"code":file_record(Path(__file__))})
    record=estimate(mesh,4,ALPHA_MAX_S_INV)
    record.update(dataset="EFVB",material_model=MATERIAL_MODEL,physical_model=PHYSICAL_MODEL,
        selected_substeps_per_ms=96,selection_rule="SameK96 as0m/175m homogeneous reference; newmesh spectralcertificate")
    if number(record["substeps_per_ms"])>96 or .001/96>number(record["dt_start_s"]):
        raise ValueError("K96 unsafe")
    write_json(stability,record)
    case,boundary=load_case("padded",spec_for("padded"))
    coords=np.asarray(case.mesh.nodes_coords,dtype=np.float64)
    old_sem=build_sem(old175,4)
    old_coords=np.asarray(old_sem.mesh.nodes_coords,dtype=np.float64)
    sem_dist,sem_match=cKDTree(coords).query(old_coords)
    if np.max(sem_dist)>1e-9 or len(np.unique(sem_match))!=len(old_coords):
        raise ValueError("Old175m SEM nodes do not map uniquely into350m mesh")
    alpha_delta=float(np.max(np.abs(profile_alpha(coords[sem_match],"fixed175plateau")-damping_alpha(old_coords,ALPHA_MAX_S_INV))))
    if alpha_delta>1e-10:
        raise ValueError("Held175 profile differs on shared SEM nodes")
    oldbox=(coords[:,0]>=-175.-1e-9)&(coords[:,0]<=875.+1e-9)&(coords[:,1]>=-875.-1e-9)
    fixed=profile_alpha(coords,"fixed175plateau")
    np.testing.assert_allclose(fixed[oldbox],damping_alpha(coords[oldbox],ALPHA_MAX_S_INV),rtol=1e-14,atol=1e-11)
    stretched=profile_alpha(coords,"stretched350")
    core=(coords[:,0]>=-1e-9)&(coords[:,0]<=700.+1e-9)&(coords[:,1]>=-700.-1e-9)
    if np.max(fixed[core])>1e-20 or np.max(stretched[core])>1e-20:
        raise ValueError("Core has damping")
    resources=os.statvfs(ROOT)
    result:dict[str,object]={"status":"prepared","utc":datetime.now(timezone.utc).isoformat(),
        "material_model":MATERIAL_MODEL,"physical_model":PHYSICAL_MODEL,"padding_m":PADDING_M,
        "nodes":len(coords),"elements":len(case.element_vp),"abc_edges":len(boundary),
        "minimum_substeps_per_ms":record["substeps_per_ms"],"selected_substeps_per_ms":96,
        "field_bytes_each":201*len(coords)*16,"disk_free_bytes":resources.f_bavail*resources.f_frsize,
        "profiles":{p:profile_record(cast(Profile,p)) for p in ("stretched350","fixed175plateau")},
        "fixed175_matches_reference_in_old_domain":True,"core_q2_coordinates_connectivity_exact":True,
        "old175_q2_cell_coordinates_exact":True,
        "old175_sem_matching_max_distance_m":float(np.max(sem_dist)),"old175_shared_alpha_max_abs_difference_s_inv":alpha_delta,
        "geometry":file_record(geometry),"stability":file_record(stability),"solver_run":False}
    write_json(output/"preparation.json",result)
    return result


def execute(case: MeshCase, spec: SurfaceSpec, variant: Variant, output: Path,
            metadata: dict[str, object], edges: NDArray[np.int64], profile: Profile) -> dict[str, object]:
    experiment = ExperimentSpec(padding_m=PADDING_M)
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    ids = np.arange(len(coords), dtype=np.int64)
    receivers = receiver_nodes(case)
    source = nodal_point_force(coords, 350., experiment.force_amplitude_n_per_m)
    model, native, success_code = make_cu_model(case, spec)
    source_times = np.linspace(0., spec.t_end_s, spec.steps + 1)
    forces = np.zeros((len(source_times), 1, 2), dtype=np.float64)
    forces[:, 0, 1] = source.force_y * ricker(source_times)
    model.add_physics(SGPhysicsModel.from_entity_values("material", 11000, case.mesh, 1,
        np.arange(len(case.element_vp), dtype=np.int64), plane_strain_parameters(case.element_vp, case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values("outer_abc", 32000, case.mesh, 2, edges,
        np.column_stack((experiment.density_kg_m3 * case.boundary_vp[edges], experiment.density_kg_m3 * case.boundary_vs[edges]))))
    physics_ids = [11000, 32000]
    if variant == "padded":
        model.add_physics(SGPhysicsModel("mass_damping", 30001, case.mesh, 0, ids,
            profile_alpha(coords, profile)[:, None]))
        physics_ids.append(30001)
    model.add_physics(SGPhysicsModel("point_ricker", 20200, case.mesh, 0,
        np.asarray([source.node_id], dtype=np.int64), forces,
        time=SGPhysicsTimeData(0., spec.t_end_s, len(source_times), 3)))
    physics_ids.append(20200)
    metadata.update(source_node_id=source.node_id, source_input_sha256=hashlib.sha256(forces.tobytes()).hexdigest(),
        source_input_first_force_y_n_per_m=float(forces[0, 0, 1]), native=file_record(native),
        nodes_sha256=hashlib.sha256(coords.tobytes()).hexdigest(), physics_ids=physics_ids, abc_edge_count=len(edges))
    surface = SurfaceRecorder.create(spec)
    fields = FieldRecorder.create(output, spec, len(coords))
    np.save(output / "nodes.npy", coords, allow_pickle=False)
    np.save(output / "field_time_s.npy", fields.times, allow_pickle=False)

    def surface_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        surface.append(frame, event)
        if (surface.count - 1) % 500 == 0:
            print(json.dumps({"stage": "surface", "variant": variant, "time_s": float(surface.times[surface.count - 1])}), flush=True)

    def field_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        fields.append(frame, event)

    model.add_receiver(SGReceiverModel.from_positions("surface_velocity", case.mesh, 0, receivers,
        np.asarray([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=spec.substeps, handler=surface_handler))
    model.add_receiver(SGReceiverModel.from_positions("all_node_velocity", case.mesh, 0, ids,
        np.asarray([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=10 * spec.substeps, handler=field_handler))
    _write_metadata(output / "metadata.started.json", metadata, None)
    model.prepare()
    print(json.dumps({"stage": "prepared", "variant": variant, "nodes": len(coords), "steps": spec.steps}), flush=True)
    try:
        result = cast(TimedResult, model.run(sync=True))
        if int(result.code) != success_code or result.steps_count != spec.steps or not np.isclose(result.current_time, spec.t_end_s, atol=2e-10, rtol=0):
            raise RuntimeError(f"Integration did not finish: {result}")
        surface.finish()
        if fields.count != len(fields.times):
            raise ValueError("Missing field frames")
        for index in range(len(fields.times)):
            if not np.array_equal(fields.values[index, receivers], surface.velocity[10 * index]):
                raise ValueError("All-node and surface recorders disagree at shared times/nodes")
        fields.finish()
    finally:
        fields.flush()
    partial = output / "surface.partial"
    _check_write(output, surface.velocity.nbytes + 100_000, None)
    with partial.open("xb") as stream:
        np.savez(stream, time_s=surface.times, coordinates_m=coords[receivers], node_ids=receivers,
                 velocity_m_s=surface.velocity)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output / "surface.npz")
    return {"runtime": {"result_code": int(result.code), "time_end_code": success_code,
        "steps_count": result.steps_count, "current_time_s": result.current_time},
        "recorded_counts": {"surface": surface.count, "fields": fields.count, "source_history": 0},
        "surface_file": stream_record(output / "surface.npz"),
        "surface_file_bytes": (output / "surface.npz").stat().st_size,
        "field_files": {name: stream_record(output / name) for name in ("velocity_m_s.npy", "nodes.npy", "field_time_s.npy")},
        "all_node_surface_values_exact": True}



def verify(output: Path, spec: SurfaceSpec) -> dict[str, object]:
    metadata = fields.verify(output, spec)
    variant = cast(Variant, metadata["variant"])
    if variant not in ("padded", "core") or metadata.get("material_model") != MATERIAL_MODEL or metadata.get("physical_model") != PHYSICAL_MODEL or metadata.get("is_original_dataset_model") is not False:
        raise ValueError("Run does not declare the requested homogeneous override")
    profile=cast(Profile,mapping(metadata["damping_profile"])["name"])
    if metadata["padding_m"] != PADDING_M or metadata["damping_profile"] != profile_record(profile):
        raise ValueError("Wrong damping profile metadata")
    case, edges = load_case(variant, spec)
    expected_physics = [11000, 32000, 30001, 20200] if variant == "padded" else [11000, 32000, 20200]
    if metadata["physics_ids"] != expected_physics or metadata["abc_edge_count"] != len(edges):
        raise ValueError("Wrong ABC or mass-damping selection")
    nodes = np.load(output / "nodes.npy", mmap_mode="r", allow_pickle=False)
    if not np.array_equal(nodes, case.mesh.nodes_coords):
        raise ValueError("Recorded coordinates differ from the homogeneous mesh")
    return metadata


def run(variant: Variant, output: Path, t_end: float = 2., profile: Profile = "stretched350") -> dict[str, object]:
    if variant != "padded" or profile not in ("stretched350", "fixed175plateau"):
        raise ValueError("Only350m padded geometry and declared profiles are supported")
    output = output.resolve()
    output.relative_to(ROOT)
    if output.exists():
        raise FileExistsError(output)
    spec = spec_for(variant, t_end)
    times = inclusive_times(t_end, .01)
    case, edges = load_case(variant, spec)
    mesh, geometry, stability = mesh_inputs(variant)
    _check_write(output.parent, (len(times) + 1) * len(case.mesh.nodes_coords) * 16 + 10_000_000, None)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "variant": variant, "padding_m": PADDING_M, "damping_profile": profile_record(profile),
        "material_model": MATERIAL_MODEL, "physical_model": PHYSICAL_MODEL, "is_original_dataset_model": False,
        "geometry_parent": read_json(geometry)["geometry_parent"],
        "model_identity_note": "Homogeneous synthetic control. Legacy EFVB0 in run_spec identifies only the reused geometry and API; this is not the original dataset model.",
        "run_spec": asdict(spec), "experiment_spec": {**asdict(ExperimentSpec()), "padding_m": PADDING_M},
        "boundary_mode": "outer_abc" if variant == "padded" else "core_abc", "mass_damping_disabled": variant == "core",
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata,
        "mesh_evidence": file_record(mesh), "geometry_evidence": file_record(geometry), "stability_evidence": file_record(stability),
        "short_test": t_end != 2., "components": ["Vx", "Vy"], "receiver_count": 70,
        "field_components": ["Vx", "Vy"], "field_cadence_s": .01,
        "field_shape": [len(times), len(case.mesh.nodes_coords), 2],
        "field_node_order": "all global SEM node IDs in increasing order, coordinates in nodes.npy",
        "initial_state": "native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3",
        "storage": "float64 uncompressed NPY memmap all-node velocity plus surface NPZ; no displacement or energy",
        "timing_note": "Homogeneous animation control with all-node field I/O; not the earlier surface-only benchmark",
        "provenance": {"python": sys.executable, "python_version": sys.version, "numpy": np.__version__,
            "semgpu2": version("semgpu2"), "platform": platform.platform(),
            "files": [file_record(Path(__file__)), file_record(Path(fields.__file__)),
                file_record(Path(__file__).with_name("surface_runner.py")), file_record(Path(__file__).with_name("no_padding_run.py")),
                *[file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source" / name) for name in ("physics.py", "run.py", "mesh.py", "stability.py")]]}}
    try:
        metadata.update(execute(case, spec, variant, output, metadata, edges, profile))
        metadata.update(status="completed", wall_seconds=perf_counter() - started)
        _write_metadata(output / "metadata.json", metadata, None)
        verified = verify(output, spec)
        write_json(output / "validation.json", {"status": "passed", "material_model": MATERIAL_MODEL,
            "physical_model": PHYSICAL_MODEL, "all_node_surface_values_exact": True,
            "field_shape": metadata["field_shape"], "recorded_counts": metadata["recorded_counts"],
            "metadata": file_record(output / "metadata.json"), "mesh": file_record(mesh),
            "scope": "Finite complete field/surface recording and requested uniform material/boundary configuration; not independent solver accuracy validation"})
        return verified
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter() - started,
            error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(output / "metadata.failed.json", metadata, None)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="CPU-only exact geometry copies and new material/time-step certificates")
    job = sub.add_parser("run", help="Explicit native/GPU calculation")
    job.add_argument("--variant", choices=("padded",), default="padded")
    job.add_argument("--profile", choices=("stretched350", "fixed175plateau"), required=True)
    job.add_argument("--out", type=Path, required=True)
    job.add_argument("--t-end", type=float, default=2.)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(), ensure_ascii=False), flush=True)
        return
    output = args.out.resolve()
    output.relative_to(ROOT)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    for variable, name in (("TMPDIR", "tmp"), ("XDG_CACHE_HOME", "xdg"), ("MPLCONFIGDIR", "matplotlib"), ("CUDA_CACHE_PATH", "cuda")):
        cache = ROOT / "runtime_cache" / name
        cache.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(cache)
    tempfile.tempdir = os.environ["TMPDIR"]
    result = run(cast(Variant, args.variant), output, args.t_end, cast(Profile,args.profile))
    print(json.dumps({"status": result["status"], "variant": args.variant, "material_model": MATERIAL_MODEL,
        "wall_seconds": result["wall_seconds"], "out": str(output)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
