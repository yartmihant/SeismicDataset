"""Q2-вход ECVB и сборка p6 SEMGPU2 с нумерацией верха для фиксированных источников."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from semgpu2.model import SGPhysicsModel
from semgpu2.model.mesh import SGMeshModel
from semgpu2.tools.mesh.numbering import with_leading_nodes

from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters, RHO_KG_M3
from scripts.dev_1_3_point_source.mesh import FrozenMesh, IntArray, edges
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import H, PAD, W
from scripts.dev_1_5_geometry_frequency.setup import ALPHA
from scripts.dev_1_6_p67s_showcase.mesh import audit, build_sem, extend


SCHEMA = "ecvb.semgpu.core-q2.v1"
CORE_BOUNDS = (0., float(W), -float(H), 0.)  # xmin, xmax, ymin, ymax
NATIVE_PROFILE = "docs/runtime_sources/seismic_p64s_v1"
# validate_surface in the p64s/p67s backend, including its absolute tolerance.
GLL6 = np.array([-1., -.8302238962785669, -.4688487934707142, 0.,
                  .4688487934707142, .8302238962785669, 1.])
TOP_X = np.r_[-350. + 10.*(np.arange(210)[:, None] + .5*(1. + GLL6[:-1])).ravel(), 1750.]
COORD_TOLERANCE = 1.e-9
RECEIVER_X = np.arange(5., W, 10.)


@dataclass(frozen=True)
class FrozenSourceProfile:
    """Пары глобальный ID (от нуля) и x в порядке опытов конкретной сборки."""
    name: str
    node_ids: tuple[int, ...]
    x_m: tuple[float, ...]


P67S = FrozenSourceProfile("p67s", (270, 390, 510, 630, 750, 870, 990),
                                   (100., 300., 500., 700., 900., 1100., 1300.))
P64S = FrozenSourceProfile("p64s", P67S.node_ids[:4], P67S.x_m[:4])


@dataclass(frozen=True)
class P6Input:
    mesh: SGMeshModel
    material: SGPhysicsModel
    pml: SGPhysicsModel
    top_nodes: IntArray
    source_nodes: IntArray
    receiver_nodes: IntArray
    full_q2_validation: dict[str, object]


def source_nodes(mesh: SGMeshModel, profile: FrozenSourceProfile) -> IntArray:
    ids = np.asarray(profile.node_ids, dtype=np.int64)
    if (ids.ndim != 1 or len(ids) == 0 or len(profile.x_m) != len(ids)
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= len(TOP_X))):
        raise ValueError(f"Invalid frozen node IDs for {profile.name}")
    target = np.column_stack((profile.x_m, np.zeros(len(ids))))
    np.testing.assert_allclose(mesh.nodes_coords[ids], target, rtol=0, atol=COORD_TOLERANCE,
                               err_msg=f"Frozen source coordinates for {profile.name}", equal_nan=False)
    return ids


def validate_extension(core: FrozenMesh, full: FrozenMesh) -> dict[str, object]:
    """По проекции на границу ядра независимо найти материал каждого элемента PML."""
    report = audit(full)
    core_count = len(core.cells)
    center = full.nodes[full.cells[core_count:, 4]]
    directed, _, _, exterior = edges(core)
    boundary = core.nodes[directed[exterior]]
    owners = np.full(len(center), -1, dtype=np.int64)
    # В нижних углах выбирается крайний нижний элемент соответствующей боковой стороны.
    # Остальные точки проектируются на свой отрезок нормалью к стороне.
    for axis, value, mask in ((0, 0., center[:, 0] < 0.),
                              (0, float(W), center[:, 0] > W),
                              (1, -float(H), (center[:, 0] >= 0.) & (center[:, 0] <= W))):
        side = np.all(np.abs(boundary[:, :, axis]-value) < COORD_TOLERANCE, axis=1)
        tangent = boundary[side][:, [0, 2], 1-axis]
        order = np.argsort(tangent.min(axis=1))
        starts, ends = tangent[order].min(axis=1), tangent[order].max(axis=1)
        expected_extent = [-H, 0.] if axis == 0 else [0., W]
        np.testing.assert_allclose([starts[0], ends[-1]], expected_extent, rtol=0, atol=COORD_TOLERANCE)
        np.testing.assert_allclose(starts[1:], ends[:-1], rtol=0, atol=COORD_TOLERANCE)
        coordinate = np.clip(center[mask, 1-axis], starts[0], ends[-1])
        owners[mask] = (exterior[side]//4)[order[np.searchsorted(ends, coordinate)]]
    if np.any(owners < 0):
        raise ValueError("PML cell without a boundary material")
    for expanded, field in ((full.regions, core.regions), (full.vp, core.vp), (full.vs, core.vs)):
        np.testing.assert_array_equal(expanded[:core_count], field)
        np.testing.assert_array_equal(expanded[core_count:], field[owners])
    report["pml_boundary_material_verified"] = True
    return report


def prepare_p6(path: Path, profile: FrozenSourceProfile = P67S) -> P6Input:
    """Прочитать Q2-ядро, нарастить PML, собрать сетку и объекты физики SGModel в памяти."""
    with np.load(path, allow_pickle=False) as data:
        if str(data["schema"]) != SCHEMA:
            raise ValueError("Unknown SEMGPU Q2 input schema")
        core = FrozenMesh(data["nodes"], data["cells"], data["vp"], data["vs"], data["regions"],
                          int(data["model_id"]), str(data["kind"]))
        np.testing.assert_array_equal(data["core_bounds_m"], CORE_BOUNDS)
        if float(data["rho_kg_m3"]) != RHO_KG_M3 or float(data["pml_thickness_m"]) != PAD:
            raise ValueError("Unexpected material density or PML thickness")
        dmax = float(data["pml_dmax_s_inv"])
        fc_ids = data["core_fc_element_ids"]
    core_count = len(core.cells)
    if (core_count == 0 or core.cells.shape != (core_count, 9) or fc_ids.shape != (core_count,)
            or len(np.unique(fc_ids)) != core_count or core.nodes.ndim != 2 or core.nodes.shape[1] != 2
            or not np.isfinite(core.nodes).all() or core.cells.min() < 0 or core.cells.max() >= len(core.nodes)
            or not np.isfinite(dmax) or dmax <= 0):
        raise ValueError("Invalid Q2 geometry or PML parameters")
    np.testing.assert_allclose(np.stack((core.nodes.min(axis=0), core.nodes.max(axis=0)), axis=1).ravel(),
                               CORE_BOUNDS, rtol=0, atol=COORD_TOLERANCE)
    for field in (core.vp, core.vs, core.regions):
        if field.shape != (core_count,) or not np.isfinite(field).all() or np.any(field <= 0):
            raise ValueError("Missing or invalid elemental material")
    frozen = extend(core)
    validation = validate_extension(core, frozen)
    count = len(frozen.cells)
    center = frozen.nodes[frozen.cells[:, 4]]
    pml_ids = np.flatnonzero((center[:, 0] < 0) | (center[:, 0] > W) | (center[:, 1] < -H))
    np.testing.assert_array_equal(pml_ids, np.arange(core_count, count))

    raw = build_sem(frozen, 6).mesh
    top = np.flatnonzero(np.abs(raw.nodes_coords[:, 1]) < COORD_TOLERANCE)
    if len(top) != len(TOP_X) or set(raw.strata) != {1}:
        raise ValueError("Expected 1261 top nodes and a single MAIN stratum")
    mesh, _ = with_leading_nodes(raw, top[np.argsort(raw.nodes_coords[top, 0])])
    top = np.arange(len(top), dtype=np.int64)
    target = np.column_stack((TOP_X, np.zeros(len(top))))
    np.testing.assert_allclose(mesh.nodes_coords[top], target, rtol=0, atol=COORD_TOLERANCE,
                               equal_nan=False, err_msg="Fixed solver upper surface")
    # At p6 the midpoint is local GLL node 3. Core starts at full edge 35.
    receivers = (35 + np.arange(len(RECEIVER_X), dtype=np.int64))*6 + 3
    np.testing.assert_allclose(mesh.nodes_coords[receivers, 0], RECEIVER_X, rtol=0,
                               atol=COORD_TOLERANCE, equal_nan=False)
    ids = source_nodes(mesh, profile)
    material = SGPhysicsModel.from_entity_values(
        "material", 11000, mesh, 1, np.arange(count, dtype=np.int64),
        plane_strain_parameters(frozen.vp, frozen.vs))
    cells = mesh.strata[1].p_nodes.reshape(count, 49)
    xy = mesh.nodes_coords[cells[pml_ids].ravel()]
    sx = np.maximum(np.maximum(-xy[:, 0], xy[:, 0] - W), 0.)/PAD
    sy = np.maximum(-xy[:, 1] - H, 0.)/PAD
    pml = SGPhysicsModel("cartesian_pml", 31100, mesh, 1, pml_ids, dmax*np.column_stack((sx*sx, sy*sy)))
    return P6Input(mesh, material, pml, top, ids, receivers, validation)


def cpu_check(model: P6Input) -> dict[str, object]:
    """Проверить в том числе DTO в формате, который SGModel передает решателю."""
    mesh = model.mesh
    coords = mesh.build_n_coords()
    pq = mesh.build_pq_space_data(1)
    pets = mesh.build_pets_data()
    np.testing.assert_array_equal(pets.p_node_counts, [7])
    np.testing.assert_array_equal(pets.q_node_counts, [7])
    np.testing.assert_array_equal(np.diff(pq.p_node_offsets), 49)
    np.testing.assert_array_equal(pq.q_node_offsets, pq.p_node_offsets)
    material = model.material.build_data()
    pml = model.pml.build_data()
    for field, components in ((material, 3), (pml, 2)):
        if (field.values.shape != (components, 49*len(field.entity_ids))
                or not np.isfinite(field.values).all()):
            raise ValueError("Invalid physics DTO for SEMGPU2")
    np.testing.assert_array_equal(material.values[0], RHO_KG_M3)
    np.testing.assert_array_equal(np.min(pml.values, axis=1), [0., 0.])
    np.testing.assert_allclose(np.max(pml.values, axis=1), [ALPHA, ALPHA], rtol=1e-12, atol=0.)
    profiles = {p.name: {"node_ids": source_nodes(mesh, p).tolist(), "x_m": list(p.x_m)}
                for p in (P64S, P67S)}
    return {"order": 6, "nodes": len(mesh.nodes_coords), "top_nodes": len(model.top_nodes),
            "transient_elements": len(material.entity_ids), "transient_pml_elements": len(pml.entity_ids),
            "transient_full_q2_validation": model.full_q2_validation,
            "top_id_range": [0, len(TOP_X)-1], "top_max_abs_x_error_m": float(np.max(np.abs(coords[0, :len(TOP_X)]-TOP_X))),
            "top_max_abs_y_m": float(np.max(np.abs(coords[1, :len(TOP_X)]))),
            "sources": profiles, "source_profile_snapshot": NATIVE_PROFILE,
            "receiver_nodes": model.receiver_nodes.tolist(), "receiver_x_m": RECEIVER_X.tolist(),
            "boundary_elements": 0, "abc": False, "pml_dmax_xy_s_inv": np.max(pml.values, axis=1).tolist(),
            "semgpu_cpu_dto_checked": True}


def prepare(fc_path: Path, cad: dict, out: Path) -> dict[str, object]:
    """Сохранить только Q2-ядро с материалом, затем проверить достроенную из файла модель."""
    from scripts.dev_1_6_fidesys_mesh.validate import read_core
    core, element_ids = read_core(fc_path, cad, require_fc_materials=True)
    with out.open("xb") as stream:
        np.savez_compressed(stream, schema=SCHEMA, nodes=core.nodes, cells=core.cells,
                            regions=core.regions, vp=core.vp, vs=core.vs, model_id=core.model_id,
                            kind=core.kind, rho_kg_m3=RHO_KG_M3, core_fc_element_ids=element_ids,
                            core_bounds_m=CORE_BOUNDS, pml_thickness_m=PAD, pml_dmax_s_inv=ALPHA)
    p6_report = cpu_check(prepare_p6(out))
    return {"schema": SCHEMA, "core_elements": len(core.cells), "core_q2_nodes": len(core.nodes),
            "stored_pml_elements": 0, "stored_order": 2, "core_bounds_m": list(CORE_BOUNDS),
            "pml_thickness_m": PAD, "pml_dmax_s_inv": ALPHA,
            "rho_kg_m3": RHO_KG_M3, "p6_cpu_check": p6_report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Путь к sem_input.npz")
    args = parser.parse_args()
    print(json.dumps(cpu_check(prepare_p6(args.input)), indent=2))
