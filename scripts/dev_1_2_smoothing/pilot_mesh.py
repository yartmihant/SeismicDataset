"""Raster control and frozen Q2 layer-fitted geometry for the ECVB pilot."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
from typing import Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import BSpline  # type: ignore[import-untyped]  # SciPy has no typing metadata here.

from semgpu2.model import SGPETALineModel
from semgpu2.model.mesh import SGMeshModel
from semgpu2.tools.mesh.box import rectangle_quad_fast


ROOT = DATA_12
OUTPUT = ROOT / "pilot_geometry"
FloatArray = NDArray[np.float64]


@dataclass
class MeshCase:
    mesh: SGMeshModel
    element_vp: FloatArray
    element_vs: FloatArray
    boundary_vp: FloatArray
    boundary_vs: FloatArray
    metadata: Dict[str, object]


def quadratic_values(values: FloatArray, x: FloatArray) -> FloatArray:
    """Evaluate fixed piecewise quadratics tabulated at x=0,5,...700."""
    cell = np.minimum(np.floor(x / 10).astype(np.int64), 69)
    xi = (x - cell * 10) / 5 - 1
    weights = np.stack((xi*(xi-1)/2, 1-xi*xi, xi*(xi+1)/2))
    return np.sum(values[:, 2*cell[:, None] + np.arange(3)] * weights.T[None, :, :], axis=-1)


def quadratic_range(values: FloatArray) -> Tuple[FloatArray, FloatArray]:
    left, middle, right = values[:, :-1:2], values[:, 1::2], values[:, 2::2]
    a = (left+right)/2 - middle
    b = (right-left)/2
    vertex = np.divide(-b, 2*a, out=np.zeros_like(a), where=np.abs(a)>1e-14)
    at_vertex = a*vertex**2 + b*vertex + middle
    inside = (np.abs(vertex) < 1) & (np.abs(a)>1e-14)
    low, high = np.minimum(left, right), np.maximum(left, right)
    return np.where(inside, np.minimum(low, at_vertex), low), np.where(inside, np.maximum(high, at_vertex), high)


def interface_change_metrics(smooth: FloatArray, raster: FloatArray) -> Dict[str, object]:
    """Integrate exact area between each frozen Q2 interface and raster steps."""
    areas = []
    maxima = []
    for fitted, original in zip(smooth[1:-1], raster, strict=True):
        area = 0.0
        maximum = 0.0
        for column in range(70):
            left, middle, right = fitted[2*column:2*column+3]
            a = float((left+right)/2-middle)
            b = float((right-left)/2)
            c = float(middle-original[column])
            polynomial = np.polynomial.Polynomial([c, b, a])
            roots = np.asarray(polynomial.roots(), dtype=np.complex128)
            cuts = [-1.0] + sorted(float(root.real) for root in roots if abs(root.imag) < 1e-12 and -1 < root.real < 1) + [1.0]
            primitive = polynomial.integ()
            area += 5 * sum(abs(float(primitive(hi)-primitive(lo))) for lo, hi in zip(cuts[:-1], cuts[1:], strict=True))
            candidates = [-1.0, 1.0]
            if abs(a) > 1e-14 and abs(b/(2*a)) < 1:
                candidates.append(-b/(2*a))
            maximum = max(maximum, max(abs(float(polynomial(value))) for value in candidates))
        areas.append(area)
        maxima.append(maximum)
    return {"area_between_interfaces_m2": areas, "max_depth_difference_including_step_sides_m": maxima,
            "sum_interface_displacement_area_m2": sum(areas), "fraction_of_domain": sum(areas)/700**2}


def prepare_geometry() -> Dict[str, object]:
    with np.load(OUTPUT / "selected_model.npz", allow_pickle=False) as data:
        model_id = int(data["model_id"])
        interfaces = np.asarray(data["interface_depth_m"], dtype=np.float64)
        x_centres = np.asarray(data["x_m"], dtype=np.float64)
        labels = data["material_labels"]
        materials = data["materials_vp_vs"]
        first = labels[:, 0]
        sequence = first[np.r_[True, np.diff(first) != 0]]
    if interfaces.shape != (len(sequence)-1, 70):
        raise ValueError("Selection must have the same nonvanishing layer sequence in all columns")
    for column in labels.T:
        if not np.array_equal(column[np.r_[True, np.diff(column) != 0]], sequence):
            raise ValueError("The chosen raster is not a layered block topology")
    x = np.r_[0.0, x_centres, 700.0]
    source = np.column_stack((interfaces[:, 0], interfaces, interfaces[:, -1]))
    nodes_x = np.linspace(0, 700, 141)
    smooth = np.empty((len(sequence)+1, 141), dtype=np.float64)
    smooth[0], smooth[-1] = 0.0, 700.0
    spline_records = []
    for row, depths in enumerate(source):
        accepted = False
        for control_count in (10, 14, 20, 28, 40):
            knots = np.r_[np.zeros(4), np.linspace(0, 700, control_count-2)[1:-1], np.full(4, 700.0)]
            basis = BSpline.design_matrix(x, knots, 3).toarray()
            rhs = depths - basis[:, 0]*depths[0] - basis[:, -1]*depths[-1]
            coefficients = np.r_[depths[0], np.linalg.lstsq(basis[:, 1:-1], rhs, rcond=None)[0], depths[-1]]
            spline = BSpline(knots, coefficients, 3)
            deviation = float(np.max(np.abs(np.asarray(spline(x_centres))-interfaces[row])))
            if deviation <= 7.5:
                smooth[row+1] = spline(nodes_x)
                spline_records.append({"knots": knots.tolist(), "coefficients": coefficients.tolist(), "max_sample_depth_error_m": deviation})
                accepted = True
                break
        if not accepted:
            raise ValueError("Could not fit an interface within the 7.5 m trial bound")
    thickness = np.diff(smooth, axis=0)
    low, high = quadratic_range(thickness)
    if float(low.min()) <= 0:
        raise ValueError("Q2 interfaces intersect")
    means = np.sum((thickness[:, :-1:2]+4*thickness[:, 1::2]+thickness[:, 2::2])*10/6, axis=1) / 700
    rows = np.maximum(1, np.rint(means/10).astype(np.int64))
    # With x affine and y linear in layer fraction, detJ = 10 * thickness(x)/(4*nrows).
    jacobian_min = float(np.min(2.5*low/rows[:, None]))
    jacobian_max = float(np.max(2.5*high/rows[:, None]))
    np.savez_compressed(OUTPUT / "frozen_q2_geometry.npz", depth_q2=smooth,
                        rows_topdown=rows, materials_topdown=materials[sequence], model_id=model_id)
    record: Dict[str, object] = {
        "model_id": model_id, "geometry_degree": [2, 1], "embedding_space": "Q2",
        "horizontal_elements": 70, "vertical_elements": int(rows.sum()),
        "rows_topdown": rows.tolist(), "layer_min_thickness_m": low.min(axis=1).tolist(),
        "layer_max_thickness_m": high.max(axis=1).tolist(),
        "vertical_cell_min_m": float(np.min(low/rows[:, None])),
        "vertical_cell_max_m": float(np.max(high/rows[:, None])),
        "det_jacobian_min": jacobian_min, "det_jacobian_max": jacobian_max,
        "jacobian_check": "Analytic min/max of each quadratic layer thickness over each x interval; x affine and y affine in vertical layer fraction",
        "geometry_refinement": "Higher p samples the frozen Q2 polynomial, never the cubic spline anew",
        "source_material_areas_m2": (np.bincount(labels.ravel(), minlength=len(materials))*100).tolist(),
        "curved_layer_areas_m2": (means*700).tolist(),
        "splines": spline_records,
        "raster_to_q2_interface_change": interface_change_metrics(smooth, interfaces),
    }
    (OUTPUT / "geometry.json").write_text(json.dumps(record, indent=2)+"\n", encoding="utf-8")
    return record


def build_case(kind: str, order: int) -> MeshCase:
    if kind not in ("raster", "curved") or order < 2:
        raise ValueError("Expected raster/curved and p>=2")
    with np.load(OUTPUT / "selected_model.npz", allow_pickle=False) as data:
        original_vp, original_vs = data["vp"], data["vs"]
        model_id = int(data["model_id"])
    info = prepare_geometry()
    with np.load(OUTPUT / "frozen_q2_geometry.npz", allow_pickle=False) as geometry:
        heights = 700 - geometry["depth_q2"][::-1]
        rows = geometry["rows_topdown"][::-1]
        materials = geometry["materials_topdown"][::-1]
    ny = 70 if kind == "raster" else int(rows.sum())
    line = SGPETALineModel("GLL", order=order)
    mesh = rectangle_quad_fast(sizes=(700.0, float(ny)), elements=(70, ny), pets=[line])
    x = mesh.nodes_coords[:, 0] + 350
    logical_y = mesh.nodes_coords[:, 1] + ny/2
    row_centres = (np.arange(ny)+0.5).repeat(70)
    col_centres = np.tile(np.arange(70)+0.5, ny)*10
    if kind == "raster":
        y = logical_y*10
        rr = 69-np.floor(row_centres).astype(np.int64)
        cc = np.floor(col_centres/10).astype(np.int64)
        element_vp, element_vs = original_vp[rr, cc], original_vs[rr, cc]
        info = {"det_jacobian_min": 25.0, "det_jacobian_max": 25.0}
    else:
        offsets = np.r_[0, np.cumsum(rows)]
        layer = np.minimum(np.searchsorted(offsets, logical_y, side="right")-1, len(rows)-1)
        fraction = (logical_y-offsets[layer])/rows[layer]
        evaluated = quadratic_values(heights, x)
        ids = np.arange(x.size)
        y = (1-fraction)*evaluated[layer, ids] + fraction*evaluated[layer+1, ids]
        cell_layer = np.searchsorted(offsets, row_centres, side="right")-1
        element_vp, element_vs = materials[cell_layer, 0], materials[cell_layer, 1]
    mesh.nodes_coords[:] = np.column_stack((x, y))
    mesh.centers = mesh.build_centers()
    # Associate each external edge with its unique incident MAIN cell by endpoint topology.
    main = mesh.strata[1].p_nodes.reshape(-1, (order+1)**2)
    boundary = mesh.strata[2].p_nodes.reshape(-1, order+1)
    ref = line.p_nodes.coords[:, 0]
    left, right = int(np.argmin(ref)), int(np.argmax(ref))
    edge_owner: Dict[Tuple[int, int], int] = {}
    for element, nodes in enumerate(main.reshape(-1, order+1, order+1)):
        corners = (int(nodes[left, left]), int(nodes[left, right]), int(nodes[right, right]), int(nodes[right, left]))
        for a, b in zip(corners, corners[1:]+corners[:1]):
            edge_owner[(min(a, b), max(a, b))] = element
    owner_ids = []
    for edge in boundary:
        a, b = int(edge[left]), int(edge[right])
        owner_ids.append(edge_owner[(min(a, b), max(a, b))])
    owners = np.asarray(owner_ids, dtype=np.int64)
    metadata: Dict[str, object] = dict(info)
    metadata.update({"kind": kind, "order": order, "model_id": model_id,
                     "nodes": int(x.size), "elements": int(main.shape[0]), "nx": 70, "ny": ny})
    return MeshCase(mesh, np.asarray(element_vp), np.asarray(element_vs),
                    np.asarray(element_vp[owners]), np.asarray(element_vs[owners]), metadata)


def main() -> None:
    summary = []
    for kind in ("raster", "curved"):
        for order in (2, 3, 4):
            case = build_case(kind, order)
            np.savez_compressed(OUTPUT / f"{kind}_p{order}_mesh.npz", nodes=case.mesh.nodes_coords,
                                cells=case.mesh.strata[1].p_nodes.reshape(-1, (order+1)**2),
                                boundary=case.mesh.strata[2].p_nodes.reshape(-1, order+1),
                                vp=case.element_vp, vs=case.element_vs)
            summary.append(case.metadata)
    (OUTPUT / "meshes.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
