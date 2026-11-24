"""Качество сеток №65 в основной области. Для чтения FC добавить FCModel/src в PYTHONPATH."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize
import numpy as np
from numpy.typing import NDArray
from fc_model import FCModel
from scipy.interpolate import BSpline
from scipy.spatial import cKDTree

from scripts.dev_1_3_point_source.mesh import FrozenMesh, basis, load_frozen
from scripts.dev_1_6_p67s_showcase.mesh import core_case
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, sequence

F = NDArray[np.float64]
I = NDArray[np.int64]
BASE = Path("dev_1_6_p67s_showcase")
SERIES = "fc_mesh_quality_v1"
FC = PROJECT_ROOT / "test_ECVB00065.fc"
OLD = PROJECT_ROOT / "data" / BASE / "cpu_v2/ECVB00065/mesh_q2.npz"
PERIMETER = [0, 2, 8, 6]


@dataclass
class Sample:
    local: F  # Q1 and Q2 stored in the same 3x3 tensor grid
    vertices: I  # counterclockwise topology, zero-based internal IDs
    blocks: I
    ids: I
    reversed_blocks: dict[int, int]


def load_fc() -> Sample:
    mesh = FCModel.load(str(FC)).mesh
    assert set(mesh.elements) == {"MITC4"} and np.all(mesh.nodes_xyz[:, 2] == 0)
    elements = list(mesh.elements["MITC4"].values())
    assert all(e.order == 1 for e in elements)
    lookup = {int(n): i for i, n in enumerate(mesh.nodes_ids)}
    vertices = np.array([[lookup[n] for n in e.nodes] for e in elements])
    blocks = np.array([e.block for e in elements])
    points = mesh.nodes_xyz[vertices, :2]
    edges = np.roll(points, -1, axis=1) - points
    determinants = np.linalg.det(np.stack((edges, -np.roll(edges, 1, axis=1)), axis=-1))
    cw = np.all(determinants < 0, axis=1)
    assert np.all(cw | np.all(determinants > 0, axis=1))
    vertices[cw] = vertices[cw][:, [0, 3, 2, 1]]
    points = mesh.nodes_xyz[vertices, :2]
    mid = (points + np.roll(points, -1, axis=1)) / 2
    local = np.empty((len(points), 9, 2))
    local[:, PERIMETER] = points
    local[:, [1, 5, 7, 3]] = mid
    local[:, 4] = points.mean(axis=1)
    return Sample(local, vertices, blocks, np.array([e.id for e in elements]),
                  dict(Counter(int(block) for block in blocks[cw])))


def from_frozen(mesh: FrozenMesh) -> Sample:
    centers = mesh.nodes[mesh.cells[:, 4]]
    select = (centers[:, 0] > 0) & (centers[:, 0] < 1400) & (centers[:, 1] > -700)
    cells = mesh.cells[select]
    return Sample(mesh.nodes[cells], cells[:, PERIMETER], mesh.regions[select], np.flatnonzero(select), {})


def derivatives(local: F, q: F) -> tuple[F, F]:
    w = basis(q)
    d = np.column_stack((q - .5, -2*q, q + .5))
    coords = local.reshape(-1, 3, 3, 2)
    return (np.einsum("ai,bj,eijc->eabc", w, d, coords),
            np.einsum("ai,bj,eijc->eabc", d, w, coords))


def distribution(values: F) -> dict[str, float]:
    return dict(zip(("min", "p01", "p05", "median", "p95", "p99", "max"),
                    map(float, np.quantile(values, [0, .01, .05, .5, .95, .99, 1])), strict=True))


def inspect(sample: Sample, cad: list[dict]) -> tuple[dict, F]:
    local, vertices = sample.local, sample.vertices
    center = local[:, 4]
    gll = np.r_[-1., np.polynomial.legendre.Legendre.basis(6).deriv().roots(), 1.]
    dx, dy = derivatives(local, gll)
    jac = dx[..., 0]*dy[..., 1] - dx[..., 1]*dy[..., 0]
    assert jac.min() > 0
    dot = (dx*dy).sum(axis=-1)
    a, b = (dx*dx).sum(axis=-1), (dy*dy).sum(axis=-1)
    sj = (jac / np.sqrt(a*b)).min(axis=(1, 2))
    largest_sq = (a+b + np.hypot(a-b, 2*dot))/2
    condition = (largest_sq/jac).max(axis=(1, 2))
    thickness = (2*jac/np.sqrt(largest_sq)).min(axis=(1, 2))
    g, weights = np.polynomial.legendre.leggauss(3)
    gx, gy = derivatives(local, g)
    area = np.einsum("eij,i,j->e", gx[..., 0]*gy[..., 1]-gx[..., 1]*gy[..., 0], weights, weights)
    assert np.isclose(area.sum(), 980000, rtol=0, atol=1e-6)
    n = len(vertices)
    directed = np.stack((vertices, np.roll(vertices, -1, axis=1)), axis=-1).reshape(-1, 2)
    _, idx, inv, counts = np.unique(np.sort(directed, axis=1), axis=0, return_index=True,
                                     return_inverse=True, return_counts=True)
    orientations = np.bincount(inv, weights=np.sign(directed[:, 1]-directed[:, 0]))
    assert counts.max() == 2 and np.all(orientations[counts==2] == 0)
    assert len(np.unique(vertices)) - len(counts) + n == 1
    edge_nodes = local[:, [[0,1,2],[2,5,8],[8,7,6],[6,3,0]]].reshape(-1,3,2)
    boundary = edge_nodes[idx[counts==1]]
    sides = np.array([np.all(np.abs(boundary[:,:,axis]-value)<1e-8, axis=1) for axis,value in
                      ((0,0),(0,1400),(1,-700),(1,0))])
    assert np.all(sides.any(axis=0))
    top = boundary[sides[-1]]
    top_x = np.sort(top[:, [0,2], 0], axis=1)
    top_x = top_x[np.argsort(top_x[:,0])]
    assert np.allclose(top_x,np.column_stack((np.arange(140)*10, np.arange(1,141)*10)),rtol=0,atol=1e-8)
    order = np.argsort(inv, kind="stable")
    offsets = np.r_[0, np.cumsum(counts)]
    two = np.flatnonzero(counts==2)
    left, right = order[offsets[two]], order[offsets[two]+1]
    assert np.allclose(edge_nodes[left],edge_nodes[right,::-1],atol=1e-8,rtol=0)
    pairs = np.sort(np.stack((sample.blocks[left//4],sample.blocks[right//4]),axis=1),axis=1)
    errors = []
    midpoint_error = 0.0
    vertex_error = 0.0
    worst_distance = -1.0
    worst_interface: dict[str, object] = {}
    # Some pairs have several CAD segments after corner preservation.
    for pair in np.unique(pairs[pairs[:,0]!=pairs[:,1]],axis=0):
        chosen = left[np.all(pairs == pair,axis=1)]
        p = np.einsum("ij,ejd->eid",basis(np.linspace(-1,1,21)),edge_nodes[chosen]).reshape(-1,2)
        distances = []
        for c in cad:
            if sorted(c["adjacent_regions"])!=pair.tolist():
                continue
            poles = np.asarray(c["control_points_m"]).copy()
            poles[:,1] -= 700
            spline = BSpline(c["knots"],poles,c["degree"])
            params = np.linspace(0,1,1025)
            t = params[cKDTree(spline(params)).query(p)[1]]
            first = spline.derivative()
            for _ in range(8):
                tangent = first(t)
                t = np.clip(t-((spline(t)-p)*tangent).sum(axis=1)/(tangent*tangent).sum(axis=1),0,1)
            distances.append(np.linalg.norm(spline(t)-p,axis=1))
        gap = np.min(distances, axis=0).reshape(-1,21)
        midpoint_error = max(midpoint_error,float(gap[:,10].max()))
        vertex_error = max(vertex_error,float(gap[:,[0,-1]].max()))
        peak = int(np.argmax(gap))
        if float(gap.max()) > worst_distance:
            worst_distance = float(gap.max())
            worst_interface = {"blocks":pair.tolist(),"xy_m":p[peak].tolist()}
        errors.extend(gap.ravel().tolist())
    worst = int(np.argmax(condition))
    return {"elements": n, "corner_nodes":len(np.unique(vertices)), "clockwise_input_by_block":sample.reversed_blocks,
            "area_m2":float(area.sum()),"top_intervals_10m":len(top),
            "sqrt_area_m":distribution(np.sqrt(area)),"min_scaled_jacobian":distribution(sj),
            "max_condition_J":distribution(condition),"min_2sigma_m":distribution(thickness),
            "sampled_interface_error_to_own_cad_m":distribution(np.array(errors)) if errors else {},
            "interface_midpoint_max_error_m":midpoint_error,
            "interface_vertex_max_error_m":vertex_error,"worst_interface_location":worst_interface,
            "elements_sj_below_0.8":int((sj<.8).sum()),"elements_sj_below_0.5":int((sj<.5).sum()),
            "worst_condition":{"id":int(sample.ids[worst]),"block":int(sample.blocks[worst]),
                               "center_xy_m":center[worst].tolist()},
            "fraction_h_8_to_12":float(np.mean((area>=64)&(area<=144)))}, sj


def main() -> None:
    original = json.loads(OLD.with_name("geometry.json").read_text())["fit"]["curves"]
    corrected, meta = core_case("ECVB00065",preserve_corners=True)
    corrected_curves = [mapping(item) for item in sequence(mapping(meta["fit"])["curves"])]
    cases = [(from_frozen(load_frozen(OLD)),original),
             (from_frozen(corrected),corrected_curves), (load_fc(),corrected_curves)]
    data_path = PROJECT_ROOT/"data"/BASE/SERIES
    img_path = PROJECT_ROOT/"img"/BASE/SERIES
    data_path.mkdir(parents=True,exist_ok=True)
    img_path.mkdir(parents=True,exist_ok=True)
    result = {}
    fig, axes = plt.subplots(1,2,figsize=(12,3.8),layout="constrained")
    for idx,(sample,cad) in enumerate(cases):
        key = ("historical_q2","same_cad_split_q2","fc")[idx]
        result[key], sj = inspect(sample,cad)
        if idx in (1,2):
            ax=axes[idx-1]
            polys=PolyCollection(list(sample.local[:,PERIMETER]),array=sj,cmap="viridis",norm=Normalize(0,1),
                                 edgecolors="face",linewidths=.12)
            ax.add_collection(polys)
            ax.autoscale_view()
            ax.set(aspect="equal",xlabel="x, м",ylabel="y, м",
                   title=("Разбиение треугольников · Q2" if idx==1 else "Сетка Fidesys · Q1"))
    fig.colorbar(polys,ax=axes.tolist(),shrink=.8,label="Минимальный scaled Jacobian")
    fig.savefig(img_path/"scaled_jacobian.png",dpi=170)
    plt.close(fig)
    result["input_sha256"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (FC,OLD)}
    (data_path/"quality.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
