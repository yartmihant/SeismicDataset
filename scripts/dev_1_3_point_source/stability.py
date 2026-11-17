"""Консервативная CPU-оценка шага на фактической сетке с αM и ABC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.dev_1_2_smoothing.pilot_stability import tensor_basis
from scripts.dev_1_3_point_source.physics import damping_alpha
from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen


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
    damping = float(np.max(abc/mass + damping_alpha(coords, alpha_max)))
    dt = float(2/(np.sqrt(damping*damping + 4*stiffness) + damping))
    return {
        "mesh": str(path.resolve()), "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "order": order, "nodes": node_count, "elements": len(cells),
        "alpha_max_s_inv": alpha_max, "stiffness_bound_s_minus_2": stiffness,
        "damping_bound_s_minus_1": damping, "dt_start_s": dt,
        "substeps_per_ms": int(np.ceil(.001/dt)), "min_mass_kg_per_m": float(mass.min()),
        "quadrature_jacobian_min_m2": jacobian_min, "abc_edge_count": abc_count,
        "method": "GLL mass; absolute-factor bound sum_q |B|^T |D| |B| wJ; ABC rhoVp edge quadrature and nodal alpha; half scalar damped central-difference limit",
        "limitation": "Conservative starting bound, not the solver spectrum or a convergence result; full-cell geometry certified separately",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--order", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--alpha-max", type=float, default=float(-3*4385*np.log(.01)/175))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = estimate(args.mesh, args.order, args.alpha_max)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
