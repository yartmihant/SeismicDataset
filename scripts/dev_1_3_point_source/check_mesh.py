"""Адресная проверка восьми GLL-вложений и топологии новой сетки №65."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from scripts.dev_1_3_point_source.mesh import ROOT, FrozenMesh, audit, build_sem, edges, load_frozen
from scripts.dev_1_3_point_source.physics import nodal_point_force
from scripts.project_paths import DATA_12, PROJECT_ROOT
from scripts.reporting import mapping, sequence, number


def topology(mesh: FrozenMesh) -> dict[str, object]:
    """Проверить связность регионов, контакты и короткий контакт материалов3/4."""
    directed, inverse, counts, _ = edges(mesh)
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    two = counts == 2
    first, second = order[starts[two]], order[starts[two]+1]
    a, b = first//4, second//4
    region_contacts = sorted({tuple(sorted((int(mesh.regions[i]), int(mesh.regions[j])))) for i, j in zip(a, b, strict=True) if mesh.regions[i] != mesh.regions[j]})
    same = mesh.regions[a] == mesh.regions[b]
    graph = coo_matrix((np.ones(2*int(same.sum())), (np.r_[a[same], b[same]], np.r_[b[same], a[same]])), shape=(len(mesh.cells), len(mesh.cells))).tocsr()
    _, component = connected_components(graph, directed=False)
    components = {str(int(r)): len(np.unique(component[mesh.regions == r])) for r in np.unique(mesh.regions)}
    if any(n != 1 for n in components.values()):
        raise ValueError("Disconnected material region")
    source = mapping(json.loads((DATA_12 / "model65/geometry/curves.json").read_text()))
    material_map = {int(number(r["id"])): int(number(r["material_id"])) for r in (mapping(item) for item in sequence(source["regions"]))}
    mat = np.asarray([material_map[int(r)] for r in mesh.regions])
    contact = ((mat[a] == 3)&(mat[b] == 4))|((mat[a] == 4)&(mat[b] == 3))
    local = mesh.nodes[directed[first[contact]]]
    g, w = np.polynomial.legendre.leggauss(8)
    derivative = np.column_stack((g-.5, -2*g, g+.5))
    length = float(np.einsum("ei,i->", np.linalg.norm(np.einsum("ij,ejc->eic", derivative, local), axis=-1), w))
    if not np.isclose(length, 10., atol=1e-8, rtol=0):
        raise ValueError(f"Short contact changed: {length}")
    return {"region_components": components, "region_contacts": [list(pair) for pair in region_contacts], "material3_material4_contact_m": length}


def main() -> None:
    output = ROOT / "validation.json"
    if output.exists():
        raise FileExistsError(output)
    checks: list[dict[str, object]] = []
    topologies: list[dict[str, object]] = []
    for model in (12404, 65):
        for kind in ("raster", "curved"):
            path = ROOT / f"{model:05d}_{kind}_q2.npz"
            frozen = load_frozen(path)
            record = audit(frozen)
            record["file_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            if model == 65:
                topologies.append(topology(frozen))
                record["topology"] = topologies[-1]
            embeddings: list[dict[str, object]] = []
            for p in (4, 5):
                case = build_sem(frozen, p)
                coords = case.mesh.nodes_coords
                sources = [nodal_point_force(coords, float(x), 1).node_id for x in (0, 175, 350, 525, 700)]
                receivers = [nodal_point_force(coords, float(x), 1).node_id for x in range(0, 701, 5)]
                if len(set(receivers)) != 141 or not set(sources).issubset(receivers):
                    raise ValueError("Incorrect source/receiver node set")
                if kind == "raster" and len(coords) != (210*p+1)*(175*p+1):
                    raise ValueError("Incorrect regular GLL node count")
                embeddings.append({"p": p, "global_nodes": len(coords), "sources": len(sources), "receivers": len(receivers),
                                   "exact_nodal_coordinates": True, "frozen_geometry_sha256": record["file_sha256"]})
                del case
            record["embeddings"] = embeddings
            checks.append(record)
            print(model, kind, embeddings, flush=True)
    if topologies[0]["region_contacts"] != topologies[1]["region_contacts"]:
        raise ValueError("Raster/curved region adjacency changed")
    inputs = [DATA_12 / "pilot_geometry/frozen_q2_geometry.npz", DATA_12 / "pilot_geometry/selected_model.npz",
              DATA_12 / "model65/geometry/model65.npz", DATA_12 / "model65/geometry/curves.json",
              Path(__file__), Path(__file__).with_name("mesh.py")]
    result: dict[str, object] = {"status": "passed", "native_solver_executed": False, "meshes": checks,
                               "model65_same_region_adjacency": True,
                               "inputs": [{"path": str(p.relative_to(PROJECT_ROOT)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in inputs]}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
