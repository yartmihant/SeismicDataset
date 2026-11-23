"""Renumber the nodes of an existing mesh before constructing a model."""
from __future__ import annotations

import numpy as np

from semgpu2.model.mesh import SGMeshModel, SGMeshStratumData
from semgpu2.solvers.solver_api import Int64Array


def with_leading_nodes(mesh: SGMeshModel, leading_nodes: Int64Array) -> tuple[SGMeshModel, Int64Array]:
    """Place the given N IDs first in order and return the mesh and old-to-new ID map."""
    count = mesh.nodes_coords.shape[0]
    leading = np.asarray(leading_nodes)
    if leading.ndim != 1 or leading.dtype != np.int64:
        raise ValueError("leading_nodes must be a one-dimensional int64 array")
    if np.any(leading < 0) or np.any(leading >= count):
        raise ValueError("leading_nodes contains an out-of-range node ID")
    if np.unique(leading).size != leading.size:
        raise ValueError("leading_nodes must contain distinct node IDs")

    remaining = np.ones(count, dtype=np.bool_)
    remaining[leading] = False
    new_to_old = np.concatenate((leading, np.flatnonzero(remaining)))
    old_to_new = np.empty(count, dtype=np.int64)
    old_to_new[new_to_old] = np.arange(count, dtype=np.int64)

    strata = {}
    for space_id, stratum in mesh.strata.items():
        if np.any(stratum.p_nodes < 0) or np.any(stratum.p_nodes >= count):
            raise ValueError(f"stratum {space_id} contains an out-of-range node ID")
        strata[space_id] = SGMeshStratumData(
            signature=stratum.signature,
            p_nodes=old_to_new[stratum.p_nodes],
            p_node_offsets=stratum.p_node_offsets.copy(),
            pet_ids=stratum.pet_ids.copy(),
            pet_offsets=stratum.pet_offsets.copy(),
            geom_tags=stratum.geom_tags.copy(),
        )
    reordered = SGMeshModel(
        mesh.nodes_coords[new_to_old],
        mesh.nodes_tags[new_to_old],
        list(mesh.pets),
        strata,
        centers={space_id: centers.copy() for space_id, centers in mesh.centers.items()},
    )
    return reordered, old_to_new
