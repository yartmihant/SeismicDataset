"""Точные B-сплайны и общие границы блоков ECVB, в метрах, верх y=0."""
from __future__ import annotations

import numpy as np
from scripts.dev_1_4_server_preparation.curved_mesh import layer_sequence, layered_core, shared_curves
from scripts.dev_1_4_server_preparation.model_input import ModelInput
from scripts.reporting import mapping, sequence


def geometry(model: ModelInput) -> dict:
    layers = layer_sequence(model.vp, model.vs)
    curves: list[dict] = []
    regions: list[dict] = []
    if layers is None:
        source = shared_curves(model, preserve_corners=True)
        regions = [mapping(r) for r in sequence(source["regions"])]
        for raw in sequence(source["curves"]):
            c = mapping(raw)
            poles = np.array(c["control_points_m"], dtype=np.float64) * [2., 1.] - [0., 700.]
            curves.append({k:c[k] for k in ("id","degree","knots","adjacent_regions")} |
                          {"control_points_m":poles.tolist()})
        method = "shared_curves_preserve_corners"
    else:
        labels, materials, _ = layers
        fit = layered_core(model,layers)
        # Greville abscissae represent the affine x-coordinate exactly in the depth spline basis.
        boundaries = [[[0., 0.],[1400., 0.]]]
        knot_vectors = [[0.,0.,1.,1.]]
        for item in sequence(fit.report["splines"]):
            spline = mapping(item)
            knots = np.array(spline["knots"])
            depth = np.array(spline["coefficients"])
            x = [2.*np.mean(knots[i+1:i+4]) for i in range(len(depth))]
            boundaries.append(np.column_stack((x,-depth)).tolist())
            knot_vectors.append((knots/700.).tolist())
        boundaries.append([[0.,-700.],[1400.,-700.]])
        knot_vectors.append([0.,0.,1.,1.])

        def add(points: list, knots: list, adjacent: list[int]) -> int:
            key = len(curves)+1
            curves.append({"id":key,"control_points_m":points,"knots":knots,
                           "degree":len(knots)-len(points)-1,"adjacent_regions":adjacent})
            return key

        for j,(points,knot_list) in enumerate(zip(boundaries,knot_vectors,strict=True)):
            add(points,knot_list,[j if j else -1,j+1 if j<len(labels) else -1])
        for i,label in enumerate(labels):
            top,bottom=boundaries[i:i+2]
            r=i+1
            left=add([top[0],bottom[0]],[0.,0.,1.,1.],[r,-1])
            right=add([bottom[-1],top[-1]],[0.,0.,1.,1.],[r,-1])
            vp,vs=materials[label]
            regions.append({"id":r,"vp":float(vp),"vs":float(vs),"loops":[[-r,left,r+1,right]]})
        method = "layered_locked_depth_splines"
    return {"schema":"fidesys.ecvb.quad8.v1","model":model.model_id,"source":model.provenance,
            "method":method,"curves":curves,"regions":regions}
