"""Квадрат1050м: отражение, квадратный слой175м без/с ABC; 1с."""
from __future__ import annotations
import argparse
from importlib.metadata import version
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
from typing import cast
import numpy as np
from scripts.dev_1_4_server_preparation.profile_sweep_run import (MeshCase, SweepSpec, FloatArray, IntArray, ExperimentSpec, nodal_point_force, ricker, make_cu_model, plane_strain_parameters, SGPhysicsModel, SGPhysicsTimeData, SurfaceRecorder, SGBaseReceiverEventData, SGReceiverRunControl, SGReceiverModel, TimedResult, receiver_nodes, file_record, _write_metadata, _check_write, fields)
from scripts.dev_1_4_server_preparation import double_layer_run as wide
from scripts.dev_1_4_server_preparation import homogeneous_run as homogeneous
from scripts.dev_1_4_server_preparation.surface_runner import ALPHA_MAX_S_INV, verify_surface
from scripts.dev_1_3_point_source.mesh import FrozenMesh, load_frozen, build_sem, edges
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json, mapping, number
MESH_ROOT=PROJECT_ROOT/'data/dev_1_4_server_preparation/square_boundaries_1s_v1'
ROOT=PROJECT_ROOT/'data/dev_1_4_server_preparation/square_main_controls_1s_v1'
VARIANTS=('main_abc','zero_abc')
FACTOR=0.5684038628158912
from scripts.dev_1_4_server_preparation.no_padding_run import CoreSpec
from scripts.dev_1_4_server_preparation.surface_runner import SurfaceSpec

def alpha_values(coords: FloatArray) -> FloatArray:
    x,y=coords.T
    sx=np.maximum.reduce((-x,x-700.,np.zeros_like(x)))/175.
    sy=np.maximum(-y-875.,0.)/175.
    return np.asarray(ALPHA_MAX_S_INV*(sx*sx+sy*sy),dtype=np.float64)

def execute(case: MeshCase, spec: SurfaceSpec, output: Path, metadata: dict[str, object],
            edges: IntArray, variant: str, save_fields: bool) -> dict[str, object]:
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    ids = np.arange(len(coords), dtype=np.int64)
    receivers = receiver_nodes(case)
    experiment = ExperimentSpec()
    source = nodal_point_force(coords, 350., experiment.force_amplitude_n_per_m)
    model, native, success_code = make_cu_model(case, spec)
    times = np.linspace(0., spec.t_end_s, spec.steps + 1)
    forces = np.zeros((len(times), 1, 2), dtype=np.float64)
    forces[:, 0, 1] = source.force_y * ricker(times)
    model.add_physics(SGPhysicsModel.from_entity_values("material", 11000, case.mesh, 1,
        np.arange(len(case.element_vp), dtype=np.int64), plane_strain_parameters(case.element_vp, case.element_vs)))
    physics_ids=[11000]
    if True:
        model.add_physics(SGPhysicsModel.from_entity_values("outer_abc",32000,case.mesh,2,edges,
            np.column_stack((experiment.density_kg_m3*case.boundary_vp[edges],experiment.density_kg_m3*case.boundary_vs[edges]))))
        physics_ids.append(32000)
    alpha=FACTOR*alpha_values(coords) if variant == 'main_abc' else np.zeros(len(coords))
    if variant == 'main_abc':
        model.add_physics(SGPhysicsModel("mass_damping",30001,case.mesh,0,ids,alpha[:,None]))
        physics_ids.append(30001)
    physics_ids.append(20200)
    model.add_physics(SGPhysicsModel("point_ricker", 20200, case.mesh, 0, np.array([source.node_id], dtype=np.int64),
        forces, time=SGPhysicsTimeData(0., spec.t_end_s, len(times), 3)))
    metadata.update(source_node_id=source.node_id, source_input_sha256=hashlib.sha256(forces.tobytes()).hexdigest(),
        source_input_first_force_y_n_per_m=float(forces[0, 0, 1]), native=file_record(native),
        nodes_sha256=hashlib.sha256(coords.tobytes()).hexdigest(), physics_ids=physics_ids,
        abc_edge_count=len(edges), nodal_alpha_sha256=hashlib.sha256(alpha.tobytes()).hexdigest())
    surface = SurfaceRecorder.create(spec)
    recorder = fields.FieldRecorder.create(output, spec, len(coords)) if save_fields else None
    if recorder is not None:
        np.save(output / "nodes.npy", coords, allow_pickle=False)
        np.save(output / "field_time_s.npy", recorder.times, allow_pickle=False)

    def surface_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        surface.append(frame, event)
        if (surface.count - 1) % 500 == 0:
            print(json.dumps({"stage": "surface", "profile": variant, "time_s": float(surface.times[surface.count - 1])}), flush=True)

    def field_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        if recorder is None:
            raise AssertionError("Unexpected field event")
        recorder.append(frame, event)

    model.add_receiver(SGReceiverModel.from_positions("surface_velocity", case.mesh, 0, receivers,
        np.array([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=spec.substeps, handler=surface_handler))
    if recorder is not None:
        model.add_receiver(SGReceiverModel.from_positions("all_node_velocity", case.mesh, 0, ids,
            np.array([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps, stride=10*spec.substeps, handler=field_handler))
    _write_metadata(output / "metadata.started.json", metadata, None)
    model.prepare()
    print(json.dumps({"stage": "prepared", "profile": variant, "nodes": len(coords), "steps": spec.steps, "fields": save_fields}), flush=True)
    try:
        result = cast(TimedResult, model.run(sync=True))
        if int(result.code) != success_code or result.steps_count != spec.steps or not np.isclose(result.current_time, spec.t_end_s, atol=2e-10, rtol=0):
            raise RuntimeError(f"Integration did not finish: {result}")
        surface.finish()
        if recorder is not None:
            for index, frame in enumerate(recorder.values):
                if not np.array_equal(frame[receivers], surface.velocity[10*index]):
                    raise ValueError("All-node and surface values disagree")
            recorder.finish()
    finally:
        if recorder is not None:
            recorder.flush()
    partial = output / "surface.partial"
    _check_write(output, surface.velocity.nbytes + 100_000, None)
    with partial.open("xb") as stream:
        np.savez(stream, time_s=surface.times, coordinates_m=coords[receivers], node_ids=receivers, velocity_m_s=surface.velocity)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output / "surface.npz")
    return {"runtime": {"result_code": int(result.code), "time_end_code": success_code,
        "steps_count": result.steps_count, "current_time_s": result.current_time},
        "recorded_counts": {"surface": surface.count, "fields": recorder.count if recorder else 0, "source_history": 0},
        "surface_file": fields.stream_record(output / "surface.npz"), "surface_file_bytes": (output / "surface.npz").stat().st_size,
        "field_files": {name: fields.stream_record(output / name) for name in ("velocity_m_s.npy", "nodes.npy", "field_time_s.npy")} if recorder else {},
        "all_node_surface_values_exact": True if recorder else None}

def run(variant: str) -> None:
    prep=read_json(MESH_ROOT/'preparation.json')
    assert prep['mesh']==file_record(MESH_ROOT/'mesh_q2.npz')
    out=ROOT/variant;out.mkdir(exist_ok=False)
    case=build_sem(load_frozen(MESH_ROOT/'mesh_q2.npz'),4)
    case.metadata['dataset']='EFVB'
    boundary=np.asarray(case.mesh.strata[2].p_nodes).reshape(-1,5)
    selected=np.flatnonzero(~np.all(np.abs(case.mesh.nodes_coords[boundary,1])<1e-9,axis=1)).astype(np.int64)
    for row in boundary[selected]:
        xy=case.mesh.nodes_coords[row]
        assert any(np.allclose(xy[:,ax],v,atol=1e-8,rtol=0) for ax,v in ((0,-175.),(0,875.),(1,-1050.)))
    spec: SurfaceSpec = SweepSpec('EFVB',0,350.,96,1.,alpha_max_s_inv=FACTOR*ALPHA_MAX_S_INV) if variant=='main_abc' else CoreSpec('EFVB',0,350.,96,1.)
    metadata:dict[str,object]={'status':'started','variant':variant,'run_spec':asdict(spec),'mesh':case.metadata,'mesh_evidence':file_record(MESH_ROOT/'mesh_q2.npz'),'physical_model':homogeneous.PHYSICAL_MODEL,'initial_state':'native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3','dt_s':spec.dt_s,'steps_count':spec.steps,'short_test':False,'components':['Vx','Vy'],'receiver_count':70,'code':file_record(Path(__file__)),'boundary_note':'All omitted exterior conditions are natural traction-free reflecting boundaries; top always free','amplitude_factor':FACTOR if variant=='main_abc' else 0.,'profile':'k*alpha0*((max(-x,x-700,0)/175)^2 + (max(-y-875,0)/175)^2)'}
    metadata['provenance']={'semgpu2':version('semgpu2'),'files':[file_record(Path(__file__))]}
    start=perf_counter()
    metadata.update(execute(case,spec,out,metadata,selected,variant,False))
    metadata.update(status='completed',wall_seconds=perf_counter()-start)
    _write_metadata(out/'metadata.json',metadata,None)
    verify_surface(out,spec)
    target=read_json(PROJECT_ROOT/'data/dev_1_4_server_preparation/profile_golden_1s_3060_v1/runs/eval_09/metadata.json')
    assert metadata['source_input_sha256']==target['source_input_sha256']
    assert mapping(metadata['native'])['sha256']==mapping(target['native'])['sha256']
    expected=[11000,32000,30001,20200] if variant=='main_abc' else [11000,32000,20200]
    assert metadata['physics_ids']==expected
    wide.write_json(out/'validation.json',{'status':'passed','metadata':file_record(out/'metadata.json'),'physics_ids':expected,'same_source_native':True,'scope':'Complete finite surface and configured physics; no independent solver validation'})
    print(json.dumps({'completed':variant,'wall_seconds':metadata['wall_seconds']}),flush=True)

def main() -> None:
    ROOT.mkdir(parents=True,exist_ok=False)
    wide.write_json(ROOT/'plan.json',{'factor':FACTOR,'variants':list(VARIANTS),'mesh':file_record(MESH_ROOT/'mesh_q2.npz'),'code':file_record(Path(__file__)),'duration_s':1.,'window_s':[.5,1.],'note':'Main factor retained from earlier geometry, not reoptimized on square1050'})
    for variant in VARIANTS:run(variant)

if __name__=='__main__':main()
