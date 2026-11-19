"""Isolated disconnected-mesh batching experiment; never touches the production queue."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import cast

import numpy as np
from numpy.typing import NDArray
from semgpu2.model.mesh import SGMeshModel, SGMeshStratumData
from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
from scripts.dev_1_3_point_source.physics import damping_alpha, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import TimedResult, abc_edges, checked_time, file_record, write_json
from scripts.dev_1_4_server_preparation.production_surface import ALPHA_MAX_S_INV, SurfaceSpec, make_cu_model, receiver_nodes, validate_stability
from scripts.project_paths import PROJECT_ROOT


@dataclass(frozen=True)
class Packed:
    case: MeshCase
    nodes_per_copy: int
    copies: int


def pack(case: MeshCase, copies: int) -> Packed:
    """Disjoint node IDs; identical coordinates avoid translation roundoff."""
    if not 1 <= copies <= 20:
        raise ValueError('Probe supports 1..20 copies')
    mesh = case.mesh
    nodes = len(mesh.nodes_coords)
    strata = {}
    for key, s in mesh.strata.items():
        strata[key] = SGMeshStratumData(s.signature,
            np.concatenate([s.p_nodes + i*nodes for i in range(copies)]),
            np.concatenate([s.p_node_offsets[:-1]+i*len(s.p_nodes) for i in range(copies)]+[np.array([copies*len(s.p_nodes)],dtype=np.int64)]),
            np.tile(s.pet_ids,copies),
            np.concatenate([s.pet_offsets[:-1]+i*len(s.pet_ids) for i in range(copies)]+[np.array([copies*len(s.pet_ids)],dtype=np.int64)]),
            np.tile(s.geom_tags,copies))
        # Every element refers to exactly one component, with no shared node IDs.
        reshaped = strata[key].p_nodes.reshape(copies,-1)
        assert all(np.array_equal(reshaped[i]-i*nodes,s.p_nodes) for i in range(copies))
    combined = SGMeshModel(np.tile(mesh.nodes_coords,(copies,1)),np.tile(mesh.nodes_tags,(copies,1)),list(mesh.pets),strata,
                           centers={k:np.tile(v,(copies,1)) for k,v in mesh.centers.items()})
    return Packed(MeshCase(combined,np.tile(case.element_vp,copies),np.tile(case.element_vs,copies),
                          np.tile(case.boundary_vp,copies),np.tile(case.boundary_vs,copies),dict(case.metadata)),nodes,copies)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh',type=Path,required=True)
    parser.add_argument('--stability',type=Path,required=True)
    parser.add_argument('--substeps',type=int,default=88)
    parser.add_argument('--copies',type=int,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    parser.add_argument('--references',type=Path)
    parser.add_argument('--source-index',type=int,default=0,choices=range(5))
    parser.add_argument('--end',type=float,default=.1)
    args=parser.parse_args()
    out=args.out.resolve();out.relative_to(PROJECT_ROOT/'data/dev_1_4_server_preparation')
    out.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    case=build_sem(load_frozen(args.mesh),4)
    case.metadata['dataset']='EFVB'
    spec=SurfaceSpec('EFVB',0,350.,args.substeps,args.end)
    validate_stability(case,spec,args.mesh,args.stability)
    packed=pack(case,args.copies); mesh=packed.case.mesh
    n=packed.nodes_per_copy; b=args.copies
    model,native,success=make_cu_model(packed.case,spec)
    edges=abc_edges(case); edge_count=len(case.boundary_vp)
    all_edges=np.concatenate([edges+i*edge_count for i in range(b)])
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(packed.case.element_vp),dtype=np.int64),plane_strain_parameters(packed.case.element_vp,packed.case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values('abc',32000,mesh,2,all_edges,np.tile(np.column_stack((1000*case.boundary_vp[edges],1000*case.boundary_vs[edges])),(b,1))))
    model.add_physics(SGPhysicsModel('damping',30001,mesh,0,np.arange(n*b,dtype=np.int64),np.tile(damping_alpha(case.mesh.nodes_coords,ALPHA_MAX_S_INV),b)[:,None]))
    source=nodal_point_force(case.mesh.nodes_coords,350.,1.)
    times=np.linspace(0,spec.t_end_s,spec.steps+1)
    source_x=[0.,175.,350.,525.,700.]
    positions=[source_x[(args.source_index+i)%5] for i in range(b)]
    sources=[nodal_point_force(case.mesh.nodes_coords,x,1.) for x in positions]
    scales=np.ones(b)
    values=np.zeros((len(times),b,2));values[:,:,1]=source.force_y*ricker(times)[:,None]*scales[None,:]
    model.add_physics(SGPhysicsModel('sources',20200,mesh,0,np.array([v.node_id+i*n for i,v in enumerate(sources)],dtype=np.int64),values,time=SGPhysicsTimeData(0.,spec.t_end_s,len(times),3)))
    ids=np.concatenate([receiver_nodes(case)+i*n for i in range(b)])
    output_times=np.linspace(0,spec.t_end_s,round(spec.t_end_s/.001)+1)
    velocity=np.empty((len(output_times),b,70,2),dtype=np.float64)
    count=0
    def handler(frame: NDArray[np.float64],event: SGBaseReceiverEventData,control: SGReceiverRunControl) -> None:
        nonlocal count
        if count>=len(output_times) or frame.size!=b*140:
            raise ValueError('Unexpected frame')
        checked_time(frame,event,float(output_times[count]))
        velocity[count]=frame.reshape(2,-1).T.reshape(b,70,2)
        count+=1
    model.add_receiver(SGReceiverModel.from_positions('velocity',mesh,0,ids,np.array([4,5],dtype=np.int32),begin_tact=0,end_tact=spec.steps,stride=spec.substeps,handler=handler))
    prepared=perf_counter();model.prepare();run_start=perf_counter()
    result=cast(TimedResult,model.run(sync=True));finish=perf_counter()
    assert int(result.code)==success and result.steps_count==spec.steps and count==len(output_times)
    assert np.isfinite(velocity).all() and np.all(velocity[0]==0)
    metrics: list[dict[str,object]]=[]
    if args.reference or args.references:
        for i,scale in enumerate(scales):
            ref_path=args.reference if args.reference else args.references/f'single{(args.source_index+i)%5}'/'surface.npz'
            with np.load(ref_path) as z: ref=z['velocity'][:,0]
            expected=scale*ref;diff=velocity[:,i]-expected
            relative=float(np.linalg.norm(diff))/max(float(np.linalg.norm(expected)),1e-300)
            peak=float(np.max(np.abs(diff)))
            assert np.allclose(velocity[:,i],expected,rtol=1e-8,atol=1e-18),(i,relative,peak)
            if scale==0: assert np.all(velocity[:,i]==0)
            metrics.append({'copy':i,'scale':float(scale),'relative_l2':relative,'max_abs':peak})
    np.savez(out/'surface.npz',time=output_times,velocity=velocity)
    write_json(out/'result.json',{'status':'passed','copies':b,'source_x_m':positions,'end_s':spec.t_end_s,'substeps':spec.substeps,'assembly_seconds':prepared-start,'prepare_seconds':run_start-prepared,'run_seconds':finish-run_start,'total_seconds':finish-start,'per_source_seconds':(finish-start)/b,'nodes':n*b,'comparisons':metrics,'native':file_record(native),'code':file_record(Path(__file__)),'mesh':file_record(args.mesh),'scope':'Identical EFVB0 geometry, cyclic five independent sources; disjoint node IDs, no geometry translation'})
    print((out/'result.json').read_text(),flush=True)


if __name__=='__main__':
    main()
