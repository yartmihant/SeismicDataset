"""Independent p3/p4/p5 and temporal controls with native CFL; no production queue access."""
from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter
from typing import cast
import numpy as np
from numpy.typing import NDArray
from semgpu2.model.model import SGModel
from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu import _native
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig, SGSpectrumData, SGRunCodeEnum
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.solver import SGSolver
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import load_frozen
from scripts.dev_1_4_server_preparation.coarse_order_mesh import build_sem
from scripts.dev_1_3_point_source.physics import damping_alpha, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import TimedResult, abc_edges, checked_time, exact_surface_nodes, file_record, write_json
from scripts.dev_1_4_server_preparation.production_surface import ALPHA_MAX_S_INV
from scripts.dev_1_4_server_preparation.curved_mesh import build_curved
from scripts.dev_1_4_server_preparation.model_input import read_model
from scripts.project_paths import PROJECT_ROOT


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--model',type=int,choices=(65,12404),required=True)
    p.add_argument('--grid',choices=('coarse','refined'),required=True)
    p.add_argument('--order',type=int,choices=(3,4,5,6,7),default=4)
    p.add_argument('--beta',type=float,choices=(0.,1/27),default=1/27)
    p.add_argument('--safety',type=float,default=.8)
    p.add_argument('--substeps',type=int)
    args=p.parse_args()
    root=args.root.resolve();root.relative_to(PROJECT_ROOT/'data/dev_1_4_server_preparation')
    base=root/f'{args.model:05d}'/args.grid
    mesh_path=base/'mesh_q2.npz'
    suffix=f'fixedk{args.substeps}' if args.substeps else f'cfl{args.safety:g}'
    out=base/f'p{args.order}_beta{args.beta:.8f}_{suffix}'
    out.mkdir(exist_ok=False)
    start=perf_counter()
    case=build_sem(load_frozen(mesh_path),args.order);mesh=case.mesh
    config=SGSolverConfig(t_begin=0.,t_end=1.,cfl_safety=args.safety if args.substeps is None else None,steps_count=1000*args.substeps if args.substeps else None,tact_dt=.001,kim_beta=args.beta,kim_gamma=.5)
    solver=SGSolver(config);model=SGModel(mesh,solver)
    edges=abc_edges(case)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(case.element_vp),dtype=np.int64),plane_strain_parameters(case.element_vp,case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values('abc',32000,mesh,2,edges,np.column_stack((1000*case.boundary_vp[edges],1000*case.boundary_vs[edges]))))
    model.add_physics(SGPhysicsModel('damping',30001,mesh,0,np.arange(len(mesh.nodes_coords),dtype=np.int64),damping_alpha(mesh.nodes_coords,ALPHA_MAX_S_INV)[:,None]))
    source=nodal_point_force(mesh.nodes_coords,350.,1.)
    # Identical dense force table for every p/beta/dt: avoid changing source interpolation.
    source_times=np.linspace(0.,1.,1000001)
    values=np.zeros((len(source_times),1,2));values[:,0,1]=source.force_y*ricker(source_times)
    model.add_physics(SGPhysicsModel('source',20200,mesh,0,np.array([source.node_id],dtype=np.int64),values,time=SGPhysicsTimeData(0.,1.,len(source_times),3)))
    coords=np.column_stack((np.arange(141)*5.,np.zeros(141)))
    boundary=mesh.strata[2]; edge_nodes=boundary.p_nodes.reshape(-1,args.order+1)
    top=np.flatnonzero(np.all(np.abs(mesh.nodes_coords[edge_nodes,1])<1e-8,axis=1))
    entities=[];refs=[]
    for x in coords[:,0]:
        found=[int(e) for e in top if mesh.nodes_coords[edge_nodes[e],0].min()-1e-8<=x<=mesh.nodes_coords[edge_nodes[e],0].max()+1e-8]
        assert found
        e=found[0];xx=mesh.nodes_coords[edge_nodes[e],0];ref=2*(x-xx[0])/(xx[-1]-xx[0])-1
        entities.append(e);refs.append([ref])
    entity_ids=np.asarray(entities,dtype=np.int64);reference_points=np.asarray(refs,dtype=np.float64)
    shape=mesh.evaluate_shape_values(2,entity_ids,reference_points)
    assert np.allclose(shape.sum(axis=1),1,atol=1e-12)
    for degree in range(args.order+1):
        nodal=(mesh.nodes_coords[edge_nodes[entity_ids],0]/700)**degree
        assert np.allclose((shape[:,:args.order+1]*nodal).sum(axis=1),(coords[:,0]/700)**degree,atol=2e-12,rtol=0)

    times=np.linspace(0.,1.,1001);velocity=np.empty((1001,141,2));count=0
    def handler(frame: NDArray[np.float64],event: SGBaseReceiverEventData,control: SGReceiverRunControl) -> None:
        nonlocal count
        assert count<1001 and frame.size==282
        checked_time(frame,event,float(times[count]));velocity[count]=frame.reshape(2,141).T;count+=1
    model.add_receiver(SGReceiverModel.from_reference_points('surface',mesh,2,entity_ids,reference_points,np.array([4,5],dtype=np.int32),begin_tact=0,end_tact=1000,stride=1,handler=handler))
    model.prepare();s=solver.spectrum;assert isinstance(s,SGSpectrumData)
    report:dict[str,object]={'grid':args.grid,'receiver_method':'SEM edge polynomial interpolation at fixed 5m positions; monomials through degree p verified','model':args.model,'order':args.order,'beta':args.beta,'gamma':.5,'safety':args.safety if args.substeps is None else None,'dt_s':s.step_dt,'steps':s.steps_count,'steps_per_ms':s.steps_per_tact,'lambda_max':s.lambda_max,'lambda_max_damping':s.lambda_max_damping,'mesh':file_record(mesh_path),'native':file_record(Path(_native.__file__)),'code':file_record(Path(__file__)),'alpha_max_s_inv':ALPHA_MAX_S_INV,'source_table_step_s':1e-6,'source_x_m':350.,'nodes':len(mesh.nodes_coords),'status':'prepared'}
    write_json(out/'started.json',report);print(args.model,args.order,args.beta,args.safety,s.steps_count,flush=True)
    run_start=perf_counter();run_result=cast(TimedResult,model.run(sync=True));end=perf_counter()
    assert int(run_result.code)==int(SGRunCodeEnum.TIME_END) and run_result.steps_count==s.steps_count and count==1001
    assert np.isfinite(velocity).all() and np.all(velocity[0]==0)
    np.savez(out/'surface.npz',time_s=times,coordinates_m=coords,velocity_m_s=velocity)
    report.update(status='completed',run_seconds=end-run_start,total_seconds=end-start,surface=file_record(out/'surface.npz'))
    write_json(out/'result.json',report);print('completed',out.name,end-start,flush=True)


if __name__=='__main__':
    main()
