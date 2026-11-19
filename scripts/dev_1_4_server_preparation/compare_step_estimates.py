"""Compare external certificate with native CUDA spectral step; prepare only, no run."""
from __future__ import annotations

import argparse
from math import ceil, hypot, sqrt
from pathlib import Path
from time import perf_counter

import numpy as np
from semgpu2.model.model import SGModel
from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu import _native
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.solver import SGSolver
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
from scripts.dev_1_3_point_source.physics import damping_alpha, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import abc_edges, file_record, write_json
from scripts.dev_1_4_server_preparation.production_surface import ALPHA_MAX_S_INV, SurfaceSpec, validate_stability
from scripts.dev_1_4_server_preparation.production_worker import select_substeps
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json, number


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--model-id',type=int,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    out=args.out.resolve(); out.relative_to(PROJECT_ROOT/'data/dev_1_4_server_preparation')
    out.mkdir(parents=True,exist_ok=False)
    mesh_path=args.input/'mesh_q2.npz'; certificate_path=args.input/'stability.json'
    certificate=read_json(certificate_path)
    k=select_substeps('EFVB',args.model_id,int(number(certificate['substeps_per_ms'])))
    spec=SurfaceSpec('EFVB',args.model_id,350.,k)
    case=build_sem(load_frozen(mesh_path),4); case.metadata['dataset']='EFVB'
    validate_stability(case,spec,mesh_path,certificate_path)
    config=SGSolverConfig(t_begin=0.,t_end=1.,cfl_safety=1.,tact_dt=.001)
    solver=SGSolver(config); model=SGModel(case.mesh,solver)
    mesh=case.mesh;edges=abc_edges(case)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(case.element_vp),dtype=np.int64),plane_strain_parameters(case.element_vp,case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values('abc',32000,mesh,2,edges,np.column_stack((1000*case.boundary_vp[edges],1000*case.boundary_vs[edges]))))
    model.add_physics(SGPhysicsModel('damping',30001,mesh,0,np.arange(len(mesh.nodes_coords),dtype=np.int64),damping_alpha(mesh.nodes_coords,ALPHA_MAX_S_INV)[:,None]))
    source=nodal_point_force(mesh.nodes_coords,350.,1.)
    times=np.linspace(0.,1.,spec.steps+1); values=np.zeros((len(times),1,2));values[:,0,1]=source.force_y*ricker(times)
    model.add_physics(SGPhysicsModel('source',20200,mesh,0,np.array([source.node_id],dtype=np.int64),values,time=SGPhysicsTimeData(0.,1.,len(times),3)))
    start=perf_counter();model.prepare();elapsed=perf_counter()-start
    s=solver.spectrum
    kim=2/(2*config.kim_gamma+1)
    bound=2*kim/(hypot(sqrt(s.lambda_max),s.lambda_max_damping/kim)+s.lambda_max_damping/kim)
    assert np.isfinite(bound) and bound>0 and s.step_dt<=bound*(1+1e-12)
    policies=[{'cfl_safety':safety,'raw_dt_s':safety*bound,'substeps_per_ms':ceil(.001/(safety*bound)),
               'aligned_dt_s':.001/ceil(.001/(safety*bound)), 'step_count_ratio_old_over_new':k/ceil(.001/(safety*bound))} for safety in (1.,.9,.8,.625,.5)]
    result:dict[str,object]={'status':'prepared_only','model_id':args.model_id,'no_time_integration':True,
       'external_certificate':certificate,'old_selected_k':k,'old_dt_s':spec.dt_s,
       'native_lambda_max':s.lambda_max,'native_lambda_max_damping':s.lambda_max_damping,
       'kim_beta':config.kim_beta,'kim_gamma':config.kim_gamma,'native_cfl1_raw_dt_s':bound,
       'native_cfl1_actual_dt_s':s.step_dt,'native_cfl1_steps_per_tact':s.steps_per_tact,
       'native_cfl1_steps_count':s.steps_count,'native_cfl1_tact_dt_s':s.tact_dt,
       'old_dt_over_native_raw_bound':spec.dt_s/bound,'policies':policies,'prepare_seconds':elapsed,
       'mesh':file_record(mesh_path),'code':file_record(Path(__file__)),'native':file_record(Path(_native.__file__)),
       'limitation':'Estimates only; no convergence or stability experiment at proposed steps; speed ratios count steps, not measured wall time'}
    write_json(out/'comparison.json',result)
    print((out/'comparison.json').read_text(),flush=True)


if __name__=='__main__':
    main()
