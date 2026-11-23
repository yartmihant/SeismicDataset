"""p6 distance-to-PML experiment: matched steps and translational controls."""
from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Callable, cast
import numpy as np
from semgpu2.model import SGModel, SGPhysicsModel
from semgpu2.model.physics import SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseSolverConfig, SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.run import checked_time, file_record, TimedResult
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray, SeismicSolver, Spectrum
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_5_geometry_frequency.run import space
from scripts.dev_1_6_p67s_showcase.proximity_mesh import ROOT, Case, cases, grid, prepare
from scripts.reporting import read_json,number,mapping
SNAPSHOT=Path('docs/runtime_sources/seismic_p67s_v1/provenance.json')


def run(root:Path,case:Case,dt:float|None,probe:bool=False)->None:
    space(20);out=root/case.name;out.mkdir(exist_ok=False);begin=perf_counter();g=grid(case);coords=g.mesh.nodes_coords
    prefix='semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_p6_cu'
    mod=importlib.import_module(prefix+'.solver');const=importlib.import_module(prefix+'.constants');native=importlib.import_module(prefix+'._native')
    cf=cast(Callable[...,SGBaseSolverConfig],const.SGSolverConfig);sf=cast(Callable[[SGBaseSolverConfig],SeismicSolver],mod.SGSolver)
    solver=sf(cf(t_begin=0.,t_end=1.,step_dt=dt,cfl_safety=.8 if dt is None else None,tact_dt=.001,kim_beta=1/27,kim_gamma=.5))
    model=SGModel(g.mesh,solver)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,g.mesh,1,np.arange(len(g.cells),dtype=np.int64),plane_strain_parameters(g.vp,g.vs)))
    if len(g.pml_ids):model.add_physics(SGPhysicsModel('PML350',31100,g.mesh,1,g.pml_ids,g.pml_values))
    times=np.linspace(0,1,1000001);q=np.pi*30*(times-.05);force=np.zeros((len(times),1,2));force[:,0,1]=-(1-2*q*q)*np.exp(-q*q)
    model.add_physics(SGPhysicsModel('source',20200,g.mesh,0,np.array([g.source_node],dtype=np.int64),force,time=SGPhysicsTimeData(0.,1.,len(times),3)))
    surface=np.empty((1001,len(g.top),2));ns=0;nf=0;diagnostics:list[dict[str,object]]=[];diagnostic_top:list[FloatArray]=[]
    core=(coords[:,0]>=0)&(coords[:,0]<=1400)&(coords[:,1]>=-700)
    def surface_handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal ns
        checked_time(frame,event,ns*.001);assert ns<1001 and np.isfinite(frame).all();surface[ns]=frame.reshape(2,-1).T;ns+=1
    def field_handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal nf
        checked_time(frame,event,nf*.05);assert nf<21 and np.isfinite(frame).all();v=frame.reshape(2,-1).T
        diagnostics.append({'t':nf*.05,'full_peak':float(np.max(np.linalg.norm(v,axis=1))),'core_peak':float(np.max(np.linalg.norm(v[core],axis=1))),'pml_peak':float(np.max(np.linalg.norm(v[~core],axis=1)))})
        diagnostic_top.append(v[g.top].copy());nf+=1
    if not probe:
        model.add_receiver(SGReceiverModel.from_positions('surface',g.mesh,0,g.top,np.array([4,5],dtype=np.int32),begin_tact=0,end_tact=1000,stride=1,handler=surface_handler))
        if case.domain=='production':model.add_receiver(SGReceiverModel.from_positions('diagnostics',g.mesh,0,np.arange(len(coords),dtype=np.int64),np.array([4,5],dtype=np.int32),begin_tact=0,end_tact=1000,stride=50,handler=field_handler))
    prep=perf_counter();model.prepare();prepared=perf_counter();s=cast(Spectrum,solver.spectrum);assert native.__file__ is not None
    record:dict[str,object]={'status':'prepared','case':case.__dict__,'dt_s':s.step_dt,'steps':s.steps_count,'steps_per_tact':s.steps_per_tact,'lambda_max':s.lambda_max,'lambda_max_damping':s.lambda_max_damping,
        'nodes':len(coords),'elements':len(g.cells),'pml_elements':len(g.pml_ids),'boundary_elements':0,'cfl_safety':.8 if dt is None else None,
        'assembly_seconds':prep-begin,'prepare_seconds':prepared-prep,'native':file_record(Path(native.__file__)),'snapshot':file_record(SNAPSHOT),'runner':file_record(Path(__file__)),
        'source_xy_m':coords[g.source_node].tolist(),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    if probe:write(out/'probe.json',record);return
    write(out/'started.json',record);print('STARTED',case.name,s.step_dt,len(coords),flush=True)
    start=perf_counter();result=cast(TimedResult,model.run(sync=True));end=perf_counter()
    assert int(result.code)==int(const.SGRunCodeEnum.TIME_END) and result.steps_count==s.steps_count and ns==1001
    assert nf==(21 if case.domain=='production' else 0)
    if diagnostic_top:np.testing.assert_array_equal(np.stack(diagnostic_top),surface[::50])
    state=np.asarray(solver.solutions[0][:,:]);assert np.isfinite(state).all();np.testing.assert_array_equal(state[2:4,g.top].T,surface[-1])
    if solver.pml_history is not None:assert np.isfinite(np.asarray(solver.pml_history[:,:])).all()
    np.savez(out/'surface.npz',time_s=times[::1000],x_m=coords[g.top,0],weights_m=g.top_weights,velocity_m_s=surface)
    record.update(status='completed',run_seconds=end-start,total_seconds=perf_counter()-begin,terminal_state_finite=True,surface_samples=ns,full_diagnostic_frames=nf,diagnostics=diagnostics,surface=file_record(out/'surface.npz'))
    write(out/'result.json',record);print('COMPLETED',case.name,record['run_seconds'],flush=True)


def series(root:Path)->None:
    assert read_json(root/'plan.json')['status']=='cpu_checked';space(22)
    write(root/'environment.json',{'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),'snapshot':file_record(SNAPSHOT),'python':sys.executable})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def launch(case:Case,dt:float|None=None,probe:bool=False)->dict[str,object]:
        cmd=[sys.executable,'-m','scripts.dev_1_6_p67s_showcase.proximity_run','--root',str(root),'--name',case.name,'--domain',case.domain,'--source',str(case.source)]
        if case.homogeneous:cmd.append('--homogeneous')
        if dt is not None:cmd+=['--dt',str(dt)]
        if probe:cmd.append('--probe')
        with(root/f'{case.name}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return read_json(root/case.name/('probe.json' if probe else 'result.json'))
    probes={domain:launch(Case('probe_'+domain,domain,700.),probe=True) for domain in ('production','reference','larger','left_only')}
    dt=min(number(p['dt_s']) for p in probes.values());write(root/'step_plan.json',{'common_dt_s':dt,'half_dt_s':dt/2,'probes':probes})
    jobs=cases()
    # References first; an interrupted series cannot silently change the normalization.
    jobs.sort(key=lambda c:(c.domain=='production',c.half,c.name))
    for case in jobs:launch(case,dt/2 if case.half else dt)
    write(root/'completed.json',{'status':'completed','full_runs':len(jobs),'probes':4,'common_dt_s':dt,'half_dt_s':dt/2,'production':'unchanged, stopped'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--series',action='store_true');p.add_argument('--name');p.add_argument('--domain',choices=('production','reference','larger','left_only'));p.add_argument('--source',type=float,default=700);p.add_argument('--homogeneous',action='store_true');p.add_argument('--dt',type=float);p.add_argument('--probe',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.series:series(a.root)
    else:
        assert a.name and a.domain;run(a.root,Case(a.name,a.domain,a.source,a.homogeneous,a.name.endswith('_half')),a.dt,a.probe)
