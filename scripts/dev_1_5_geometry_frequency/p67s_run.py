"""Compare optimized p67s and isolate surface-receiver costs on A100."""
from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
import subprocess
from time import perf_counter, thread_time
from typing import Callable, cast

os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'.cache_matplotlib'))
import numpy as np
from semgpu2.model import SGModel, SGPhysicsModel
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseSolverConfig, SGBaseReceiverEventData, SGReceiverRunControl, SGRunHandle
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.run import checked_time, file_record, TimedResult
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray, SeismicSolver, Spectrum, SOURCE_POSITIONS, material, pml
from scripts.dev_1_5_geometry_frequency.p6_7s_compare import leading_grid
from scripts.dev_1_5_geometry_frequency.run import space, nvtx_stage
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.reporting import read_json, number

ROOT=Path('data/dev_1_5_geometry_frequency/p67s_receiver_v1')
SNAPSHOT=Path('docs/runtime_sources/seismic_p67s_v1/provenance.json')
MODES=('old_sync','new_none','new_discard','new_sync','new_async')
REPEATS=5


def prepare(root:Path)->None:
    root.mkdir(parents=True,exist_ok=False);g=leading_grid();assert len(g.mesh.nodes_coords)==795691
    write(root/'plan.json',{'status':'cpu_checked','modes':list(MODES),'repeats':REPEATS,'runs':len(MODES)*(REPEATS+1),'probes':2,'profiles':list(MODES),
        'case':'p6/h10, 1400x700, two layers, 350m quadratic PML/no ABC,30Hz Ricker,7 shots,1s,beta1/27,gamma.5,CFL.8',
        'receivers':'1001 times; all1261 top GLL nodes per shot; Vx,Vy; no field frames',
        'async':'frame_buf_size4, run(sync=False).wait(); framework queue fully drained before timing ends',
        'output':'NPZ after timed run; no SEG-Y requested after clarification',
        'protocol':'one warmup then five rotated rounds on GPU0; profiling separate from timing',
        'none':'no receivers; still1000 tacts and native terminal SAVE; final full velocity checked',
        'discard':'same sync receiver, time/finite checks, no persistent surface copy',
        'snapshot':file_record(SNAPSHOT)})


def run(root:Path,mode:str,label:str,probe:bool=False,profile:bool=False)->None:
    space(20);out=root/mode/label;out.mkdir(parents=True,exist_ok=False)
    start=perf_counter();g=leading_grid();mesh=g.mesh
    solver_name='p6_7s' if mode=='old_sync' else 'p67s'
    prefix=f'semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_{solver_name}_cu'
    mod=importlib.import_module(prefix+'.solver');const=importlib.import_module(prefix+'.constants');native=importlib.import_module(prefix+'._native')
    cf=cast(Callable[...,SGBaseSolverConfig],const.SGSolverConfig);sf=cast(Callable[[SGBaseSolverConfig],SeismicSolver],mod.SGSolver)
    solver=sf(cf(t_begin=0.,t_end=1.,cfl_safety=.8,tact_dt=.001,kim_beta=1/27,kim_gamma=.5))
    model=SGModel(mesh,solver);vp,vs=material(g)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(g.cells),dtype=np.int64),plane_strain_parameters(vp,vs)))
    model.add_physics(pml(g));ns=0;callback_wall=0.;callback_cpu=0.
    keep=mode in ('old_sync','new_sync','new_async')
    surface=np.empty((1001,8827,2),dtype=np.float64) if keep else None
    def handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal ns,callback_wall,callback_cpu
        begin=perf_counter();cpu=thread_time()
        checked_time(frame,event,ns*.001);assert ns<1001 and np.isfinite(frame).all()
        if surface is not None:surface[ns]=frame.reshape(7,2,1261).transpose(0,2,1).reshape(8827,2)
        ns+=1;callback_cpu+=thread_time()-cpu;callback_wall+=perf_counter()-begin
    if mode!='new_none' and not probe:
        model.add_receiver(SGReceiverModel.from_positions('surface',mesh,0,g.top,
            np.array([6*s+c for s in range(7) for c in (4,5)],dtype=np.int32),begin_tact=0,end_tact=1000,stride=1,
            handler=handler,frame_buf_size=4 if mode=='new_async' else 0))
    prep=perf_counter()
    with nvtx_stage('semgpu_prepare',profile):model.prepare()
    prepared=perf_counter();s=cast(Spectrum,solver.spectrum)
    assert native.__file__ is not None
    r:dict[str,object]={'status':'prepared','mode':mode,'label':label,'profiled':profile,'dt_s':s.step_dt,'steps':s.steps_count,
        'steps_per_tact':s.steps_per_tact,'lambda_max':s.lambda_max,'lambda_max_damping':s.lambda_max_damping,
        'assembly_seconds':prep-start,'prepare_seconds':prepared-prep,'nodes_count':len(mesh.nodes_coords),'elements':len(g.cells),
        'native':file_record(Path(native.__file__)),'snapshot':file_record(SNAPSHOT),'runner':file_record(Path(__file__)),
        'receiver_registered':mode!='new_none','frame_buf_size':4 if mode=='new_async' else 0,'source_x_m':list(SOURCE_POSITIONS)}
    if probe:write(out/'probe.json',r);return
    assert s.steps_count==9000 and s.step_dt==1/9000
    write(out/'started.json',r);print('STARTED',mode,label,flush=True)
    begin=perf_counter()
    with nvtx_stage('semgpu_run',profile):
        if mode=='new_async':
            handle=cast(SGRunHandle,model.run(sync=False));result=cast(TimedResult,handle.wait())
        else:result=cast(TimedResult,model.run(sync=True))
    end=perf_counter()
    assert int(result.code)==int(const.SGRunCodeEnum.TIME_END) and result.steps_count==9000
    assert ns==(0 if mode=='new_none' else 1001)
    assert solver.pml_history is not None
    history=np.asarray(solver.pml_history[:,:]);assert np.isfinite(history).all()
    # Native terminal SAVE already copied final state; these inspections are outside run.
    state=np.asarray(solver.solutions[0][:,:]);assert state.shape==(42,795691) and np.isfinite(state).all()
    final=state.reshape(7,6,-1)[:,2:4].transpose(0,2,1).copy()
    final_top=final[:,:1261].reshape(8827,2)
    if surface is not None:np.testing.assert_array_equal(surface[-1],final_top)
    np.save(out/'final_top.npy',final_top)
    if label=='warmup':np.save(out/'final_velocity.npy',final)
    saved=perf_counter()
    with nvtx_stage('output_npz',profile):
        if surface is not None:
            np.savez(out/'surface.npz',time_s=np.arange(1001)*.001,coordinates_m=np.tile(mesh.nodes_coords[g.top],(7,1)),velocity_m_s=surface)
    save_end=perf_counter()
    r.update(status='completed',run_seconds=end-begin,solve_wall_seconds=end-start,callback_wall_seconds=callback_wall,
        callback_thread_cpu_seconds=callback_cpu,surface_samples=ns,output_npz_seconds=save_end-saved,
        validation_after_run_seconds=saved-end,total_seconds=save_end-start,pml_history_max_abs=float(np.max(abs(history))),
        final_velocity_max_abs=float(np.max(abs(final))),terminal_state_finite=True)
    for name in ('final_top.npy','final_velocity.npy','surface.npz'):
        if (out/name).exists():r[name]=file_record(out/name)
    write(out/'result.json',r);print('COMPLETED',mode,label,r['run_seconds'],flush=True)


def series(root:Path)->None:
    assert read_json(root/'plan.json')['status']=='cpu_checked';assert not(root/'environment.json').exists()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(root/'environment.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),
        'nsys':subprocess.check_output(['nsys','--version'],text=True),'snapshot':file_record(SNAPSHOT)})
    def launch(mode:str,label:str,probe:bool=False)->dict[str,object]:
        space(20);runtime='p6_7s' if mode=='old_sync' else 'p67s'
        cmd=[f'.runtime_ch15_{runtime}_v1/venv/bin/python','-m','scripts.dev_1_5_geometry_frequency.p67s_run','--root',str(root),'--mode',mode,'--label',label]
        if probe:cmd+=['--probe']
        with(root/f'{mode}_{label}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return read_json(root/mode/label/('probe.json' if probe else 'result.json'))
    probes={mode:launch(mode,'probe',True) for mode in ('old_sync','new_sync')}
    write(root/'step_plan.json',dict(probes));assert all(number(r['dt_s'])==1/9000 for r in probes.values())
    for mode in MODES:launch(mode,'warmup')
    for repeat in range(REPEATS):
        for mode in MODES[repeat:]+MODES[:repeat]:launch(mode,f'repeat{repeat}')
    write(root/'runs_completed.json',{'status':'completed','runs':30,'probes':2,'production':'unchanged, stopped'})


def profiles(root:Path)->None:
    assert (root/'runs_completed.json').exists()
    out=root/'profiles';out.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for mode in MODES:
        space(20);dest=out/mode;dest.mkdir();runtime='p6_7s' if mode=='old_sync' else 'p67s'
        cmd=['nsys','profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--output',str(dest/'trace'),
            f'.runtime_ch15_{runtime}_v1/venv/bin/python','-m','scripts.dev_1_5_geometry_frequency.p67s_run',
            '--root',str(root),'--mode',mode,'--label','profile','--profile']
        with(dest/'profile.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        with(dest/'export.log').open('x') as log:
            subprocess.run(['nsys','export','--type','sqlite','--output',str(dest/'trace.sqlite'),str(dest/'trace.nsys-rep')],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(out/'completed.json',{'status':'completed','modes':list(MODES),'timing_warning':'profiled wall times excluded from performance medians'})


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--series',action='store_true');p.add_argument('--profiles',action='store_true');p.add_argument('--mode',choices=MODES);p.add_argument('--label');p.add_argument('--probe',action='store_true');p.add_argument('--profile',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.series:series(a.root)
    elif a.profiles:profiles(a.root)
    else:
        assert a.mode and a.label;run(a.root,a.mode,a.label,a.probe,a.profile)

if __name__=='__main__':main()
