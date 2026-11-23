"""Шесть семиисточниковых опытов p67s на Saturn; поля и сейсмограммы FP64."""
from __future__ import annotations
import argparse
from math import ceil
import importlib
import os
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Callable, cast
import numpy as np
from semgpu2.model import SGModel, SGPhysicsModel
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseSolverConfig, SGBaseReceiverEventData, SGReceiverRunControl, SGRunHandle
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import load_frozen
from scripts.dev_1_3_point_source.run import checked_time, file_record, TimedResult
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray, SeismicSolver, Spectrum, SOURCE_POSITIONS
from scripts.dev_1_5_geometry_frequency.setup import ALPHA, write
from scripts.dev_1_5_geometry_frequency.run import space
from scripts.dev_1_6_p67s_showcase.mesh import ROOT, CASES, leading
from scripts.reporting import read_json,number
SNAPSHOT=Path('docs/runtime_sources/seismic_p67s_v1/provenance.json')


def run(root:Path,name:str)->None:
    space(20);base=root/name;out=base/'run';out.mkdir(exist_ok=False)
    start=perf_counter();frozen=load_frozen(base/'mesh_q2.npz');mesh,cells,top=leading(frozen);n=len(mesh.nodes_coords)
    prefix='semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_p67s_cu'
    mod=importlib.import_module(prefix+'.solver');const=importlib.import_module(prefix+'.constants');native=importlib.import_module(prefix+'._native')
    cf=cast(Callable[...,SGBaseSolverConfig],const.SGSolverConfig);sf=cast(Callable[[SGBaseSolverConfig],SeismicSolver],mod.SGSolver)
    solver=sf(cf(t_begin=0.,t_end=1.,cfl_safety=.8,tact_dt=.001,kim_beta=1/27,kim_gamma=.5))
    model=SGModel(mesh,solver)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(cells),dtype=np.int64),plane_strain_parameters(frozen.vp,frozen.vs)))
    center=frozen.nodes[frozen.cells[:,4]];ids=np.flatnonzero((center[:,0]<0)|(center[:,0]>1400)|(center[:,1]<-700))
    xy=mesh.nodes_coords[cells[ids].ravel()];sx=np.maximum(np.maximum(-xy[:,0],xy[:,0]-1400),0)/350;sy=np.maximum(-xy[:,1]-700,0)/350
    model.add_physics(SGPhysicsModel('PML350',31100,mesh,1,ids,ALPHA*np.column_stack((sx*sx,sy*sy))))
    surface=np.empty((1001,7,1261,2),dtype=np.float64)
    fields=np.lib.format.open_memmap(out/'velocity.npy',mode='w+',dtype=np.float64,shape=(21,7,n,2))
    ns=0;nf=0;peaks=[];core_peaks=[]
    core=(mesh.nodes_coords[:,0]>=0)&(mesh.nodes_coords[:,0]<=1400)&(mesh.nodes_coords[:,1]>=-700)
    def surface_handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal ns
        checked_time(frame,event,ns*.001);assert ns<1001 and np.isfinite(frame).all()
        surface[ns]=frame.reshape(7,2,1261).transpose(0,2,1);ns+=1
    def field_handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal nf
        checked_time(frame,event,nf*.05);assert nf<21 and np.isfinite(frame).all()
        value=frame.reshape(7,2,n).transpose(0,2,1);fields[nf]=value
        peaks.append(np.max(np.abs(value),axis=1).tolist());core_peaks.append(np.max(np.abs(value[:,core]),axis=1).tolist());nf+=1
    components=np.array([6*s+c for s in range(7) for c in (4,5)],dtype=np.int32)
    model.add_receiver(SGReceiverModel.from_positions('surface',mesh,0,top,components,begin_tact=0,end_tact=1000,stride=1,handler=surface_handler,frame_buf_size=4))
    model.add_receiver(SGReceiverModel.from_positions('fields',mesh,0,np.arange(n,dtype=np.int64),components,begin_tact=0,end_tact=1000,stride=50,handler=field_handler,frame_buf_size=2))
    prep=perf_counter();model.prepare();prepared=perf_counter();s=cast(Spectrum,solver.spectrum)
    assert native.__file__ is not None
    record:dict[str,object]={'status':'prepared','case':name,'dt_s':s.step_dt,'steps':s.steps_count,'steps_per_tact':s.steps_per_tact,
        'lambda_max':s.lambda_max,'lambda_max_damping':s.lambda_max_damping,'assembly_seconds':prep-start,'prepare_seconds':prepared-prep,
        'nodes':n,'elements':len(cells),'pml_elements':len(ids),'boundary_elements':0,'ABC':False,'PML_m':350,'dmax_s_inv':ALPHA,
        'source_x_m':list(SOURCE_POSITIONS),'source':'builtin Ricker30Hz t0=.05, Fy peak=-1N/m; independent seven states',
        'native':file_record(Path(native.__file__)),'snapshot':file_record(SNAPSHOT),'runner':file_record(Path(__file__)),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    write(out/'started.json',record);print('STARTED',name,s.step_dt,s.steps_count,flush=True)
    begin=perf_counter();handle=cast(SGRunHandle,model.run(sync=False));result=cast(TimedResult,handle.wait());end=perf_counter()
    assert int(result.code)==int(const.SGRunCodeEnum.TIME_END) and result.steps_count==s.steps_count
    assert ns==1001 and nf==21
    fields.flush()
    for t in range(21):np.testing.assert_array_equal(fields[t,:,:1261],surface[t*50])
    state=np.asarray(solver.solutions[0][:,:]);assert state.shape==(42,n) and np.isfinite(state).all()
    assert solver.pml_history is not None and np.isfinite(np.asarray(solver.pml_history[:,:])).all()
    np.testing.assert_array_equal(state.reshape(7,6,n)[:,2:4].transpose(0,2,1),fields[-1])
    np.save(out/'nodes.npy',mesh.nodes_coords);np.save(out/'cells.npy',cells)
    np.savez(out/'surface.npz',time_s=np.arange(1001)*.001,coordinates_m=mesh.nodes_coords[top],velocity_m_s=surface,source_x_m=SOURCE_POSITIONS)
    record.update(status='completed',run_seconds=end-begin,surface_samples=ns,field_frames=nf,field_time_s=(np.arange(21)*.05).tolist(),
        full_peak_m_s=peaks,core_peak_m_s=core_peaks,terminal_state_finite=True,surface_equals_fields=True,
        output_note='full nodal fields FP64, 21 frames; all1261 top nodes including PML; no SEG-Y')
    for filename in ('velocity.npy','surface.npz','nodes.npy','cells.npy'):record[filename]=file_record(out/filename)
    write(out/'result.json',record);print('COMPLETED',name,end-begin,flush=True)


def series(root:Path)->None:
    plan=read_json(root/'plan.json');space(ceil(number(plan['estimated_raw_bytes'])/2**30)+20)
    write(root/'environment.json',{'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),
        'snapshot':file_record(SNAPSHOT),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for name in CASES:
        with(root/f'{name}.log').open('x') as log:
            subprocess.run(['.runtime_ch15_p67s_v1/venv/bin/python','-m','scripts.dev_1_6_p67s_showcase.run','--root',str(root),'--case',name],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'completed.json',{'status':'completed','geometries':6,'independent_sources':42,'frames_per_source':21,'samples_per_source':1001,'production':'stopped, unchanged'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--case',choices=CASES);args=parser.parse_args()
    if args.case:run(args.root,args.case)
    else:series(args.root)
