"""Four/seven-shot and single p6 solves on the same flat EFVB mesh."""
from __future__ import annotations
import argparse
import importlib
import os
from pathlib import Path
from time import perf_counter
from typing import Callable,cast
import numpy as np
from semgpu2.model import SGModel,SGPhysicsModel
from semgpu2.model.physics import SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseSolverConfig,SGBaseReceiverEventData,SGReceiverRunControl,SGRunHandle
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.run import checked_time,file_record,TimedResult
from scripts.dev_1_4_server_preparation.model_input import read_model
from scripts.dev_1_5_geometry_frequency.p6_7s_compare import leading_grid
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray,SeismicSolver,Spectrum,pml
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_5_geometry_frequency.run import space

SNAPSHOT=Path('docs/runtime_sources/seismic_p64s_v1/provenance.json')
SOURCES=(100.,300.,500.,700.)
RECEIVERS=np.arange(5,1400,10,dtype=np.float64)


def execute(model_id:int,variant:str,out:Path,save_state:bool=False,before_write:Callable[[int],None]|None=None)->dict[str,object]:
    space(20);out.mkdir(parents=True,exist_ok=False);begin=perf_counter();g=leading_grid();mesh=g.mesh;n=len(mesh.nodes_coords)
    source=read_model('EFVB',model_id)
    assert np.all(source.vp==source.vp[:,:1]) and np.all(source.vs==source.vs[:,:1]),'Flat model required'
    row=np.clip((-g.centers[:,1]/10).astype(np.int64),0,69);vp,vs=source.vp[row,0],source.vs[row,0]
    nodes=np.array([int(np.argmin(abs(mesh.nodes_coords[g.top,0]-x))) for x in RECEIVERS],dtype=np.int64)
    np.testing.assert_allclose(mesh.nodes_coords[nodes],np.column_stack((RECEIVERS,np.zeros(140))),atol=1e-9)
    np.testing.assert_allclose(mesh.nodes_coords[[270,390,510,630,750,870,990],0],np.arange(100,1400,200),atol=1e-9)
    prefix='semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_'+variant+'_cu'
    mod=importlib.import_module(prefix+'.solver');const=importlib.import_module(prefix+'.constants');native=importlib.import_module(prefix+'._native')
    cf=cast(Callable[...,SGBaseSolverConfig],const.SGSolverConfig);sf=cast(Callable[[SGBaseSolverConfig],SeismicSolver],mod.SGSolver)
    shots=7 if variant=='p67s' else 4;batch=1 if variant=='p6' else shots
    surface=np.empty((1001,shots,140,2));states:list[FloatArray]=[];histories:list[FloatArray]=[];records:list[dict[str,object]]=[]
    run_seconds=prepare_seconds=0.;setup_done=perf_counter()
    def reserve(size:int)->None:
        if before_write is not None:before_write(size)
    reserve(100000)
    with (out/'input.npz').open('xb') as stream:np.savez(stream,vp=source.vp,vs=source.vs)
    for offset in (range(4) if variant=='p6' else range(1)):
        solver=sf(cf(t_begin=0.,t_end=1.,cfl_safety=.8,tact_dt=.001,kim_beta=1/27,kim_gamma=.5));model=SGModel(mesh,solver)
        model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(g.cells),dtype=np.int64),plane_strain_parameters(vp,vs)))
        model.add_physics(pml(g))
        if variant=='p6':
            times=np.linspace(0,1,1000001);q=np.pi*30*(times-.05);force=np.zeros((len(times),1,2));force[:,0,1]=-(1-2*q*q)*np.exp(-q*q)
            model.add_physics(SGPhysicsModel('source',20200,mesh,0,np.array([270+120*offset],dtype=np.int64),force,time=SGPhysicsTimeData(0.,1.,len(times),3)))
        count=0
        def handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
            nonlocal count
            checked_time(frame,event,count*.001);assert count<1001 and np.isfinite(frame).all()
            value=frame.reshape(batch,2,140).transpose(0,2,1)
            surface[count,offset:offset+1 if variant=='p6' else shots]=value;count+=1
        model.add_receiver(SGReceiverModel.from_positions('surface',mesh,0,nodes,np.array([6*s+c for s in range(batch) for c in (4,5)],dtype=np.int32),begin_tact=0,end_tact=1000,stride=1,frame_buf_size=4,handler=handler))
        prep=perf_counter();model.prepare();prepare_seconds+=perf_counter()-prep;s=cast(Spectrum,solver.spectrum)
        tick=perf_counter();handle=cast(SGRunHandle,model.run(sync=False));result=cast(TimedResult,handle.wait());run_seconds+=perf_counter()-tick
        assert int(result.code)==int(const.SGRunCodeEnum.TIME_END) and result.steps_count==s.steps_count and count==1001
        state=np.asarray(solver.solutions[0][:,:]);assert np.isfinite(state).all() and state.shape==(batch*6,n)
        np.testing.assert_array_equal(state.reshape(batch,6,n)[:,2:4,nodes].transpose(0,2,1),surface[-1,offset:offset+1 if variant=='p6' else shots])
        assert solver.pml_history is not None;history=np.asarray(solver.pml_history[:,:]);assert np.isfinite(history).all()
        if save_state:states.append(state.copy());histories.append(history.copy())
        records.append({'dt_s':s.step_dt,'steps':s.steps_count,'lambda_max':s.lambda_max,'lambda_max_damping':s.lambda_max_damping,'state_max_abs':float(np.max(abs(state))),'pml_max_abs':float(np.max(abs(history)))})
        del state,history,handle,model,solver
    assert len({r['dt_s'] for r in records})==1 and np.isfinite(surface).all()
    reserve(100000+surface.nbytes+30000)
    np.save(out/'velocity.npy',surface)
    # Verify the stored result, not only the receiver's RAM buffer.
    stored=np.load(out/'velocity.npy',mmap_mode='r');np.testing.assert_array_equal(stored,surface)
    if save_state:
        np.save(out/'state.npy',np.concatenate(states));np.save(out/'pml_history.npy',np.concatenate(histories))
    assert native.__file__ is not None
    record:dict[str,object]={'status':'completed','model_id':model_id,'variant':variant,'shots':shots,'source_x_m':np.arange(100,1400,200).tolist() if shots==7 else list(SOURCES),'receiver_x_m':RECEIVERS.tolist(),'components':['Vx','Vy'],'surface_shape':list(surface.shape),'samples':1001,'duration_s':1.,'tact_dt_s':.001,'p':6,'h_m':10,'core_m':[1400,700],'PML_m':350,'ABC':False,'dmax_s_inv':173.0886108476381,'cfl_safety':.8,'beta':1/27,'gamma':.5,'nodes':n,'elements':len(g.cells),'boundary_elements':0,'source':'Ricker30Hz Fy=-1N/m t0=.05s','source_input':source.provenance,'solver_records':records,'native':file_record(Path(native.__file__)),'snapshot':file_record(SNAPSHOT),'runner':file_record(Path(__file__)),'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'setup_seconds':setup_done-begin,'prepare_seconds':prepare_seconds,'run_seconds':run_seconds,'total_seconds':perf_counter()-begin,'surface':file_record(out/'velocity.npy'),'input':file_record(out/'input.npz'),'terminal_state_finite':True,'surface_terminal_exact':True}
    if save_state:
        record['state']=file_record(out/'state.npy');record['pml_history']=file_record(out/'pml_history.npy')
    reserve(100000+surface.nbytes+30000);write(out/'result.json',record)
    print('COMPLETED',model_id,variant,record['run_seconds'],record['total_seconds'],flush=True);return record

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=int,required=True);p.add_argument('--variant',choices=('p6','p64s','p67s'),required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--state',action='store_true');a=p.parse_args();execute(a.model,a.variant,a.out,a.state)
