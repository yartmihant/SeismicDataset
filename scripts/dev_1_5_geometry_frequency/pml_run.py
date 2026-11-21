"""Explicit isolated PML comparison on Saturn; common observed rectangle."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import cast
import numpy as np
from numpy.typing import NDArray
from semgpu2.model import SGModel,SGPhysicsModel
from semgpu2.model.physics import SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData,SGReceiverRunControl
from semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_cu import _native
from semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_cu.solver import SGSolver
from semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_cu.constants import SGSolverConfig,SGSpectrumData,SGRunCodeEnum
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import build_sem,load_frozen
from scripts.dev_1_3_point_source.run import TimedResult,checked_time,file_record
from scripts.dev_1_5_geometry_frequency.run import space
from scripts.dev_1_5_geometry_frequency.setup import ALPHA,build_damping,write
from scripts.dev_1_5_geometry_frequency.pml_setup import ROOT,THICK_ROOT,THICKNESSES,PADDED,W,H,L,VP,VS,RHO,pml_field,observation,core_mass
from scripts.reporting import read_json,number,mapping

FloatArray=NDArray[np.float64]


def execute(order: int,mode: str,dt: float | None,thickness: int | None=None,half: bool=False) -> None:
    space();padding=L if thickness is None else thickness
    out=ROOT/f'p{order}'/mode if thickness is None else THICK_ROOT/f'p{order}'/(f'L{padding}'+('_half' if half else ''))
    out.mkdir(parents=True,exist_ok=False)
    if thickness is not None:assert thickness in THICKNESSES and mode=='pml_abc' and dt is not None
    start=perf_counter();reference=mode=='reference';has_pml=mode.startswith('pml');old=mode.startswith('old')
    assert order in (4,5) and mode in ('old','pml','pml_abc','reference','old_half','pml_half')
    if mode!='old' and dt is None:raise ValueError('Explicit common dt required outside baseline CFL run')
    path=ROOT/'meshes/reference_q2.npz' if reference else PADDED
    if thickness is not None:path=THICK_ROOT/'meshes'/f'L{padding}.npz'
    frozen=load_frozen(path);assert np.all(frozen.vp==VP) and np.all(frozen.vs==VS)
    case=build_sem(frozen,order);mesh=case.mesh;coords=mesh.nodes_coords
    selected,surface_ids=observation(mesh);view=coords[selected]
    nx=(W+2*padding)//5;ny=(H+padding)//5
    assert len(selected)==(nx*order+1)*(ny*order+1) and len(surface_ids)==nx*order+1
    field_mass=core_mass(frozen,mesh,order,selected)
    solver=SGSolver(SGSolverConfig(t_begin=0.,t_end=1.,step_dt=dt,cfl_safety=.8 if dt is None else None,tact_dt=.001,kim_beta=1/27,kim_gamma=.5))
    model=SGModel(mesh,solver)
    ids=np.arange(len(frozen.cells),dtype=np.int64)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,ids,plane_strain_parameters(case.element_vp,case.element_vs)))
    abc=old or mode=='pml_abc';damping_nodes=0;pml_elements=0
    if abc:
        edges=mesh.strata[2].p_nodes.reshape(-1,order+1)
        outer=np.flatnonzero(~np.all(np.abs(coords[edges,1])<1e-8,axis=1))
        model.add_physics(SGPhysicsModel.from_entity_values('abc',32000,mesh,2,outer,np.tile([RHO*VP,RHO*VS],(len(outer),1))))
    if old:
        field=build_damping(frozen,mesh,W,H);model.add_physics(field);damping_nodes=len(field.entities)
    if has_pml:
        field=pml_field(frozen,mesh,thickness=padding);model.add_physics(field);pml_elements=len(field.entities)
    source=np.flatnonzero(np.all(np.abs(coords-[W/2,0])<1e-9,axis=1));assert len(source)==1
    times=np.linspace(0.,1.,1000001);phase=np.pi*30*(times-.05)
    force=np.zeros((len(times),1,2));force[:,0,1]=-(1-2*phase**2)*np.exp(-phase**2)
    model.add_physics(SGPhysicsModel('point_force',20200,mesh,0,source,force,time=SGPhysicsTimeData(0.,1.,len(times),3)))
    np.save(out/'nodes.npy',view);np.save(out/'core_mass.npy',field_mass)
    surface=np.empty((1001,len(surface_ids),2))
    fields=np.lib.format.open_memmap(out/'velocity.npy',mode='w+',dtype=np.float64,shape=(21,len(selected),2))
    ns=0;nf=0
    def surface_handler(frame: FloatArray,event: SGBaseReceiverEventData,control: SGReceiverRunControl) -> None:
        nonlocal ns
        checked_time(frame,event,ns*.001);assert ns<1001 and np.isfinite(frame).all()
        surface[ns]=frame.reshape(2,-1).T;ns+=1
    def field_handler(frame: FloatArray,event: SGBaseReceiverEventData,control: SGReceiverRunControl) -> None:
        nonlocal nf
        space(20);checked_time(frame,event,nf*.05);assert nf<21 and np.isfinite(frame).all()
        fields[nf]=frame.reshape(2,-1).T;nf+=1
    components=np.array([4,5],dtype=np.int32)
    model.add_receiver(SGReceiverModel.from_positions('surface',mesh,0,surface_ids,components,begin_tact=0,end_tact=1000,stride=1,handler=surface_handler))
    model.add_receiver(SGReceiverModel.from_positions('fields',mesh,0,selected,components,begin_tact=0,end_tact=1000,stride=50,handler=field_handler))
    prep=perf_counter();model.prepare();s=solver.spectrum;assert isinstance(s,SGSpectrumData)
    prep_end=perf_counter();assert len(solver.pml_entity_ids)==pml_elements
    record:dict[str,object]={'status':'prepared','order':order,'mode':mode,'native':file_record(Path(_native.__file__)),'code':file_record(Path(__file__)),'setup_code':file_record(Path(__file__).with_name('pml_setup.py')),'mesh':file_record(path),'global_nodes':len(coords),'elements':len(frozen.cells),'field_nodes':len(view),'surface_nodes':len(surface_ids),'field_bounds':[-175,1575,-875,0],'domain_bounds':[float(coords[:,0].min()),float(coords[:,0].max()),float(coords[:,1].min()),0.],'pml_elements':pml_elements,'damping_nodes':damping_nodes,'ABC':abc,'dmax_or_alpha_s_inv':ALPHA,'dt_s':s.step_dt,'steps':s.steps_count,'steps_per_ms':s.steps_per_tact,'cfl_safety':.8 if dt is None else None,'explicit_dt_s':dt,'kim_beta':1/27,'kim_gamma':.5,'frequency_hz':30,'source_x_m':700,'source_y_m':0,'source_t0_s':.05,'force_peak_n_per_m':1.,'material':[VP,VS,RHO],'assembly_seconds':prep-start,'prepare_seconds':prep_end-prep,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES')}
    record.update(thickness_m=padding,dmax_or_alpha_s_inv=ALPHA*(L/padding),field_bounds=[-padding,W+padding,-H-padding,0],half_step=half)
    write(out/'started.json',record);print('STARTED',order,out.name,s.steps_count,len(coords),flush=True)
    run_start=perf_counter();result=cast(TimedResult,model.run(sync=True));run_end=perf_counter()
    assert int(result.code)==int(SGRunCodeEnum.TIME_END) and result.steps_count==s.steps_count and ns==1001 and nf==21
    fields.flush();surface_positions=np.flatnonzero(np.abs(view[:,1])<1e-8)
    np.testing.assert_array_equal(fields[:,surface_positions],surface[::50])
    np.savez(out/'surface.npz',time_s=np.arange(1001)*.001,coordinates_m=coords[surface_ids],velocity_m_s=surface)
    peaks=np.zeros(2)
    for frame in fields:peaks=np.maximum(peaks,np.max(np.abs(frame),axis=0))
    if has_pml:
        assert solver.pml_history is not None
        history=np.asarray(solver.pml_history[:,:]);assert history.shape==(4,pml_elements*(order+1)**2) and np.isfinite(history).all()
        record['pml_history_max_abs']=float(np.abs(history).max())
    record.update(status='completed',run_seconds=run_end-run_start,total_seconds=perf_counter()-start,frames=nf,surface_samples=ns,surface_field_exact=True,field_peak_m_s=peaks.tolist(),fields=file_record(out/'velocity.npy'),nodes=file_record(out/'nodes.npy'),surface=file_record(out/'surface.npz'),mass=file_record(out/'core_mass.npy'))
    write(out/'result.json',record);print('COMPLETED',order,mode,record['run_seconds'],flush=True)


def thickness_suite(order: int) -> None:
    out=THICK_ROOT/f'p{order}';out.mkdir(parents=True,exist_ok=False)
    baseline=read_json(ROOT/f'p{order}'/'old/result.json')
    assert file_record(Path(_native.__file__))['sha256']==mapping(baseline['native'])['sha256']
    dt=number(baseline['dt_s'])
    write(out/'environment.json',{'hostname':platform.node(),'python':sys.executable,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'native':file_record(Path(_native.__file__)),'baseline_dt_s':dt,'source_baseline':file_record(ROOT/f'p{order}'/'old/result.json')})
    cases=[(thickness,False) for thickness in THICKNESSES]+([(10,True)] if order==5 else [])
    for thickness,half in cases:
        label=f'L{thickness}'+('_half' if half else '')
        args=[sys.executable,'-m','scripts.dev_1_5_geometry_frequency.pml_run','--order',str(order),'--mode','pml_abc','--dt',str(dt/2 if half else dt),'--thickness',str(thickness)]
        if half:args.append('--half')
        with (out/f'{label}.log').open('x') as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
    print('THICKNESS SUITE COMPLETED',order,flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--order',type=int,choices=(4,5),required=True);p.add_argument('--mode',choices=('old','pml','pml_abc','reference','old_half','pml_half'));p.add_argument('--dt',type=float);p.add_argument('--suite',action='store_true');p.add_argument('--thickness',type=int,choices=THICKNESSES);p.add_argument('--thickness-suite',action='store_true');p.add_argument('--half',action='store_true');a=p.parse_args()
    if a.thickness_suite:thickness_suite(a.order);return
    if a.mode:execute(a.order,a.mode,a.dt,a.thickness,a.half);return
    if not a.suite:raise ValueError('Need --suite or --mode')
    out=ROOT/f'p{a.order}';out.mkdir(parents=True,exist_ok=False)
    write(out/'environment.json',{'hostname':platform.node(),'python':sys.executable,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES'),'snapshot':file_record(Path('docs/runtime_sources/seismic_pml_v1/provenance.json'))})
    def launch(mode: str,dt: float | None=None) -> None:
        args=[sys.executable,'-m','scripts.dev_1_5_geometry_frequency.pml_run','--order',str(a.order),'--mode',mode]
        if dt is not None:args+=['--dt',str(dt)]
        with (out/f'{mode}.log').open('x') as log:subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
    launch('old');dt=number(read_json(out/'old/result.json')['dt_s'])
    write(out/'plan.json',{'order':a.order,'native_baseline_dt_s':dt,'common_dt_modes':['old','pml','pml_abc','reference'],'half_dt_modes':['old_half','pml_half'] if a.order==5 else [],'reference_min_geometric_boundary_return_s':5525/VP})
    for mode in ('pml','pml_abc'):launch(mode,dt)
    if a.order==5:
        for mode in ('old_half','pml_half'):launch(mode,dt/2)
    launch('reference',dt)
    print('SUITE COMPLETED',a.order,flush=True)


if __name__=='__main__':main()
