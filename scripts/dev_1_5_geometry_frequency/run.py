"""Explicit isolated Saturn runs, full velocity fields and all top nodes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
from typing import cast
import numpy as np
from numpy.typing import NDArray
from semgpu2.model.model import SGModel
from semgpu2.model.physics import SGPhysicsModel,SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData,SGReceiverRunControl
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu import _native
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig,SGSpectrumData,SGRunCodeEnum
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.solver import SGSolver
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.mesh import build_sem,load_frozen
from scripts.dev_1_3_point_source.run import TimedResult,checked_time,file_record
from scripts.dev_1_5_geometry_frequency.setup import ROOT,ALPHA,Spec,specs,damping,write
from scripts.reporting import read_json

FloatArray=NDArray[np.float64]


def space(minimum_gib: int=40) -> None:
    if shutil.disk_usage(ROOT).free < minimum_gib*2**30:
        raise OSError(f'Required free disk reserve {minimum_gib} GiB')


def execute(spec: Spec) -> None:
    space()
    out=ROOT/'runs'/spec.run_id;out.mkdir(parents=True,exist_ok=False)
    start=perf_counter()
    mesh_path=ROOT/'meshes'/spec.mesh_id/'mesh_q2.npz'
    geometry=read_json(mesh_path.with_name('geometry.json'))
    case=build_sem(load_frozen(mesh_path),5);mesh=case.mesh;coords=mesh.nodes_coords
    assert len(coords)==geometry['sem_nodes']
    config=SGSolverConfig(t_begin=0.,t_end=1.,cfl_safety=.8,tact_dt=.001,kim_beta=1/27,kim_gamma=.5)
    solver=SGSolver(config);model=SGModel(mesh,solver)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(case.element_vp),dtype=np.int64),plane_strain_parameters(case.element_vp,case.element_vs)))
    edge_nodes=mesh.strata[2].p_nodes.reshape(-1,6);boundary_coords=coords[edge_nodes]
    top=np.all(np.abs(boundary_coords[:,:,1])<1e-8,axis=1)
    sides=np.all(np.abs(boundary_coords[:,:,0]+175)<1e-8,axis=1)|np.all(np.abs(boundary_coords[:,:,0]-spec.width-175)<1e-8,axis=1)
    bottom=np.all(np.abs(boundary_coords[:,:,1]+spec.height+175)<1e-8,axis=1)
    assert np.all(top|sides|bottom) and not np.any(top&(sides|bottom))
    edges=np.flatnonzero(sides|bottom)
    model.add_physics(SGPhysicsModel.from_entity_values('abc',32000,mesh,2,edges,np.column_stack((1000*case.boundary_vp[edges],1000*case.boundary_vs[edges]))))
    ids=np.arange(len(coords),dtype=np.int64)
    alpha=damping(coords,spec.width,spec.height)
    assert np.all(alpha>=0) and np.isclose(alpha.max(),2*ALPHA)
    model.add_physics(SGPhysicsModel('damping',30001,mesh,0,ids,alpha[:,None]))
    source=np.flatnonzero(np.all(np.abs(coords-[spec.width/2,0])<1e-9,axis=1));assert len(source)==1
    times=np.linspace(0.,1.,1000001);phase=np.pi*spec.frequency*(times-1.5/spec.frequency)
    force=np.zeros((len(times),1,2));force[:,0,1]=-(1-2*phase**2)*np.exp(-phase**2)
    assert np.isclose(force[:,0,1].min(),-1.)
    model.add_physics(SGPhysicsModel('point_source',20200,mesh,0,source,force,time=SGPhysicsTimeData(0.,1.,len(times),3)))
    receivers=np.flatnonzero(np.abs(coords[:,1])<1e-9)
    receivers=receivers[np.argsort(coords[receivers,0])]
    assert len(receivers)==geometry['surface_nodes'] and np.all(np.diff(coords[receivers,0])>0)
    assert np.allclose(coords[receivers[[0,-1]],0],[-175,spec.width+175])
    np.save(out/'nodes.npy',coords)
    surface=np.empty((1001,len(receivers),2));field_times=np.linspace(0.,1.,21)
    fields=np.lib.format.open_memmap(out/'velocity.npy',mode='w+',dtype=np.float64,shape=(21,len(coords),2))
    surface_count=0;field_count=0
    def surface_handler(frame: FloatArray,event: SGBaseReceiverEventData,control: SGReceiverRunControl) -> None:
        nonlocal surface_count
        assert surface_count<1001 and frame.size==2*len(receivers)
        checked_time(frame,event,surface_count*.001)
        surface[surface_count]=frame.reshape(2,-1).T;surface_count+=1
    def field_handler(frame: FloatArray,event: SGBaseReceiverEventData,control: SGReceiverRunControl) -> None:
        nonlocal field_count
        space(20)
        assert field_count<21 and frame.size==2*len(coords)
        checked_time(frame,event,float(field_times[field_count]))
        assert np.isfinite(frame).all()
        fields[field_count]=frame.reshape(2,-1).T;field_count+=1
    components=np.array([4,5],dtype=np.int32)
    model.add_receiver(SGReceiverModel.from_positions('surface',mesh,0,receivers,components,begin_tact=0,end_tact=1000,stride=1,handler=surface_handler))
    model.add_receiver(SGReceiverModel.from_positions('fields',mesh,0,ids,components,begin_tact=0,end_tact=1000,stride=50,handler=field_handler))
    model.prepare();s=solver.spectrum;assert isinstance(s,SGSpectrumData)
    meta:dict[str,object]={'spec':asdict(spec),'status':'prepared','geometry':geometry,'native':file_record(Path(_native.__file__)),
        'mesh':file_record(mesh_path),'code':file_record(Path(__file__)),'dt_s':s.step_dt,'steps':s.steps_count,'steps_per_ms':s.steps_per_tact,
        'source_node':int(source[0]),'source_x_m':spec.width/2,'source_t0_s':1.5/spec.frequency,'force_peak_n_per_m':1.,
        'alpha_max_s_inv':ALPHA,'kim_beta':1/27,'kim_gamma':.5,'cfl_safety':.8,'tact_dt_s':.001,
        'lambda_max':s.lambda_max,'lambda_max_damping':s.lambda_max_damping,'source_table_dt_s':1e-6,
        'surface_nodes':len(receivers),'field_nodes':len(coords),'frame_interval_s':.05,'surface_interval_s':.001}
    write(out/'started.json',meta);print('started',spec.run_id,'steps',s.steps_count,flush=True)
    run_start=perf_counter();result=cast(TimedResult,model.run(sync=True));run_end=perf_counter()
    assert int(result.code)==int(SGRunCodeEnum.TIME_END) and result.steps_count==s.steps_count
    assert surface_count==1001 and field_count==21 and np.isfinite(surface).all()
    fields.flush()
    assert np.array_equal(surface[::50],fields[:,receivers,:])
    assert np.all(fields[0]==0) and np.all(surface[0]==0)
    np.savez(out/'surface.npz',time_s=np.linspace(0.,1.,1001),coordinates_m=coords[receivers],velocity_m_s=surface)
    peaks=np.zeros(2)
    for frame in fields:peaks=np.maximum(peaks,np.max(np.abs(frame),axis=0))
    meta.update(status='completed',run_seconds=run_end-run_start,total_seconds=perf_counter()-start,
        fields=file_record(out/'velocity.npy'),nodes=file_record(out/'nodes.npy'),surface=file_record(out/'surface.npz'),
        field_peak_m_s=peaks.tolist(),surface_peak_m_s=np.max(np.abs(surface),axis=(0,1)).tolist(),
        surface_field_exact=True,frames=21,surface_samples=1001)
    write(out/'result.json',meta);print('completed',spec.run_id,meta['total_seconds'],flush=True)


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--index',type=int);parser.add_argument('--worker',type=int,choices=(0,1));args=parser.parse_args()
    if args.index is not None:
        execute(specs()[args.index]);return
    if args.worker is None:raise ValueError('Need --index or --worker')
    cache=ROOT/'cache';cache.mkdir(exist_ok=True)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.worker),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(cache),XDG_CACHE_HOME=str(cache),CUDA_CACHE_PATH=str(cache))
    for index,spec in enumerate(specs()):
        if index%2!=args.worker:continue
        directory=ROOT/'runs'/spec.run_id
        if (directory/'result.json').exists():
            assert read_json(directory/'result.json')['status']=='completed';continue
        if directory.exists():raise RuntimeError(f'Incomplete existing run: {directory}')
        subprocess.run([sys.executable,'-m','scripts.dev_1_5_geometry_frequency.run','--index',str(index)],check=True)
    print('worker finished',args.worker,flush=True)


if __name__=='__main__':main()
