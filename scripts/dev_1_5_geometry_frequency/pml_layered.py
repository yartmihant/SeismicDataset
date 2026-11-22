"""P5 two-layer PML thickness study with cached recurrence coefficients."""
from __future__ import annotations
import argparse
import os
import re
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
from dataclasses import dataclass
from scripts.dev_1_3_point_source.mesh import FrozenMesh,build_sem,load_frozen,save_frozen,structured_cells
from scripts.dev_1_3_point_source.run import TimedResult,checked_time,file_record
from scripts.dev_1_5_geometry_frequency.run import space
from scripts.dev_1_5_geometry_frequency.setup import ALPHA,build_damping,write
from scripts.dev_1_5_geometry_frequency.pml_setup import W,H,L,VP,VS,RHO,pml_field,observation,core_mass
from scripts.reporting import read_json,number,mapping

FloatArray=NDArray[np.float64]


ROOT=Path(__file__).resolve().parents[2]/'data/dev_1_5_geometry_frequency/pml_layered_cached_saturn_v1'
THICKNESSES=(10,25,50,75,100,125,150,175,250,350)

@dataclass(frozen=True)
class Case:
    label: str
    thickness: int
    mode: str='pml_abc'
    homogeneous: bool=False
    fixed: bool=False

    @property
    def mesh_name(self) -> str:
        return ('homogeneous_' if self.homogeneous else 'layered_')+str(self.thickness)

CASES={c.label:c for c in [*(Case(f'L{d}',d) for d in THICKNESSES),Case('old',175,'old'),Case('reference',2500,'reference'),Case('homogeneous_reference',2500,'reference',True),Case('L700',700),*(Case(f'F{d}',d,fixed=True) for d in (250,350,500,700))]}


def resolve_case(label: str) -> Case:
    if label=='abc_only':return Case(label,0,'abc_only')
    if label in CASES:return CASES[label]
    if re.fullmatch(r'[LF][0-9]+',label) is None:raise ValueError(label)
    thickness=int(label[1:])
    if thickness%5 or not 5<=thickness<=2500:raise ValueError(label)
    return Case(label,thickness,fixed=label.startswith('F'))


def study_cases() -> dict[str,Case]:
    cases=dict(CASES)
    for path in ROOT.glob('case_plan_*.json'):
        label=path.stem.removeprefix('case_plan_');cases[label]=resolve_case(label)
    return cases


def frozen_mesh(spec: Case) -> FrozenMesh:
    d=spec.thickness
    x=np.arange(-d,W+d+2.5,2.5);y=np.arange(-H-d,2.5,2.5)
    xx,yy=np.meshgrid(x,y);nodes=np.column_stack((xx.ravel(),yy.ravel()))
    cells=structured_cells((len(x)-1)//2,(len(y)-1)//2)
    lower=(nodes[cells[:,4],1]<-350)&(not spec.homogeneous)
    return FrozenMesh(nodes,cells,np.where(lower,3963.,VP),np.where(lower,2357.84473,VS),lower.astype(np.int64),0,'two_layer_pml_cached')


def select(mesh: object,reference: bool) -> tuple[NDArray[np.int64],NDArray[np.int64]]:
    from semgpu2.model.mesh import SGMeshModel
    assert isinstance(mesh,SGMeshModel)
    x,y=mesh.nodes_coords.T
    selected=np.flatnonzero((x>=-1e-8)&(x<=W+1e-8)&(y>=-H-1e-8)) if reference else np.arange(len(x),dtype=np.int64)
    xy=np.round(mesh.nodes_coords[selected],8);selected=selected[np.lexsort((xy[:,0],xy[:,1]))]
    return selected,selected[np.abs(mesh.nodes_coords[selected,1])<1e-8]


def prepare(out: Path,save: bool) -> None:
    out.mkdir(parents=True,exist_ok=False);records=[]
    for spec in CASES.values():
        meshpath=out/'meshes'/f'{spec.mesh_name}.npz'
        if meshpath.exists():continue
        frozen=frozen_mesh(spec)
        if save:
            meshpath.parent.mkdir(exist_ok=True);save_frozen(frozen,meshpath)
        if spec.mode=='reference':
            records.append({'label':spec.label,'elements':len(frozen.cells),'boundary_return_lower_bound_s':5700/VP});continue
        case=build_sem(frozen,5);mesh=case.mesh
        selected,surface=select(mesh,False)
        mass=core_mass(frozen,mesh,5,selected)
        assert np.count_nonzero(mass)==1401*701
        assert np.all(case.element_vp[np.mean(mesh.nodes_coords[mesh.strata[1].p_nodes.reshape(-1,36)],axis=1)[:,1]>-350]==VP)
        edges=mesh.strata[2].p_nodes.reshape(-1,6);mid=mesh.nodes_coords[edges].mean(axis=1)
        np.testing.assert_array_equal(case.boundary_vp,np.where(mid[:,1]<-350,3963.,VP))
        pml=pml_field(frozen,mesh,scale=spec.thickness/175 if spec.fixed else 1.,thickness=spec.thickness).build_data().values
        assert np.all(pml>=0) and np.isclose(pml.max(),ALPHA if spec.fixed else ALPHA*175/spec.thickness)
        assert len(surface)==W+2*spec.thickness+1
        records.append({'label':spec.label,'elements':len(frozen.cells),'nodes':len(selected),'pml_rows':spec.thickness//5,'dmax_s_inv':ALPHA if spec.fixed else ALPHA*175/spec.thickness})
    write(out/'plan.json',{'status':'cpu_checked','cases':records,'order':5,'h_m':5,'duration_s':1,'source_hz':30,'reference_interface_y_m':-350,'physical_reflection_definition':'extended two-layer minus extended homogeneous surface; same mesh near source/receivers, dt and source','criterion':'boundary error <= 0.01 of physical reflected signal in weighted L2 and maximum amplitude; report echoes individually; reject field growth','profile':'quadratic dmax=173.0886108476381*175/L; ABC; no mass damping','dt':'explicit 1/53000 from earlier homogeneous native CFL0.8, checked against new two-layer old baseline','raw_budget_bytes':10000000000})


def execute(spec: Case,dt: float | None,*,order: int=5,root: Path=ROOT,output: Path | None=None,probe: bool=False,omit_pml: bool=False) -> dict[str,object]:
    space();mode=spec.mode;padding=spec.thickness
    out=output if output is not None else root/spec.label;out.mkdir(parents=True,exist_ok=False)
    start=perf_counter();reference=mode=='reference';has_pml=mode.startswith('pml');old=mode.startswith('old')
    assert order in (4,5)
    path=root/'meshes'/f'{spec.mesh_name}.npz'
    frozen=load_frozen(path)
    case=build_sem(frozen,order);mesh=case.mesh;coords=mesh.nodes_coords
    selected,surface_ids=select(mesh,reference);view=coords[selected]
    nx=W//5 if reference else (W+2*padding)//5;ny=H//5 if reference else (H+padding)//5
    assert len(selected)==(nx*order+1)*(ny*order+1) and len(surface_ids)==nx*order+1
    field_mass=core_mass(frozen,mesh,order,selected)
    solver=SGSolver(SGSolverConfig(t_begin=0.,t_end=1.,step_dt=dt,cfl_safety=.8 if dt is None else None,tact_dt=.001,kim_beta=1/27,kim_gamma=.5))
    model=SGModel(mesh,solver)
    ids=np.arange(len(frozen.cells),dtype=np.int64)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,ids,plane_strain_parameters(case.element_vp,case.element_vs)))
    abc=old or mode in ('pml_abc','abc_only');damping_nodes=0;pml_elements=0
    if abc:
        edges=mesh.strata[2].p_nodes.reshape(-1,order+1)
        outer=np.flatnonzero(~np.all(np.abs(coords[edges,1])<1e-8,axis=1))
        model.add_physics(SGPhysicsModel.from_entity_values('abc',32000,mesh,2,outer,np.column_stack((RHO*case.boundary_vp[outer],RHO*case.boundary_vs[outer]))))
    if old:
        field=build_damping(frozen,mesh,W,H);model.add_physics(field);damping_nodes=len(field.entities)
    if has_pml and not omit_pml:
        field=pml_field(frozen,mesh,scale=padding/L if spec.fixed else 1.,thickness=padding);model.add_physics(field);pml_elements=len(field.entities)
    source=np.flatnonzero(np.all(np.abs(coords-[W/2,0])<1e-9,axis=1));assert len(source)==1
    times=np.linspace(0.,1.,1000001);phase=np.pi*30*(times-.05)
    force=np.zeros((len(times),1,2));force[:,0,1]=-(1-2*phase**2)*np.exp(-phase**2)
    model.add_physics(SGPhysicsModel('point_force',20200,mesh,0,source,force,time=SGPhysicsTimeData(0.,1.,len(times),3)))
    if probe:
        prep_start=perf_counter()
        model.prepare();spectrum=solver.spectrum;assert isinstance(spectrum,SGSpectrumData)
        record_probe:dict[str,object]={'status':'prepared','order':order,'mode':mode,'thickness_m':padding,'pml_omitted_for_cfl':omit_pml,'pml_elements':pml_elements,'ABC':abc,'dt_s':spectrum.step_dt,'steps':spectrum.steps_count,'steps_per_ms':spectrum.steps_per_tact,'lambda_max':spectrum.lambda_max,'lambda_max_damping':spectrum.lambda_max_damping,'cfl_safety':.8,'native':file_record(Path(_native.__file__)),'mesh':file_record(path),'prepare_seconds':perf_counter()-prep_start}
        write(out/'probe.json',record_probe);return record_probe
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
    record:dict[str,object]={'status':'prepared','order':order,'mode':mode,'native':file_record(Path(_native.__file__)),'code':file_record(Path(__file__)),'setup_code':file_record(Path(__file__)),'mesh':file_record(path),'global_nodes':len(coords),'elements':len(frozen.cells),'field_nodes':len(view),'surface_nodes':len(surface_ids),'field_bounds':[-175,1575,-875,0],'domain_bounds':[float(coords[:,0].min()),float(coords[:,0].max()),float(coords[:,1].min()),0.],'pml_elements':pml_elements,'damping_nodes':damping_nodes,'ABC':abc,'dmax_or_alpha_s_inv':ALPHA,'dt_s':s.step_dt,'steps':s.steps_count,'steps_per_ms':s.steps_per_tact,'cfl_safety':.8 if dt is None else None,'explicit_dt_s':dt,'kim_beta':1/27,'kim_gamma':.5,'frequency_hz':30,'source_x_m':700,'source_y_m':0,'source_t0_s':.05,'force_peak_n_per_m':1.,'material':[[VP,VS,RHO],[3963.,2357.84473,RHO]] if not spec.homogeneous else [[VP,VS,RHO]],'interface_y_m':-350.,'homogeneous':spec.homogeneous,'assembly_seconds':prep-start,'prepare_seconds':prep_end-prep,'gpu_visible':os.environ.get('CUDA_VISIBLE_DEVICES')}
    record.update(thickness_m=padding,fixed_strength=spec.fixed,dmax_or_alpha_s_inv=(ALPHA if spec.fixed else ALPHA*(L/padding)) if has_pml else ALPHA if old else 0.,field_bounds=[float(view[:,0].min()),float(view[:,0].max()),float(view[:,1].min()),0.],snapshot=file_record(Path('docs/runtime_sources/seismic_pml_cached_v1/provenance.json')))
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
    return record


def prepare_case(spec: Case) -> None:
    target=ROOT/(f'control_plan_{spec.label}.json' if spec.mode=='abc_only' else f'case_plan_{spec.label}.json')
    if target.exists():raise FileExistsError(target)
    path=ROOT/'meshes'/f'{spec.mesh_name}.npz'
    frozen=load_frozen(path) if path.exists() else frozen_mesh(spec)
    if not path.exists():save_frozen(frozen,path)
    case=build_sem(frozen,5);mesh=case.mesh;selected,surface=select(mesh,False)
    assert len(surface)==W+2*spec.thickness+1
    assert np.count_nonzero(core_mass(frozen,mesh,5,selected))==1401*701
    if spec.mode=='abc_only':
        assert spec.thickness==0 and len(selected)==1401*701
        dmax=0.
    else:
        field=pml_field(frozen,mesh,scale=spec.thickness/175 if spec.fixed else 1.,thickness=spec.thickness).build_data().values
        dmax=ALPHA if spec.fixed else ALPHA*175/spec.thickness
        assert np.all(field>=0) and np.isclose(field.max(),dmax)
    write(target,{'status':'cpu_checked','label':spec.label,'nodes':len(selected),'thickness_m':spec.thickness,'dmax_s_inv':dmax,'fixed_strength':spec.fixed,'mesh':file_record(path),'reason':'Refinement of thickness bracket for weak interface echoes; prior results preserved.'})


def prepare_extra() -> None:
    records=[]
    for label in ('L700','F250','F350','F500','F700'):
        spec=CASES[label];path=ROOT/'meshes'/f'{spec.mesh_name}.npz'
        frozen=load_frozen(path) if path.exists() else frozen_mesh(spec)
        if not path.exists():save_frozen(frozen,path)
        case=build_sem(frozen,5);mesh=case.mesh;selected,surface=select(mesh,False)
        assert len(surface)==W+2*spec.thickness+1
        mass=core_mass(frozen,mesh,5,selected);assert np.count_nonzero(mass)==1401*701
        scale=spec.thickness/175 if spec.fixed else 1.
        field=pml_field(frozen,mesh,scale=scale,thickness=spec.thickness).build_data().values
        target=ALPHA if spec.fixed else ALPHA*175/spec.thickness
        assert np.all(field>=0) and np.isclose(field.max(),target)
        records.append({'label':label,'nodes':len(selected),'thickness_m':spec.thickness,'dmax_s_inv':target,'fixed_strength':spec.fixed,'mesh':file_record(path)})
    write(ROOT/'extra_plan.json',{'status':'cpu_checked','cases':records,'reason':'Constant-integral 175/250/350m leaves boundary errors larger than weak multiples. Add constant maximum profiles and 700m constant-integral delay control.','original_plan_preserved':True})


def cfl_check() -> None:
    frozen=load_frozen(ROOT/'meshes/layered_175.npz');case=build_sem(frozen,5);mesh=case.mesh
    solver=SGSolver(SGSolverConfig(t_begin=0.,t_end=1.,cfl_safety=.8,tact_dt=.001,kim_beta=1/27,kim_gamma=.5))
    model=SGModel(mesh,solver);ids=np.arange(len(frozen.cells),dtype=np.int64)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,ids,plane_strain_parameters(case.element_vp,case.element_vs)))
    edges=mesh.strata[2].p_nodes.reshape(-1,6);outer=np.flatnonzero(~np.all(np.abs(mesh.nodes_coords[edges,1])<1e-8,axis=1))
    model.add_physics(SGPhysicsModel.from_entity_values('abc',32000,mesh,2,outer,np.column_stack((RHO*case.boundary_vp[outer],RHO*case.boundary_vs[outer]))))
    model.add_physics(build_damping(frozen,mesh,W,H));model.prepare();spectrum=solver.spectrum
    assert isinstance(spectrum,SGSpectrumData) and 1/53000<=spectrum.step_dt
    write(ROOT/'cfl_check.json',{'status':'passed','native':file_record(Path(_native.__file__)),'native_mass_abc_dt_s':spectrum.step_dt,'native_steps_per_ms':spectrum.steps_per_tact,'common_explicit_dt_s':1/53000,'note':'Baseline elastic/mass/ABC CFL0.8; not a PML stability proof.'})
    print('CFL CHECK PASSED',spectrum.step_dt,flush=True)


TIMING_ROOT=ROOT.parent/'pml_timing_native_cfl_v1'
TIMING_CASES=(resolve_case('abc_only'),resolve_case('L100'),resolve_case('L175'),resolve_case('F250'))


def timing_prepare(root: Path) -> None:
    root.mkdir(parents=True,exist_ok=False);(root/'meshes').mkdir();rows=[]
    for spec in TIMING_CASES:
        frozen=frozen_mesh(spec);path=root/'meshes'/f'{spec.mesh_name}.npz';save_frozen(frozen,path)
        for order in (4,5):
            case=build_sem(frozen,order);mesh=case.mesh;selected,surface=select(mesh,False)
            assert len(selected)==(((W+2*spec.thickness)//5)*order+1)*(((H+spec.thickness)//5)*order+1)
            mass=core_mass(frozen,mesh,order,selected);assert np.count_nonzero(mass)==(280*order+1)*(140*order+1)
            assert len(surface)==(W+2*spec.thickness)//5*order+1
            maximum=0.
            if spec.thickness:
                data=pml_field(frozen,mesh,scale=spec.thickness/L if spec.fixed else 1.,thickness=spec.thickness).build_data().values
                maximum=ALPHA if spec.fixed else ALPHA*L/spec.thickness
                assert np.all(data>=0) and np.isclose(data.max(),maximum)
            rows.append({'order':order,'label':spec.label,'nodes':len(selected),'dmax_s_inv':maximum,'thickness_m':spec.thickness,'mesh':file_record(path)})
    write(root/'plan.json',{'status':'cpu_checked','cases':rows,'repeats':3,'warmups_per_case':1,'half_step_controls_per_case':1,'runs_total':40,'CFL':.8,'requested_step':'native CFL; active PML unsupported, surrogate requires user authorization','profiles':'L100 integral-preserving; L175 base; F250 fixed maximum as in last comparison','physical_domain':[0,1400,-700,0],'order':[4,5],'element_m':5,'frequency_hz':30,'source':[700,0],'duration_s':1,'field_frames':21,'surface_samples':1001,'estimated_array_budget_bytes':sum(int(number(r['nodes']))*21*2*8*5 for r in rows),'native_snapshot':file_record(Path('docs/runtime_sources/seismic_pml_cached_v1/provenance.json'))})
    print('TIMING CPU CHECK PASSED',flush=True)


def timing_series(root: Path,allow_surrogate: bool) -> None:
    assert allow_surrogate,'Explicit authorization is required for PML-free CFL estimate'
    assert (root/'plan.json').exists() and not (root/'environment.json').exists()
    space();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if gpu:raise RuntimeError('GPU is occupied: '+gpu)
    write(root/'environment.json',{'hostname':platform.node(),'python':sys.executable,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total','--format=csv'],text=True),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'native':file_record(Path(_native.__file__)),'native_snapshot':file_record(Path('docs/runtime_sources/seismic_pml_cached_v1/provenance.json')),'surrogate_authorized':allow_surrogate,'timing':'single GPU0, sequential separate processes; full warmup per case, three rotated repeats; synchronous model.run includes 21 fields and 1001 surface samples; half-dt controls excluded from timings'})
    def launch(order: int,spec: Case,label: str,*,probe: bool=False,omit: bool=False,dt: float | None=None,expect_failure: bool=False) -> None:
        folder=root/f'p{order}'/spec.label/label;folder.parent.mkdir(parents=True,exist_ok=True)
        completed=folder/('probe.json' if probe else 'result.json')
        if completed.exists():
            previous=read_json(completed)
            assert previous['order']==order and mapping(previous['native'])['sha256']==file_record(Path(_native.__file__))['sha256']
            assert previous['status']==('prepared' if probe else 'completed')
            if dt is not None:assert np.isclose(number(previous['dt_s']),dt,rtol=1e-13,atol=0)
            print('REUSE COMPLETED',order,spec.label,label,flush=True);return
        cmd=[sys.executable,'-m','scripts.dev_1_5_geometry_frequency.pml_layered','--timing-root',str(root),'--timing-case',spec.label,'--order',str(order),'--timing-label',label]
        if probe:cmd+=['--probe']
        if omit:cmd+=['--omit-pml']
        if dt is not None:cmd+=['--dt',str(dt)]
        print('TIMING LAUNCH',order,spec.label,label,flush=True)
        with folder.with_suffix('.log').open('x') as log:result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        if expect_failure:
            message=folder.with_suffix('.log').read_text()
            assert result.returncode!=0 and 'PML requires explicit step_dt or steps_count' in message
            write(folder/'expected_rejection.json',{'status':'expected_native_rejection','returncode':result.returncode,'reason':'PML requires explicit step_dt or steps_count'})
        elif result.returncode:raise RuntimeError(f'Failed timing case: {folder}')
    for order in (4,5):
        steps:dict[str,float]={}
        launch(order,TIMING_CASES[1],'native_pml_rejection',probe=True,expect_failure=True)
        for spec in TIMING_CASES:
            launch(order,spec,'cfl_probe',probe=True,omit=bool(spec.thickness))
            steps[spec.label]=number(read_json(root/f'p{order}'/spec.label/'cfl_probe/probe.json')['dt_s'])
            launch(order,spec,'warmup',dt=steps[spec.label] if spec.thickness else None)
        for repeat in range(3):
            rotated=TIMING_CASES[repeat:]+TIMING_CASES[:repeat]
            for spec in rotated:launch(order,spec,f'repeat{repeat}',dt=steps[spec.label] if spec.thickness else None)
        for spec in TIMING_CASES:launch(order,spec,'half_dt',dt=steps[spec.label]/2)
    write(root/'runs_completed.json',{'status':'completed','runs':40,'orders':[4,5],'cases':[s.label for s in TIMING_CASES]})
    print('TIMING SERIES COMPLETED',flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--check',type=Path);p.add_argument('--prepare',action='store_true');p.add_argument('--case');p.add_argument('--prepare-case');p.add_argument('--native-cfl',action='store_true');p.add_argument('--cfl-check',action='store_true');p.add_argument('--extra',action='store_true');p.add_argument('--timing-prepare',action='store_true');p.add_argument('--timing-series',action='store_true');p.add_argument('--allow-cfl-surrogate',action='store_true');p.add_argument('--timing-root',type=Path,default=TIMING_ROOT);p.add_argument('--timing-case');p.add_argument('--timing-label');p.add_argument('--order',type=int,choices=(4,5),default=5);p.add_argument('--dt',type=float);p.add_argument('--probe',action='store_true');p.add_argument('--omit-pml',action='store_true');a=p.parse_args()
    if a.timing_prepare:timing_prepare(a.timing_root);return
    if a.timing_series:timing_series(a.timing_root,a.allow_cfl_surrogate);return
    if a.timing_case:
        assert a.timing_label
        execute(resolve_case(a.timing_case),a.dt,order=a.order,root=a.timing_root,output=a.timing_root/f'p{a.order}'/a.timing_case/a.timing_label,probe=a.probe,omit_pml=a.omit_pml);return
    if a.check:prepare(a.check,False)
    if a.prepare:prepare(ROOT,True)
    if a.extra:prepare_extra()
    if a.cfl_check:cfl_check()
    if a.prepare_case:prepare_case(resolve_case(a.prepare_case))
    if a.case:execute(resolve_case(a.case),None if a.native_cfl else 1/53000)

if __name__=='__main__':main()
