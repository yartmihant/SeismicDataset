"""A100 comparison of frozen p6, current p6 and shared-mesh p6_7s."""
from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Callable, cast

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd()/".cache_matplotlib"))

import numpy as np
from semgpu2.model import SGModel, SGPhysicsModel
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseSolverConfig, SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.run import checked_time, file_record, TimedResult
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import (
    Case, FloatArray, Grid, SeismicSolver, Spectrum, SOURCE_POSITIONS, grid, material, pml, execute,
)
from scripts.dev_1_5_geometry_frequency.pml_fixed_report import equivalence
from scripts.dev_1_5_geometry_frequency.run import space
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.reporting import read_json, mapping, number

ROOT = Path('data/dev_1_5_geometry_frequency/p6_7s_comparison_v2')
SNAPSHOT = Path('docs/runtime_sources/seismic_p6_7s_v1/provenance.json')
VARIANTS = ('frozen_p6', 'current_p6', 'p6_7s')


def leading_grid() -> Grid:
    from semgpu2.tools.mesh.numbering import with_leading_nodes
    base = grid(Case(6, 10))
    mesh, old_to_new = with_leading_nodes(base.mesh, base.top)
    cells = old_to_new[base.cells]
    np.testing.assert_array_equal(mesh.nodes_coords[cells], base.mesh.nodes_coords[base.cells])
    top = old_to_new[base.top]
    np.testing.assert_array_equal(top, np.arange(1261))
    np.testing.assert_allclose(mesh.nodes_coords[[315,420,525,630,735,840,945]],
                               np.column_stack((SOURCE_POSITIONS, np.zeros(7))), atol=1e-9)
    assert set(mesh.strata) == {1}
    return Grid(mesh, cells, base.centers, top, base.axis_x, base.axis_y, base.weights)


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    g = leading_grid()
    pf = pml(g)
    assert len(g.mesh.nodes_coords) == 795691 and len(g.cells) == 22050 and len(pf.entities) == 12250
    write(root/'plan.json', {'status':'cpu_checked', 'variants':list(VARIANTS), 'runs':12, 'probes':3,
        'geometry_m':[1400,700], 'PML_m':350, 'dmax_s_inv':173.0886108476381,
        'order':6, 'h_m':10, 'duration_s':1, 'cfl_safety':.8, 'beta':1/27, 'gamma':.5,
        'source_x_m':list(SOURCE_POSITIONS), 'surface_shape':[1001,8827,2], 'frames':0,
        'new_solver':'one mesh, seven state blocks; built-in analytic Ricker30Hz, t0=.05, Fy=-1N/m',
        'references':'seven disconnected copies, cubic interpolation of 1us Ricker table',
        'protocol':'one warmup per variant, three alternating repeats, same GPU0; compare all seven shots separately',
        'estimated_surface_bytes':12*1001*8827*16, 'snapshot':file_record(SNAPSHOT),
        'acceptance':'component relative L2<1e-8 and peak-relative difference<1e-7; full time and full top'})


def result_path(root: Path, variant: str, label: str) -> Path:
    return root/variant/'p6_h10'/'fixed'/('batch7_'+label)


def run_seven(root: Path, label: str, probe: bool, dt: float | None) -> dict[str,object]:
    space(20)
    out = result_path(root,'p6_7s',label);out.mkdir(parents=True,exist_ok=False)
    start = perf_counter();g=leading_grid();mesh=g.mesh
    prefix='semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_p6_7s_cu'
    mod=importlib.import_module(prefix+'.solver');const=importlib.import_module(prefix+'.constants');native=importlib.import_module(prefix+'._native')
    cf=cast(Callable[...,SGBaseSolverConfig],const.SGSolverConfig)
    sf=cast(Callable[[SGBaseSolverConfig],SeismicSolver],mod.SGSolver)
    solver=sf(cf(t_begin=0.,t_end=1.,step_dt=dt,cfl_safety=.8 if dt is None else None,tact_dt=.001,kim_beta=1/27,kim_gamma=.5))
    model=SGModel(mesh,solver);vp,vs=material(g)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(g.cells),dtype=np.int64),plane_strain_parameters(vp,vs)))
    model.add_physics(pml(g))
    surface=np.empty((1001,8827,2),dtype=np.float64);ns=0
    def handler(frame: FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal ns
        checked_time(frame,event,ns*.001);assert ns<1001 and np.isfinite(frame).all()
        surface[ns]=frame.reshape(7,2,1261).transpose(0,2,1).reshape(8827,2);ns+=1
    # Native receiver codes advance by six per shot, distinct from solution rows.
    components=np.array([6*s+c for s in range(7) for c in (4,5)],dtype=np.int32)
    if not probe:
        model.add_receiver(SGReceiverModel.from_positions('surface',mesh,0,g.top,components,begin_tact=0,end_tact=1000,stride=1,handler=handler))
    prep=perf_counter();model.prepare();prepared=perf_counter();s=cast(Spectrum,solver.spectrum)
    assert native.__file__ is not None
    record:dict[str,object]={'status':'prepared','variant':'p6_7s','dt_s':s.step_dt,'steps':s.steps_count,
        'steps_per_ms':s.steps_per_tact,'lambda_max':s.lambda_max,'lambda_max_damping':s.lambda_max_damping,
        'assembly_seconds':prep-start,'prepare_seconds':prepared-prep,'nodes_count':len(mesh.nodes_coords),
        'elements':len(g.cells),'pml_elements':len(solver.pml_entity_ids),'copies':7,'ABC':False,'boundary_elements':0,
        'source_x_m':list(SOURCE_POSITIONS),'source':'builtin analytic Ricker30Hz/Fy=-1/t0=.05',
        'source_table_samples':s.steps_count,'native':file_record(Path(native.__file__)), 'snapshot':file_record(SNAPSHOT),
        'runner':file_record(Path(__file__)),'field_receiver_registered':False,'output_mode':'surface_only'}
    assert solver.pml_history is not None and solver.pml_history.shape==(28,600250)
    record['pml_history_shape']=list(solver.pml_history.shape)
    if probe:
        write(out/'probe.json',record);return record
    write(out/'started.json',record);print('STARTED',label,s.step_dt,flush=True)
    begin=perf_counter();result=cast(TimedResult,model.run(sync=True));end=perf_counter()
    assert int(result.code)==int(const.SGRunCodeEnum.TIME_END) and result.steps_count==9000 and ns==1001
    history=np.asarray(solver.pml_history[:,:]);assert np.isfinite(history).all()
    np.savez(out/'surface.npz',time_s=np.arange(1001)*.001,coordinates_m=np.tile(mesh.nodes_coords[g.top],(7,1)),velocity_m_s=surface)
    record.update(status='completed',run_seconds=end-begin,solve_wall_seconds=end-start,total_seconds=perf_counter()-start,
        surface_samples=ns,frames=0,pml_history_max_abs=float(np.max(abs(history))))
    record['surface.npz']=file_record(out/'surface.npz');write(out/'result.json',record)
    print('COMPLETED',label,record['run_seconds'],flush=True);return record


def run(root:Path,variant:str,label:str,probe:bool,dt:float|None)->None:
    if variant=='p6_7s':run_seven(root,label,probe,dt);return
    r=execute(root/variant,Case(6,10),'fixed','batch7_'+label,dt,probe,
              source_positions=SOURCE_POSITIONS,save_fields=False)
    r['variant']=variant
    r['comparison_snapshot']=file_record(SNAPSHOT)
    # execute records its historical setup snapshot; actual binary SHA determines implementation.
    write(result_path(root,variant,label)/'comparison_metadata.json',r)


def series(root:Path)->None:
    assert read_json(root/'plan.json')['status']=='cpu_checked'
    assert not (root/'environment.json').exists()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    old=Path('.runtime_ch15_fixed_pml_orders_v1/venv/bin/python');new=Path('.runtime_ch15_p6_7s_v1/venv/bin/python')
    write(root/'environment.json',{'hostname':platform.node(),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),'snapshot':file_record(SNAPSHOT)})
    def launch(variant:str,label:str,probe:bool=False)->dict[str,object]:
        space(20);python=old if variant=='frozen_p6' else new
        cmd=[str(python),'-m','scripts.dev_1_5_geometry_frequency.p6_7s_compare','--root',str(root),'--variant',variant,'--label',label]
        if probe:cmd+=['--probe']
        with (root/f'{variant}_{label}.log').open('x') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return read_json(result_path(root,variant,label)/('probe.json' if probe else 'result.json'))
    probes={v:launch(v,'probe',True) for v in VARIANTS}
    write(root/'step_plan.json',dict(probes))
    assert all(number(p['dt_s'])==1/9000 and p['steps']==9000 for p in probes.values())
    for v in VARIANTS:launch(v,'warmup')
    for i in range(3):
        for v in VARIANTS[i:]+VARIANTS[:i]:launch(v,f'repeat{i}')
    write(root/'runs_completed.json',{'status':'completed','runs':12,'probes':3,'production':'unchanged, stopped'})


def analyze(root:Path)->None:
    assert read_json(root/'runs_completed.json')['status']=='completed'
    out=root/'analysis';out.mkdir(exist_ok=False)
    verified=[];timings=[];pairs=[]
    historical=Path('data/dev_1_5_geometry_frequency/pml_p6_sources7_surface_v1/p6_h10/fixed/batch7_repeat0')
    def load(path:Path)->FloatArray:
        r=read_json(path/'result.json');assert r['status']=='completed'
        f=path/'surface.npz';assert file_record(f)['sha256']==mapping(r['surface.npz'])['sha256']
        if str(f) not in verified:verified.append(str(f))
        with np.load(f) as z:
            np.testing.assert_array_equal(z['time_s'],np.arange(1001)*.001)
            coords=np.asarray(z['coordinates_m']);assert coords.shape==(8827,2)
            for i in range(7):np.testing.assert_array_equal(coords[i*1261:(i+1)*1261],coords[:1261])
            np.testing.assert_allclose(coords[:1261],grid(Case(6,10)).mesh.nodes_coords[-1261:],atol=1e-12,rtol=0)
            v=np.asarray(z['velocity_m_s'],dtype=np.float64);assert v.shape==(1001,8827,2) and np.isfinite(v).all()
            return v
    hist=load(historical)
    for variant in VARIANTS:
        rows=[read_json(result_path(root,variant,f'repeat{i}')/'result.json') for i in range(3)]
        timings.append({'variant':variant,**{key:[number(r[key]) for r in rows] for key in ('run_seconds','assembly_seconds','prepare_seconds','solve_wall_seconds')},
            'median_run_seconds':float(np.median([number(r['run_seconds']) for r in rows])),
            'median_solve_wall_seconds':float(np.median([number(r['solve_wall_seconds']) for r in rows]))})
    for label in ['warmup','repeat0','repeat1','repeat2']:
        ref=load(result_path(root,'frozen_p6',label))
        for variant in VARIANTS:
            actual=load(result_path(root,variant,label))
            for shot in range(7):
                a=actual[:,shot*1261:(shot+1)*1261];b=ref[:,shot*1261:(shot+1)*1261]
                check=equivalence(a,b);assert check['within_tolerance'],(variant,label,shot,check)
                historical_check=equivalence(a,hist[:,shot*1261:(shot+1)*1261]);assert historical_check['within_tolerance']
                pairs.append({'variant':variant,'label':label,'source_x_m':SOURCE_POSITIONS[shot],**check,'historical':historical_check})
    write(out/'metrics.json',{'status':'completed','timings':timings,'comparisons':pairs,'verified_sha256':verified,
        'max_relative_l2':max(max(cast(list[float],r['relative_l2_components'])) for r in pairs),
        'limits':'surface only, 1s, no full-field stability validation; built-in analytic vs interpolated source; numerical equivalence not spatial convergence'})
    print('ANALYZED',timings,flush=True)


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--prepare',action='store_true');p.add_argument('--series',action='store_true');p.add_argument('--analyze',action='store_true')
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--label');p.add_argument('--probe',action='store_true');p.add_argument('--dt',type=float)
    a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.series:series(a.root)
    elif a.analyze:analyze(a.root)
    else:
        assert a.variant and a.label;run(a.root,a.variant,a.label,a.probe,a.dt)

if __name__=='__main__':main()
