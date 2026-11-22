"""Fixed-order PML benchmark; rectangular volume mesh with no boundary stratum."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import os
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Callable, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from semgpu2.model import SGModel, SGPETALineModel, SGPhysicsModel
from semgpu2.model.mesh import SGMeshModel, SGMeshStratumData
from semgpu2.model.physics import SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseSolver, SGBaseSolverConfig, SGBaseReceiverEventData, SGReceiverRunControl, SGStateArray
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.run import checked_time, file_record, TimedResult
from scripts.dev_1_5_geometry_frequency.run import space
from scripts.dev_1_5_geometry_frequency.setup import ALPHA, write
from scripts.reporting import read_json, number, mapping

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ROOT = Path(__file__).resolve().parents[2] / 'data/dev_1_5_geometry_frequency/pml_fixed_orders_v1'
SNAPSHOT = Path('docs/runtime_sources/seismic_fixed_pml_orders_v1/provenance.json')
PREFIX = 'semgpu2.solvers.lin_elast_2d_seismic_newmark_kim'
W, H, PAD = 1400, 700, 350


@dataclass(frozen=True)
class Case:
    order: int
    h: int

    @property
    def label(self) -> str:
        return f'p{self.order}_h{self.h}'


CASES = (Case(4, 5), Case(5, 5), Case(6, 10), Case(7, 10))
CONTROLS = (Case(8, 5), Case(9, 5))


class Spectrum(Protocol):
    step_dt: float
    steps_count: int
    steps_per_tact: int
    lambda_max: float
    lambda_max_damping: float


class SeismicSolver(SGBaseSolver[SGBaseSolverConfig]):
    pml_history: SGStateArray | None
    pml_entity_ids: IntArray


@dataclass
class Grid:
    mesh: SGMeshModel
    cells: IntArray
    centers: FloatArray
    top: IntArray
    axis_x: FloatArray
    axis_y: FloatArray
    weights: FloatArray


def grid(case: Case) -> Grid:
    """Direct tensor-product numbering: create only 2D elements, never edge elements."""
    p, h = case.order, case.h
    nx, ny = (W + 2*PAD)//h, (H+PAD)//h
    assert nx*h == W+2*PAD and ny*h == H+PAD
    line = SGPETALineModel('GLL', order=p)
    ref = np.asarray(line.p_nodes.coords[:, 0], dtype=np.float64)
    def axis(n: int, begin: float) -> FloatArray:
        return np.r_[begin + h*(np.arange(n)[:, None] + (ref[:-1]+1)/2).ravel(), begin+n*h]
    x, y = axis(nx, -PAD), axis(ny, -H-PAD)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    ex, ey = np.meshgrid(np.arange(nx), np.arange(ny))
    local = np.arange(p+1)
    cells = ((ey.ravel()[:, None, None]*p + local[None, :, None]) * len(x)
             + ex.ravel()[:, None, None]*p + local[None, None, :]).reshape(-1, (p+1)**2).astype(np.int64)
    ix, iy = np.meshgrid(np.arange(len(x)), np.arange(len(y)))
    tags = np.zeros((len(coords), 3), dtype=np.int64)
    tags[:, 0] = ((ix.ravel()%p != 0).astype(np.int64) + (iy.ravel()%p != 0))
    tags[:, 1] = 2 - ((ix.ravel() == 0) | (ix.ravel() == len(x)-1)).astype(np.int64) - ((iy.ravel() == 0) | (iy.ravel() == len(y)-1)).astype(np.int64)
    count = len(cells)
    strata = {1: SGMeshStratumData((2, 2, 2), cells.ravel(), np.arange(count+1, dtype=np.int64)*(p+1)**2,
                                 np.zeros(2*count, dtype=np.int32), np.arange(count+1, dtype=np.int64)*2, np.ones(count, dtype=np.int64))}
    mesh = SGMeshModel(coords, tags, [line], strata)
    centers = np.column_stack((-PAD+(ex.ravel()+.5)*h, -H-PAD+(ey.ravel()+.5)*h))
    weights = np.asarray(line.build_data(use_symmetry=False).weights, dtype=np.float64)
    return Grid(mesh, cells, centers, np.arange(len(coords)-len(x), len(coords), dtype=np.int64), x, y, weights)


def material(g: Grid) -> tuple[FloatArray, FloatArray]:
    lower = g.centers[:, 1] < -350
    return np.where(lower, 3963., 4087.), np.where(lower, 2357.84473, 2431.62036)


def pml(g: Grid) -> SGPhysicsModel:
    c = g.centers
    ids = np.flatnonzero((c[:, 0] < 0) | (c[:, 0] > W) | (c[:, 1] < -H))
    xy = g.mesh.nodes_coords[g.cells[ids].ravel()]
    sx = np.maximum(np.maximum(-xy[:, 0], xy[:, 0]-W), 0)/PAD
    sy = np.maximum(-xy[:, 1]-H, 0)/PAD
    return SGPhysicsModel('cartesian_pml', 31100, g.mesh, 1, ids, ALPHA*np.column_stack((sx*sx, sy*sy)))


def interpolate_surface(values: FloatArray, case: Case) -> FloatArray:
    """Evaluate SEM boundary polynomial at common x=0,1,...,1400m, no edge entities."""
    p, h = case.order, case.h
    ref = np.asarray(SGPETALineModel('GLL', order=p).p_nodes.coords[:, 0], dtype=np.float64)
    target = np.arange(W+1, dtype=np.float64)
    element = np.floor((target+PAD)/h).astype(np.int64)
    xi = 2*((target+PAD)/h-element)-1
    basis = np.ones((len(target), p+1))
    for j in range(p+1):
        for k in range(p+1):
            if k != j:
                basis[:, j] *= (xi-ref[k])/(ref[j]-ref[k])
    ids = element[:, None]*p+np.arange(p+1)
    return np.asarray(np.einsum('txjc,xj->txc', values[:, ids], basis), dtype=np.float64)


def prepare(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, object]] = []
    for case in (*CASES, *CONTROLS):
        g = grid(case)
        assert set(g.mesh.strata) == {1}
        assert len(np.unique(g.cells)) == len(g.mesh.nodes_coords)
        assert np.allclose(g.mesh.nodes_coords[g.cells[:, -1]]-g.mesh.nodes_coords[g.cells[:, 0]], case.h)
        assert np.count_nonzero(np.all(abs(g.mesh.nodes_coords-[700, 0]) < 1e-9, axis=1)) == 1
        assert np.allclose(g.mesh.nodes_coords[g.top, 1], 0)
        pf = pml(g); pv = pf.build_data().values
        assert np.isclose(pv.max(), ALPHA) and np.all(pv >= 0)
        vp, vs = material(g)
        assert set(vp) == {4087., 3963.} and len(vs) == len(g.cells)
        # Polynomial reproduction on every element, also odd orders without midpoints.
        scaled = g.axis_x/W
        test = np.stack([scaled**min(case.order, 5), np.ones_like(scaled)], axis=-1)[None]
        sample = interpolate_surface(test, case)
        np.testing.assert_allclose(sample[0, :, 0], (np.arange(W+1)/W)**min(case.order, 5), atol=2e-14)
        np.testing.assert_allclose(sample[0, :, 1], 1., atol=2e-14)
        rows.append({'case':case.label, 'order':case.order, 'h_m':case.h, 'nodes':len(g.mesh.nodes_coords), 'elements':len(g.cells), 'pml_elements':len(pf.entities), 'boundary_elements':0, 'raw_run_bytes':21*len(g.mesh.nodes_coords)*16+1001*len(g.top)*16+len(g.mesh.nodes_coords)*16})
    budget = sum(int(number(r['raw_run_bytes']))*10 for r in rows[:4]) + int(number(rows[4]['raw_run_bytes'])) + 2*int(number(rows[5]['raw_run_bytes']))
    write(root/'plan.json', {'status':'cpu_checked', 'cases':rows, 'runs':43, 'probes':9, 'estimated_raw_bytes':budget,
        'physical_domain_m':[0, W, -H, 0], 'full_domain_m':[-PAD, W+PAD, -H-PAD, 0], 'source':'central downward Ricker30Hz,1N/m,t0=.05s',
        'PML':'quadratic fixed dmax173.0886108476381,350m; no ABC, no boundary stratum', 'CFL':.8, 'kim_beta':1/27, 'duration_s':1,
        'timing':'generic and fixed, each one full warmup and three repeats; same GPU, sequential; native dt, assert dt agrees within each pair',
        'accuracy':'same physical receivers x0..1400/1m using SEM interpolation, internal receivers only; native, half-native and common dt; p8/p9 h5 controls and p9 common_dt/2',
        'common_dt':'min(all native steps including p9/h5)/2', 'reference_limit':'same PML350, no exact PDE solution; p8/p9 and p9 half-step discrepancies quantify control uncertainty',
        'snapshot':file_record(SNAPSHOT)})


def execute(root: Path, case: Case, backend: str, label: str, dt: float | None, probe: bool=False, *, source_positions:tuple[float,...]=(700.,)) -> dict[str, object]:
    space(); out=root/case.label/backend/label; out.mkdir(parents=True, exist_ok=False)
    start=perf_counter(); base=grid(case); copies=len(source_positions); nodes_per_copy=len(base.mesh.nodes_coords); top_per_copy=len(base.top)
    g=pack_grid(base,copies); mesh=g.mesh; coords=mesh.nodes_coords
    suffix='_cu' if backend == 'generic' else f'_p{case.order}_cu'
    mod=importlib.import_module(PREFIX+suffix+'.solver')
    constants=importlib.import_module(PREFIX+suffix+'.constants')
    native=importlib.import_module(PREFIX+suffix+'._native')
    config_factory=cast(Callable[..., SGBaseSolverConfig], constants.SGSolverConfig)
    solver_factory=cast(Callable[[SGBaseSolverConfig], SeismicSolver], mod.SGSolver)
    solver=solver_factory(config_factory(t_begin=0., t_end=1., step_dt=dt, cfl_safety=.8 if dt is None else None, tact_dt=.001, kim_beta=1/27, kim_gamma=.5))
    model=SGModel(mesh, solver)
    vp, vs=material(g)
    model.add_physics(SGPhysicsModel.from_entity_values('material',11000,mesh,1,np.arange(len(g.cells),dtype=np.int64),plane_strain_parameters(vp,vs)))
    pf=pml(g); model.add_physics(pf)
    source_ids=[]
    for i,x in enumerate(source_positions):
        found=np.flatnonzero(np.all(abs(base.mesh.nodes_coords-[x,0])<1e-9,axis=1));assert len(found)==1
        source_ids.append(int(found[0]+i*nodes_per_copy))
    source=np.asarray(source_ids,dtype=np.int64)
    times=np.linspace(0,1,1000001); phase=np.pi*30*(times-.05)
    force=np.zeros((len(times),copies,2)); force[:,:,1]=(-(1-2*phase**2)*np.exp(-phase**2))[:,None]
    model.add_physics(SGPhysicsModel('point_force',20200,mesh,0,source,force,time=SGPhysicsTimeData(0.,1.,len(times),3)))
    assert native.__file__ is not None
    record:dict[str,object]={'status':'prepared','case':case.label,'order':case.order,'h_m':case.h,'backend':backend,'ABC':False,'boundary_elements':0,'strata':[1], 'nodes_count':len(coords),'elements':len(g.cells),'pml_elements':len(pf.entities),'native':file_record(Path(native.__file__)), 'snapshot':file_record(SNAPSHOT), 'runner':file_record(Path(__file__)), 'explicit_dt_s':dt,'cfl_safety':.8 if dt is None else None,'kim_beta':1/27,'dmax_s_inv':ALPHA,'duration_s':1,'source_hz':30,'source_xy_m':[[x,0.] for x in source_positions],'source_x_m':list(source_positions),'copies':copies,'nodes_per_copy':nodes_per_copy,'surface_nodes_per_copy':top_per_copy,'gpu':os.environ.get('CUDA_VISIBLE_DEVICES')}
    if probe:
        model.prepare(); s=cast(Spectrum, solver.spectrum)
        record.update(dt_s=s.step_dt, steps=s.steps_count, steps_per_ms=s.steps_per_tact, lambda_max=s.lambda_max, lambda_max_damping=s.lambda_max_damping, total_seconds=perf_counter()-start)
        write(out/'probe.json',record); return record
    surface=np.empty((1001,len(g.top),2)); ns=0; nf=0
    fields=np.lib.format.open_memmap(out/'velocity.npy',mode='w+',dtype=np.float64,shape=(21,len(coords),2))
    peaks: list[float]=[]; core_energy: list[float]=[]; pml_peaks: list[float]=[]
    core=(g.centers[:,0]>0)&(g.centers[:,0]<W)&(g.centers[:,1]>-H)
    local=(1000*(case.h/2)**2*np.outer(g.weights,g.weights)).ravel()
    mass=np.bincount(g.cells[core].ravel(), weights=np.tile(local,np.count_nonzero(core)), minlength=len(coords))
    assert np.isclose(mass.sum(),copies*1000*W*H)
    layer_nodes=(coords[:,0]<0)|(coords[:,0]>W)|(coords[:,1]<-H)
    def surface_handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal ns
        checked_time(frame,event,ns*.001);assert ns<1001 and np.isfinite(frame).all()
        surface[ns]=frame.reshape(2,-1).T;ns+=1
    def field_handler(frame:FloatArray,event:SGBaseReceiverEventData,control:SGReceiverRunControl)->None:
        nonlocal nf
        space(20);checked_time(frame,event,nf*.05);assert nf<21 and np.isfinite(frame).all()
        fields[nf]=frame.reshape(2,-1).T;nf+=1
    components=np.array([4,5],dtype=np.int32)
    model.add_receiver(SGReceiverModel.from_positions('surface',mesh,0,g.top,components,begin_tact=0,end_tact=1000,stride=1,handler=surface_handler))
    model.add_receiver(SGReceiverModel.from_positions('fields',mesh,0,np.arange(len(coords),dtype=np.int64),components,begin_tact=0,end_tact=1000,stride=50,handler=field_handler))
    prep=perf_counter();model.prepare(); prep_end=perf_counter();s=cast(Spectrum, solver.spectrum)
    assert len(solver.pml_entity_ids)==len(pf.entities)
    record.update(dt_s=s.step_dt,steps=s.steps_count,steps_per_ms=s.steps_per_tact,lambda_max=s.lambda_max,lambda_max_damping=s.lambda_max_damping,assembly_seconds=prep-start,prepare_seconds=prep_end-prep)
    write(out/'started.json',record); print('STARTED',case.label,backend,label,s.steps_count,flush=True)
    run_start=perf_counter();result=cast(TimedResult,model.run(sync=True));run_end=perf_counter()
    assert int(result.code)==int(constants.SGRunCodeEnum.TIME_END) and result.steps_count==s.steps_count and ns==1001 and nf==21
    fields.flush(); np.testing.assert_array_equal(fields[:,g.top],surface[::50])
    for frame in fields:
        peaks.append(float(np.max(np.linalg.norm(frame,axis=1))))
        pml_peaks.append(float(np.max(np.linalg.norm(frame[layer_nodes],axis=1))))
        core_energy.append(float(.5*np.sum(mass*np.sum(frame**2,axis=1))))
    assert solver.pml_history is not None
    history=np.asarray(solver.pml_history[:,:]); assert np.isfinite(history).all()
    np.save(out/'nodes.npy',coords)
    np.savez(out/'surface.npz',time_s=np.arange(1001)*.001,coordinates_m=coords[g.top],velocity_m_s=surface)
    common=np.stack([interpolate_surface(surface[:,i*top_per_copy:(i+1)*top_per_copy],case) for i in range(copies)])
    np.savez(out/'common_surface.npz',time_s=np.arange(1001)*.001,x_m=np.arange(W+1),velocity_m_s=common[0] if copies==1 else common,source_x_m=np.asarray(source_positions))
    record.update(status='completed',solve_wall_seconds=run_end-start,run_seconds=run_end-run_start,total_seconds=perf_counter()-start,frames=nf,surface_samples=ns,surface_field_exact=True,pml_history_max_abs=float(np.max(abs(history))),field_peak_m_s=peaks,pml_peak_m_s=pml_peaks,core_kinetic_energy=core_energy)
    for name in ('velocity.npy','nodes.npy','surface.npz','common_surface.npz'):
        record[name]=file_record(out/name)
    write(out/'result.json',record);print('COMPLETED',case.label,backend,label,record['run_seconds'],flush=True)
    return record


def series(root:Path)->None:
    assert read_json(root/'plan.json')['status']=='cpu_checked'
    if (root/'environment.json').exists():raise FileExistsError('Series already started; inspect results before resuming explicitly')
    space();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(root/'environment.json',{'hostname':platform.node(),'python':sys.executable,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),'snapshot':file_record(SNAPSHOT)})
    def launch(case:Case,backend:str,label:str,dt:float|None=None,probe:bool=False)->dict[str,object]:
        cmd=[sys.executable,'-m','scripts.dev_1_5_geometry_frequency.pml_fixed_orders','--root',str(root),'--case',case.label,'--backend',backend,'--label',label]
        if dt is not None:cmd+=['--dt',str(dt)]
        if probe:cmd+=['--probe']
        with (root/f'{case.label}_{backend}_{label}.log').open('x') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return read_json(root/case.label/backend/label/('probe.json' if probe else 'result.json'))
    dts:dict[str,float]={}
    for case in CASES:
        a=launch(case,'generic','probe',probe=True); fixed_probe=launch(case,'fixed','probe',probe=True)
        assert a['dt_s']==fixed_probe['dt_s'],(case,a['dt_s'],fixed_probe['dt_s'])
        dts[case.label]=number(a['dt_s'])
    ref=launch(CONTROLS[-1],'fixed','probe',probe=True)
    common=min(*dts.values(),number(ref['dt_s']))/2
    write(root/'step_plan.json',{'native_dt_s':dts,'p9_native_dt_s':ref['dt_s'],'common_dt_s':common,'finest_reference_dt_s':common/2})
    jobs=[(case,b) for case in CASES for b in ('generic','fixed')]
    for case,b in jobs:launch(case,b,'warmup')
    for repeat in range(3):
        rotated=jobs[repeat:]+jobs[:repeat]
        for case,b in rotated:launch(case,b,f'repeat{repeat}')
    for case in CASES:
        launch(case,'fixed','half_dt',dts[case.label]/2)
        launch(case,'fixed','common_dt',common)
    for case in CONTROLS:launch(case,'fixed','common_dt',common)
    launch(CONTROLS[-1],'fixed','half_common_dt',common/2)
    write(root/'runs_completed.json',{'status':'completed','runs':43,'probes':9,'common_dt_s':common,'production_queue':'not modified'})


BATCH_ROOT=ROOT.parent/'pml_p6_sources7_v1'
SOURCE_POSITIONS=(175.,350.,525.,700.,875.,1050.,1225.)


def pack_grid(base:Grid,copies:int)->Grid:
    """Disconnected mesh components share coordinates but never node IDs."""
    if not 1<=copies<=7:raise ValueError(copies)
    if copies==1:return base
    n=len(base.mesh.nodes_coords);e=len(base.cells);q=base.cells.shape[1]
    cells=np.concatenate([base.cells+i*n for i in range(copies)])
    assert all(np.array_equal(cells[i*e:(i+1)*e]-i*n,base.cells) for i in range(copies))
    count=e*copies
    strata={1:SGMeshStratumData((2,2,2),cells.ravel(),np.arange(count+1,dtype=np.int64)*q,
        np.zeros(2*count,dtype=np.int32),np.arange(count+1,dtype=np.int64)*2,np.ones(count,dtype=np.int64))}
    mesh=SGMeshModel(np.tile(base.mesh.nodes_coords,(copies,1)),np.tile(base.mesh.nodes_tags,(copies,1)),list(base.mesh.pets),strata)
    return Grid(mesh,cells,np.tile(base.centers,(copies,1)),np.concatenate([base.top+i*n for i in range(copies)]),base.axis_x,base.axis_y,base.weights)


def batch_prepare(root:Path)->None:
    root.mkdir(parents=True,exist_ok=False)
    case=Case(6,10);base=grid(case);packed=pack_grid(base,7)
    n=len(base.mesh.nodes_coords);e=len(base.cells);ids=[]
    for i,x in enumerate(SOURCE_POSITIONS):
        node=np.flatnonzero(np.all(abs(base.mesh.nodes_coords-[x,0])<1e-9,axis=1));assert len(node)==1
        ids.append(int(node[0]+i*n))
        np.testing.assert_array_equal(packed.mesh.nodes_coords[i*n:(i+1)*n],base.mesh.nodes_coords)
        assert np.all(packed.cells[i*e:(i+1)*e]//n==i)
        assert np.all(packed.top[i*len(base.top):(i+1)*len(base.top)]//n==i)
    assert set(packed.mesh.strata)=={1} and len(set(ids))==7
    pf=pml(packed);values=pf.build_data().values;single=pml(base).build_data().values
    for i in range(7):np.testing.assert_array_equal(values[:,i*single.shape[1]:(i+1)*single.shape[1]],single)
    vp,vs=material(packed);v0,s0=material(base)
    np.testing.assert_array_equal(vp,np.tile(v0,7));np.testing.assert_array_equal(vs,np.tile(s0,7))
    raw_single=21*n*16+n*16+1001*len(base.top)*16+1001*1401*16
    write(root/'plan.json',{'status':'cpu_checked','order':6,'h_m':10,'source_x_m':list(SOURCE_POSITIONS),'source_node_ids':ids,'nodes_per_copy':n,'nodes_batch':7*n,'elements_per_copy':e,'elements_batch':7*e,'pml_elements_per_copy':len(pml(base).entities),'boundary_elements':0,'copies':7,'copies_disconnected':True,'translation_m':0,'duration_s':1,'source_hz':30,'source_force_y_n_per_m':-1.,'source_t0_s':.05,'kim_beta':1/27,'PML_thickness_m':350,'PML_dmax_s_inv':ALPHA,'ABC':False,'CFL':.8,'field_frames_per_source':21,'surface_samples_per_source':1001,'runs':32,'probes':2,'estimated_array_bytes':raw_single*7*8,'timing':'7 singles versus one packed model; one full warmup of each source and package, then3 paired rounds; sequential on GPU0. Compare median of summed7 run times with median package run; also assembly+prepare+run.','dt':'native estimate for one copy and disconnected package; require same dt before timing','validation':'compare each of7 package components against corresponding single, fields and surface; same order, dt and point force','snapshot':file_record(SNAPSHOT)})


def batch_series(root:Path)->None:
    assert read_json(root/'plan.json')['status']=='cpu_checked'
    if (root/'environment.json').exists():raise FileExistsError('Batch series already started; inspect before resuming')
    space();env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(root/'environment.json',{'hostname':platform.node(),'python':sys.executable,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),'snapshot':file_record(SNAPSHOT),'production':'stopped; not modified'})
    def launch(name:str,label:str,probe:bool=False)->dict[str,object]:
        cmd=[sys.executable,'-m','scripts.dev_1_5_geometry_frequency.pml_fixed_orders','--root',str(root),'--batch-run',name,'--label',label]
        if probe:cmd+=['--probe']
        with (root/f'{name}_{label}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        return read_json(root/'p6_h10/fixed'/f'{name}_{label}'/('probe.json' if probe else 'result.json'))
    single=launch('x700','probe',True);package=launch('batch7','probe',True)
    write(root/'step_plan.json',{'single_dt_s':single['dt_s'],'batch_dt_s':package['dt_s'],'single_lambda_max':single['lambda_max'],'batch_lambda_max':package['lambda_max'],'native_dt_equal':single['dt_s']==package['dt_s']})
    if single['dt_s']!=package['dt_s']:raise ValueError('Native steps differ; preserve probes and plan equal-step controls before comparing speed')
    for x in SOURCE_POSITIONS:launch(f'x{int(x)}','warmup')
    launch('batch7','warmup')
    rounds=[]
    for repeat in range(3):
        label=f'repeat{repeat}';begin=perf_counter()
        if repeat%2:launch('batch7',label)
        singles_begin=perf_counter()
        positions=SOURCE_POSITIONS[repeat:]+SOURCE_POSITIONS[:repeat]
        for x in positions:launch(f'x{int(x)}',label)
        singles_elapsed=perf_counter()-singles_begin
        if not repeat%2:launch('batch7',label)
        rounds.append({'repeat':repeat,'singles_source_order':list(positions),'batch_first':bool(repeat%2),'singles_wall_seconds_with_process_and_hashing':singles_elapsed,'round_wall_seconds':perf_counter()-begin})
    write(root/'runs_completed.json',{'status':'completed','runs':32,'probes':2,'rounds':rounds,'native_dt_s':single['dt_s'],'production_queue':'not modified'})

def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--prepare',action='store_true');parser.add_argument('--series',action='store_true');parser.add_argument('--case',choices=[c.label for c in (*CASES,*CONTROLS)]);parser.add_argument('--backend',choices=('generic','fixed'),default='fixed');parser.add_argument('--label');parser.add_argument('--dt',type=float);parser.add_argument('--probe',action='store_true');parser.add_argument('--batch-prepare',action='store_true');parser.add_argument('--batch-series',action='store_true');parser.add_argument('--batch-run',choices=['batch7']+[f'x{int(x)}' for x in SOURCE_POSITIONS]);args=parser.parse_args()
    if (args.batch_prepare or args.batch_series or args.batch_run) and args.root==ROOT:args.root=BATCH_ROOT
    if args.batch_prepare:batch_prepare(args.root);return
    if args.batch_series:batch_series(args.root);return
    if args.batch_run:
        assert args.label
        positions=SOURCE_POSITIONS if args.batch_run=='batch7' else (float(args.batch_run[1:]),)
        execute(args.root,Case(6,10),'fixed',args.batch_run+'_'+args.label,args.dt,args.probe,source_positions=positions);return
    if args.prepare:prepare(args.root)
    elif args.series:series(args.root)
    else:
        assert args.case and args.label
        case=next(c for c in (*CASES,*CONTROLS) if c.label==args.case)
        execute(args.root,case,args.backend,args.label,args.dt,args.probe)


if __name__=='__main__':main()
