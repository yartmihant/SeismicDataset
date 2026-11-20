"""Full corrected-damping p4/p5 problem: separate-process warmups and three rotated paired repeats."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import platform
import numpy as np
from scripts.dev_1_5_geometry_frequency.setup import ROOT, Spec, write
from scripts.dev_1_5_geometry_frequency.run import execute, space
from scripts.dev_1_5_geometry_frequency.compare_solvers import difference
from scripts.dev_1_3_point_source.run import file_record
from scripts.reporting import read_json, number, mapping

def backends(order: int) -> tuple[str,str,str]:
    return ('tenzor','seismic',f'seismic_p{order}')
CASE='1400x700_two_layer_30Hz'


def analyze(out: Path, order: int) -> None:
    variants=backends(order)
    comparisons: list[dict[str,object]]=[]
    timings: list[dict[str,object]]=[]
    for backend in variants:
        rows=[read_json(out/f'{backend}_{i}'/CASE/'result.json') for i in range(3)]
        timings.append({'backend':backend,**{key:[number(r[key]) for r in rows] for key in ('run_seconds','prepare_seconds','assembly_seconds','total_seconds')},
                        'median_run_seconds':float(np.median([number(r['run_seconds']) for r in rows])),
                        'median_total_seconds':float(np.median([number(r['total_seconds']) for r in rows]))})
    for i in range(3):
        ref=out/f'tenzor_{i}'/CASE
        for backend in variants[1:]:
            run=out/f'{backend}_{i}'/CASE
            a=read_json(run/'result.json');b=read_json(ref/'result.json')
            assert a['status']==b['status']=='completed'
            assert a['order']==b['order']==order
            assert a['damping_nodes']==b['damping_nodes'] and a['setup_code']==b['setup_code']
            assert a['dt_s']==b['dt_s'] and a['steps']==b['steps'] and a['mesh']==b['mesh']
            np.testing.assert_array_equal(np.load(run/'nodes.npy'),np.load(ref/'nodes.npy'))
            fields=difference(np.load(run/'velocity.npy',mmap_mode='r'),np.load(ref/'velocity.npy',mmap_mode='r'))
            with np.load(run/'surface.npz') as av,np.load(ref/'surface.npz') as bv:
                np.testing.assert_array_equal(av['time_s'],bv['time_s'])
                np.testing.assert_array_equal(av['coordinates_m'],bv['coordinates_m'])
                surface=difference(av['velocity_m_s'],bv['velocity_m_s'])
            comparisons.append({'backend':backend,'reference':'tenzor','repeat':i,'fields':fields,'surface':surface})
    historical: list[dict[str,object]]=[]
    if order==5:
        for i in range(3):
            before=ROOT.parent/'p5_saturn_v1'/f'tenzor_{i}'/CASE
            after=out/f'tenzor_{i}'/CASE
            old=read_json(before/'result.json');new=read_json(after/'result.json')
            assert old['dt_s']==new['dt_s'] and mapping(old['native'])['sha256']==mapping(new['native'])['sha256']
            np.testing.assert_array_equal(np.load(before/'nodes.npy'),np.load(after/'nodes.npy'))
            fields=difference(np.load(after/'velocity.npy',mmap_mode='r'),np.load(before/'velocity.npy',mmap_mode='r'))
            with np.load(after/'surface.npz') as a,np.load(before/'surface.npz') as b:
                surface=difference(a['velocity_m_s'],b['velocity_m_s'])
            historical.append({'repeat':i,'fields':fields,'surface':surface})
    verified=[]
    for run in sorted(out.glob('*/'+CASE+'/result.json')):
        record=read_json(run)
        for key,name in [('fields','velocity.npy'),('nodes','nodes.npy'),('surface','surface.npz')]:
            assert file_record(run.parent/name)['sha256']==mapping(record[key])['sha256']
        verified.append(str(run.relative_to(out)))
    agree=all(mapping(c[k])['within_tolerance'] for c in comparisons for k in ('fields','surface'))
    write(out/'analysis.json',{'status':'completed','order':order,'timings':timings,'comparisons':comparisons,'all_agree':agree,'verified_runs':verified,'tenzor_before_after_damping_support':historical,
        'method':'one full warmup per backend excluded; three rotated repeats, separate processes on same GPU, full fields and surface retained; synchronous run includes receiver output',
        'tolerance':'component relative L2 <1e-8 and peak difference/reference peak <1e-7'})
    assert agree


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--order',type=int,choices=(4,5),required=True)
    parser.add_argument('--backend',choices=('tenzor','seismic','seismic_p4','seismic_p5'))
    parser.add_argument('--label')
    parser.add_argument('--dt',type=float)
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve();variants=backends(args.order)
    if args.analyze:analyze(out,args.order);return
    if args.backend:
        assert args.label
        execute(Spec(1400,700,'two_layer',30),output_root=out/args.label,backend=args.backend,explicit_dt=args.dt,order=args.order)
        return
    space();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(out/'environment.json',{'hostname':platform.node(),'python':sys.executable,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':file_record(Path('docs/runtime_sources/seismic_p45_damping_v1/provenance.json'))})
    def launch(backend: str,label: str,dt: float | None=None) -> None:
        cmd=[sys.executable,'-m','scripts.dev_1_5_geometry_frequency.compare_orders','--out',str(out),'--order',str(args.order),'--backend',backend,'--label',label]
        if dt is not None:cmd+=['--dt',str(dt)]
        print('LAUNCH',label,flush=True)
        with (out/f'{label}.log').open('x') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for backend in variants:launch(backend,'warmup_'+backend)
    dts=[number(read_json(out/f'warmup_{b}'/CASE/'result.json')['dt_s']) for b in variants]
    common=min(dts) if len(set(dts))>1 else None
    write(out/'plan.json',{'backends':variants,'native_dt_s':dts,'explicit_common_dt_s':common,'repeats':3,'warmups':3,'spec':CASE,'order':args.order,'element_m':5})
    for i in range(3):
        for backend in variants[i:]+variants[:i]:launch(backend,f'{backend}_{i}',common)
    analyze(out,args.order)
    print('COMPLETED',out,flush=True)


if __name__=='__main__':main()
