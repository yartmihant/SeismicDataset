"""Full p5 problem: separate-process warmups and three rotated paired repeats."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import platform
import numpy as np
from scripts.dev_1_5_geometry_frequency.setup import Spec, write
from scripts.dev_1_5_geometry_frequency.run import execute, space
from scripts.dev_1_5_geometry_frequency.compare_solvers import difference
from scripts.dev_1_3_point_source.run import file_record
from scripts.reporting import read_json, number, mapping

BACKENDS=('tenzor','seismic','seismic_p5')
CASE='1400x700_two_layer_30Hz'


def analyze(out: Path) -> None:
    comparisons: list[dict[str,object]]=[]
    timings: list[dict[str,object]]=[]
    for backend in BACKENDS:
        rows=[read_json(out/f'{backend}_{i}'/CASE/'result.json') for i in range(3)]
        timings.append({'backend':backend,**{key:[number(r[key]) for r in rows] for key in ('run_seconds','prepare_seconds','assembly_seconds','total_seconds')},
                        'median_run_seconds':float(np.median([number(r['run_seconds']) for r in rows])),
                        'median_total_seconds':float(np.median([number(r['total_seconds']) for r in rows]))})
    for i in range(3):
        ref=out/f'tenzor_{i}'/CASE
        for backend in ('seismic','seismic_p5'):
            run=out/f'{backend}_{i}'/CASE
            a=read_json(run/'result.json');b=read_json(ref/'result.json')
            assert a['status']==b['status']=='completed'
            assert a['dt_s']==b['dt_s'] and a['steps']==b['steps'] and a['mesh']==b['mesh']
            np.testing.assert_array_equal(np.load(run/'nodes.npy'),np.load(ref/'nodes.npy'))
            fields=difference(np.load(run/'velocity.npy',mmap_mode='r'),np.load(ref/'velocity.npy',mmap_mode='r'))
            with np.load(run/'surface.npz') as av,np.load(ref/'surface.npz') as bv:
                np.testing.assert_array_equal(av['time_s'],bv['time_s'])
                np.testing.assert_array_equal(av['coordinates_m'],bv['coordinates_m'])
                surface=difference(av['velocity_m_s'],bv['velocity_m_s'])
            comparisons.append({'backend':backend,'reference':'tenzor','repeat':i,'fields':fields,'surface':surface})
    verified=[]
    for run in sorted(out.glob('*/'+CASE+'/result.json')):
        record=read_json(run)
        for key,name in [('fields','velocity.npy'),('nodes','nodes.npy'),('surface','surface.npz')]:
            assert file_record(run.parent/name)['sha256']==mapping(record[key])['sha256']
        verified.append(str(run.relative_to(out)))
    agree=all(mapping(c[k])['within_tolerance'] for c in comparisons for k in ('fields','surface'))
    write(out/'analysis.json',{'status':'completed','timings':timings,'comparisons':comparisons,'all_agree':agree,'verified_runs':verified,
        'method':'one full warmup per backend excluded; three rotated repeats, separate processes on same GPU, full fields and surface retained; synchronous run includes receiver output',
        'tolerance':'component relative L2 <1e-8 and peak difference/reference peak <1e-7'})
    assert agree


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--backend',choices=BACKENDS)
    parser.add_argument('--label')
    parser.add_argument('--dt',type=float)
    parser.add_argument('--analyze',action='store_true')
    args=parser.parse_args()
    out=args.out.resolve()
    if args.analyze:analyze(out);return
    if args.backend:
        assert args.label
        execute(Spec(1400,700,'two_layer',30),output_root=out/args.label,backend=args.backend,explicit_dt=args.dt)
        return
    space();out.mkdir(parents=True,exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    write(out/'environment.json',{'hostname':platform.node(),'python':sys.executable,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,driver_version,memory.total,utilization.gpu','--format=csv'],text=True),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':file_record(Path('docs/runtime_sources/seismic_p5_v1/provenance.json'))})
    def launch(backend: str,label: str,dt: float | None=None) -> None:
        cmd=[sys.executable,'-m','scripts.dev_1_5_geometry_frequency.compare_p5','--out',str(out),'--backend',backend,'--label',label]
        if dt is not None:cmd+=['--dt',str(dt)]
        print('LAUNCH',label,flush=True)
        with (out/f'{label}.log').open('x') as log:
            subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for backend in BACKENDS:launch(backend,'warmup_'+backend)
    dts=[number(read_json(out/f'warmup_{b}'/CASE/'result.json')['dt_s']) for b in BACKENDS]
    common=min(dts) if len(set(dts))>1 else None
    write(out/'plan.json',{'backends':BACKENDS,'native_dt_s':dts,'explicit_common_dt_s':common,'repeats':3,'warmups':3,'spec':CASE,'order':5,'element_m':5})
    for i in range(3):
        for backend in BACKENDS[i:]+BACKENDS[:i]:launch(backend,f'{backend}_{i}',common)
    analyze(out)
    print('COMPLETED',out,flush=True)


if __name__=='__main__':main()
