"""Paired A100 timing and velocity agreement; historical runs stay immutable."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from numpy.typing import NDArray
from scripts.dev_1_5_geometry_frequency.setup import ROOT, Spec, write
from scripts.dev_1_5_geometry_frequency.run import execute, space
from scripts.reporting import read_json, number, mapping

OUT=ROOT.parent/'solver_comparison_v1'


def difference(a: NDArray[np.float64], b: NDArray[np.float64]) -> dict[str, object]:
    assert a.shape==b.shape
    sums=np.zeros(2); refs=np.zeros(2); peak=np.zeros(2); refpeak=np.zeros(2)
    per_frame=[]
    for av,bv in zip(a,b):
        assert np.isfinite(av).all() and np.isfinite(bv).all()
        d=av-bv
        sums+=np.sum(d*d,axis=0);refs+=np.sum(bv*bv,axis=0)
        peak=np.maximum(peak,np.max(np.abs(d),axis=0));refpeak=np.maximum(refpeak,np.max(np.abs(bv),axis=0))
        denom=float(np.sum(bv*bv))
        per_frame.append(float(np.sqrt(np.sum(d*d)/denom)) if denom else 0.)
    relative=np.sqrt(sums/refs); normalized_peak=peak/refpeak
    return {'relative_l2_components':relative.tolist(),'relative_l2':float(np.sqrt(sums.sum()/refs.sum())),
            'max_abs_m_s':peak.tolist(),'max_abs_over_reference_peak':normalized_peak.tolist(),
            'per_sample_relative_l2':per_frame,'bitwise_equal':bool(np.array_equal(a,b)),
            'within_tolerance':bool(np.all(relative<1e-8) and np.all(normalized_peak<1e-7))}


def analyze() -> None:
    comparisons=[];timings=[]
    for freq in (15,30):
        records={name:[read_json(OUT/f'{name}_{freq}_{i}'/f'1400x700_two_layer_{freq}Hz/result.json') for i in range(3)] for name in ('tenzor','seismic')}
        for name,rows in records.items():
            timings.append({'frequency_hz':freq,'backend':name,'run_seconds':[number(r['run_seconds']) for r in rows],
                            'median_run_seconds':float(np.median([number(r['run_seconds']) for r in rows])),
                            'prepare_seconds':[number(r['prepare_seconds']) for r in rows],
                            'total_seconds':[number(r['total_seconds']) for r in rows]})
        for i in range(3):
            old=OUT/f'tenzor_{freq}_{i}'/f'1400x700_two_layer_{freq}Hz'
            new=OUT/f'seismic_{freq}_{i}'/f'1400x700_two_layer_{freq}Hz'
            om=records['tenzor'][i];nm=records['seismic'][i]
            assert om['dt_s']==nm['dt_s'] and om['steps']==nm['steps']
            assert om['mesh']==nm['mesh']
            np.testing.assert_array_equal(np.load(old/'nodes.npy'),np.load(new/'nodes.npy'))
            field=difference(np.load(new/'velocity.npy',mmap_mode='r'),np.load(old/'velocity.npy',mmap_mode='r'))
            with np.load(new/'surface.npz') as a,np.load(old/'surface.npz') as b:
                np.testing.assert_array_equal(a['time_s'],b['time_s']);np.testing.assert_array_equal(a['coordinates_m'],b['coordinates_m'])
                surface=difference(a['velocity_m_s'],b['velocity_m_s'])
            historical=ROOT/'runs'/f'1400x700_two_layer_{freq}Hz'
            with np.load(old/'surface.npz') as a,np.load(historical/'surface.npz') as b:
                historical_surface=difference(a['velocity_m_s'],b['velocity_m_s'])
            comparisons.append({'frequency_hz':freq,'repeat':i,'dt_s':om['dt_s'],'steps':om['steps'],
                                'fields':field,'surface':surface,'old_vs_historical_surface':historical_surface})
    write(OUT/'analysis.json',{'status':'completed','timings':timings,'comparisons':comparisons,
          'tolerance':'component relative L2 <1e-8 and max absolute difference/reference peak <1e-7',
          'method':'same GPU, separate processes, two full warmups excluded, three alternating repeats per solver/frequency; run includes receiver transfer and recording',
          'all_agree':all(mapping(c[k])['within_tolerance'] for c in comparisons for k in ('fields','surface'))})


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['tenzor','seismic']);p.add_argument('--frequency',type=int,default=15);p.add_argument('--label');p.add_argument('--dt',type=float);p.add_argument('--profile',action='store_true');p.add_argument('--profile-root',type=Path);p.add_argument('--analyze',action='store_true');args=p.parse_args()
    if args.analyze: analyze();return
    if args.backend:
        assert args.label
        execute(Spec(1400,700,'two_layer',args.frequency),output_root=(args.profile_root or OUT)/args.label,backend=args.backend,explicit_dt=args.dt,profile_markers=args.profile)
        return
    space();OUT.mkdir(exist_ok=False)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def launch(backend: str,freq: int,label: str,dt: float | None = None) -> None:
        subprocess.run([sys.executable,'-m',__spec__.name if __spec__ else 'scripts.dev_1_5_geometry_frequency.compare_solvers','--backend',backend,'--frequency',str(freq),'--label',label]+([] if dt is None else ['--dt',str(dt)]),env=env,check=True)
    for backend in ('tenzor','seismic'):launch(backend,15,f'warmup_{backend}')
    native_dts=[number(read_json(OUT/f'warmup_{b}'/'1400x700_two_layer_15Hz/result.json')['dt_s']) for b in ('tenzor','seismic')]
    common_dt=min(native_dts) if native_dts[0]!=native_dts[1] else None
    write(OUT/'plan.json',{'native_dt_s':native_dts,'explicit_common_dt_s':common_dt,'frequencies_hz':[15,30],'repeats':3,'warmup_runs':2})
    for freq in (15,30):
        for i in range(3):
            for backend in (('tenzor','seismic') if i%2==0 else ('seismic','tenzor')):launch(backend,freq,f'{backend}_{freq}_{i}',common_dt)
    analyze()


if __name__=='__main__':main()
