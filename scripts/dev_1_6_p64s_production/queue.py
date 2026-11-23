"""Explicit first EFVB chunk only: shared disk reservations and fail-stop workers."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import numpy as np
from scripts.dev_1_4_server_preparation.production_plan import DiskGuard,GuardConfig,GIB,build_chunks
from scripts.dev_1_4_server_preparation.production_worker import directory_bytes,progress_counts
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_6_p64s_production.run import execute,SOURCES,RECEIVERS,SNAPSHOT
from scripts.dev_1_3_point_source.run import file_record
from scripts.reporting import read_json,number
ROOT=Path('data/dev_1_6_p64s_production/chunk01_v1')
CONFIG=GuardConfig(model_budget_bytes=10*1024**2,execution_chunk_limit=1)


def prepare(root:Path)->None:
    assert read_json(Path('data/dev_1_6_p64s_production/comparison_v1/analysis/metrics.json'))['status']=='completed'
    assert build_chunks()[0].models==3000 and build_chunks()[0].last_id==2999
    root.mkdir(parents=True,exist_ok=False)
    guard=DiskGuard(root/'queue.sqlite',root,CONFIG);free=int(number(guard.status()['available_bytes']))
    assert free>=3000*CONFIG.model_budget_bytes+CONFIG.reserve_bytes+CONFIG.working_floor_bytes,'Insufficient full-chunk budget'
    write(root/'plan.json',{'status':'ready_not_started','dataset':'EFVB','first_id':0,'last_id':2999,'models':3000,'sources_x_m':list(SOURCES),'receivers_x_m':RECEIVERS.tolist(),'shape':[1001,4,140,2],'dtype':'float64','dt':'native CFL0.8 individually per model; tact .001s','raw_velocity_bytes':3000*1001*4*140*2*8,'budget':asdict(CONFIG),'available_bytes':free,'snapshot':file_record(SNAPSHOT),'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'production':'only this first chunk; no automatic cleanup or next chunk','symmetry':'900/1100/1300 not calculated or stored; mirror first three, reverse receivers, negate Vx when needed'})


def worker(root:Path,gpu:int)->None:
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpu)
    guard=DiskGuard(root/'queue.sqlite',root,CONFIG)
    while True:
        job=guard.reserve_next(gpu)
        if job is None:break
        assert job.dataset=='EFVB' and job.chunk_number==1 and 0<=job.model_id<3000
        out=root/'models'/f'{job.model_id:05d}'
        def before_write(size:int)->None:
            _,failed=progress_counts(guard.status())
            if failed:raise RuntimeError('Other GPU failed; stop before writing')
            guard.check_before_write(job.job_id,size)
        try:
            before_write(100000);execute(job.model_id,'p64s',out,before_write=before_write)
            data=np.load(out/'velocity.npy',mmap_mode='r');assert data.shape==(1001,4,140,2) and np.isfinite(data).all()
            record=read_json(out/'result.json');assert record['status']=='completed'
            guard.complete(job.job_id,directory_bytes(out))
        except BaseException:
            reason=traceback.format_exc();guard.fail(job.job_id,reason);raise
    write(root/f'worker{gpu}_finished.json',{'gpu':gpu,'status':guard.status()})


def launch(root:Path)->None:
    assert (root/'plan.json').exists() and not (root/'launcher.json').exists()
    guard=DiskGuard(root/'queue.sqlite',root,CONFIG);assert not guard.status()['progress']
    (root/'logs').mkdir(exist_ok=False)
    workers=[]
    for gpu in (0,1):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(Path('.runtime_ch16_p64s_v1/tmp').resolve()),XDG_CACHE_HOME=str(Path('.runtime_ch16_p64s_v1/cache').resolve()),MPLCONFIGDIR=str(Path('.runtime_ch16_p64s_v1/cache').resolve()))
        with (root/'logs'/f'gpu{gpu}.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-m','scripts.dev_1_6_p64s_production.queue','--root',str(root),'--worker',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        workers.append({'gpu':gpu,'pid':child.pid})
    write(root/'launcher.json',{'workers':workers,'only_first_chunk':True})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--status',action='store_true');a=p.parse_args()
    if a.prepare:prepare(a.root)
    elif a.launch:launch(a.root)
    elif a.status:print(json.dumps(DiskGuard(a.root/'queue.sqlite',a.root,CONFIG).status(),indent=2))
    else:assert a.worker is not None;worker(a.root,a.worker)
