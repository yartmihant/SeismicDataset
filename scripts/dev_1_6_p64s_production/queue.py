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
import shutil
import hashlib
import tarfile
from time import perf_counter
import numpy as np
from scripts.dev_1_4_server_preparation.production_plan import DiskGuard,GuardConfig,GIB,build_chunks
from scripts.dev_1_4_server_preparation.production_worker import directory_bytes,progress_counts
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_6_p64s_production.run import execute,SOURCES,RECEIVERS,SNAPSHOT
from scripts.dev_1_3_point_source.run import file_record
from scripts.reporting import read_json,number,sequence,mapping
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
    assert read_json(Path('data/dev_1_6_p64s_production/dual_gpu_pilot_v1/completed.json'))['status']=='completed'
    assert read_json(CLEANUP/'completed.json')['status']=='completed'
    guard=DiskGuard(root/'queue.sqlite',root,CONFIG);assert not guard.status()['progress']
    (root/'logs').mkdir(exist_ok=False)
    workers=[]
    for gpu in (0,1):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',TMPDIR=str(Path('.runtime_ch16_p64s_v1/tmp').resolve()),XDG_CACHE_HOME=str(Path('.runtime_ch16_p64s_v1/cache').resolve()),MPLCONFIGDIR=str(Path('.runtime_ch16_p64s_v1/cache').resolve()))
        with (root/'logs'/f'gpu{gpu}.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-m','scripts.dev_1_6_p64s_production.queue','--root',str(root),'--worker',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        workers.append({'gpu':gpu,'pid':child.pid})
    write(root/'launcher.json',{'workers':workers,'only_first_chunk':True})

def pilot_worker(root:Path,gpu:int)->None:
    begin=perf_counter();rss=[]
    for model in range(gpu,6,2):
        execute(model,'p64s',root/f'{model:05d}')
        rss.append(next(line.strip() for line in Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:')))
    write(root/f'gpu{gpu}.json',{'status':'completed','gpu':gpu,'models':list(range(gpu,6,2)),'wall_seconds':perf_counter()-begin,'rss_after_each_model':rss})


def pilot(root:Path)->None:
    root.mkdir(parents=True,exist_ok=False);workers=[];begin=perf_counter()
    for gpu in (0,1):
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (root/f'gpu{gpu}.log').open('xb') as log:
            workers.append(subprocess.Popen([sys.executable,'-m','scripts.dev_1_6_p64s_production.queue','--root',str(root),'--pilot-worker',str(gpu)],env=env,stdout=log,stderr=subprocess.STDOUT))
    codes=[child.wait() for child in workers];assert codes==[0,0],codes
    write(root/'completed.json',{'status':'completed','models':6,'wall_seconds':perf_counter()-begin,'gpu_workers':[read_json(root/f'gpu{gpu}.json') for gpu in (0,1)]})


CLEANUP=Path('data/dev_1_6_p64s_production/cleanup_v1')
OLD_QUEUES=tuple(Path('data/dev_1_4_server_preparation')/name for name in ('efvb_chunk01_saturn_v3','efvb_chunk01_saturn_v4','production_smoke_saturn_v3'))


def cleanup_plan()->None:
    CLEANUP.mkdir(parents=True,exist_ok=False);paths:set[Path]=set()
    old=Path('data/dev_1_5_geometry_frequency')
    for folder in old.iterdir():
        if not folder.is_dir() or folder.name=='series_v1':continue
        for path in folder.rglob('*'):
            if path.is_file() and (path.suffix in ('.npy','.npz','.nsys-rep') or (path.suffix=='.sqlite' and path.name.startswith('trace'))):paths.add(path)
    for folder in OLD_QUEUES:
        if folder.exists():paths.update(p for p in folder.rglob('*') if p.is_file())
    entries=[]
    for path in sorted(paths):
        assert not path.is_symlink() and path.resolve().is_relative_to(Path.cwd())
        st=path.stat();entries.append({'path':str(path),'size':st.st_size,'blocks_bytes':st.st_blocks*512,'inode':st.st_ino,'mtime_ns':st.st_mtime_ns})
    write(CLEANUP/'plan.json',{'status':'prepared_not_deleted','entries':entries,'allocated_bytes':sum(int(number(e['blocks_bytes'])) for e in entries),'apparent_bytes':sum(int(number(e['size'])) for e in entries),'available_before_bytes':shutil.disk_usage('.').free,'preserved':'src,Git,code,JSON and galleries; original30-case series1.5; all1.6 fields; current comparison and new queue; installed runtimes'})


def cleanup_apply()->None:
    assert read_json(Path('data/dev_1_6_p64s_production/comparison_v1/analysis/metrics.json'))['status']=='completed'
    assert not (CLEANUP/'completed.json').exists()
    processes=subprocess.check_output(['ps','-u',str(os.getuid()),'-o','args='],text=True)
    assert not any(str(folder) in processes for folder in OLD_QUEUES),'Old queue still active'
    plan=read_json(CLEANUP/'plan.json');entries=[mapping(e) for e in sequence(plan['entries'])]
    # Prevalidate every path before any deletion. No wildcard deletion and no source files.
    for entry in entries:
        path=Path(str(entry['path']));assert not path.is_symlink() and path.resolve().is_relative_to(Path.cwd())
        assert path.is_relative_to('data/dev_1_5_geometry_frequency') or any(path.is_relative_to(p) for p in OLD_QUEUES)
        st=path.stat();assert (st.st_size,st.st_ino,st.st_mtime_ns)==(entry['size'],entry['inode'],entry['mtime_ns'])
    archive=CLEANUP/'old_queue_metadata.tar.gz'
    with tarfile.open(archive,'x:gz') as stream:
        for folder in OLD_QUEUES:
            if folder.exists():
                for path in sorted(folder.rglob('*')):
                    if path.is_file() and path.suffix not in ('.npy','.npz','.msh'):stream.add(path,arcname=str(path),recursive=False)
    with tarfile.open(archive,'r:gz') as stream:
        for member in stream.getmembers():
            if member.isfile():
                extracted=stream.extractfile(member);assert extracted is not None
                assert hashlib.sha256(extracted.read()).hexdigest()==hashlib.sha256(Path(member.name).read_bytes()).hexdigest()
    # Metadata is retained and verified before the old queues are removed.
    for entry in entries:Path(str(entry['path'])).unlink()
    for folder in OLD_QUEUES:
        if folder.exists():
            for path in sorted(folder.rglob('*'),key=lambda p:len(p.parts),reverse=True):
                if path.is_dir():path.rmdir()
            folder.rmdir()
    write(CLEANUP/'completed.json',{'status':'completed','files_deleted':len(entries),'planned_allocated_bytes':plan['allocated_bytes'],'available_before_bytes':plan['available_before_bytes'],'available_after_bytes':shutil.disk_usage('.').free,'old_queue_metadata':file_record(archive),'preserved':plan['preserved']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--prepare',action='store_true');p.add_argument('--launch',action='store_true');p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--status',action='store_true');p.add_argument('--cleanup-plan',action='store_true');p.add_argument('--cleanup-apply',action='store_true');p.add_argument('--pilot',action='store_true');p.add_argument('--pilot-worker',type=int,choices=(0,1));a=p.parse_args()
    if a.pilot:pilot(a.root)
    elif a.pilot_worker is not None:pilot_worker(a.root,a.pilot_worker)
    elif a.cleanup_plan:cleanup_plan()
    elif a.cleanup_apply:cleanup_apply()
    elif a.prepare:prepare(a.root)
    elif a.launch:launch(a.root)
    elif a.status:print(json.dumps(DiskGuard(a.root/'queue.sqlite',a.root,CONFIG).status(),indent=2))
    else:assert a.worker is not None;worker(a.root,a.worker)
