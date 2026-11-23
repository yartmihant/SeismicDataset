"""Warmup, rotated timings, full-state equivalence and mirror-shot validation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_6_p64s_production.run import SNAPSHOT
from scripts.reporting import read_json,mapping,number,sequence
ROOT=Path('data/dev_1_6_p64s_production/comparison_v1')
VARIANTS=('p6','p64s','p67s')


def series(root:Path)->None:
    root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',{'models':[1,0,2999],'variants':list(VARIANTS),'benchmark_model':1,'warmups':1,'rotated_repeats':3,'shots':'p6 four sequential; p64s four simultaneous; p67s seven simultaneous; sources100:200:1300','receiver':'140 points x5:10:1395, Vx/Vy,1001 times,FP64; async buffer4; no frames','snapshot':file_record(SNAPSHOT),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu','--format=csv'],text=True)})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def launch(model:int,variant:str,label:str,state:bool=False)->None:
        out=root/f'{model:05d}'/variant/label;cmd=[sys.executable,'-m','scripts.dev_1_6_p64s_production.run','--model',str(model),'--variant',variant,'--out',str(out)]
        if state:cmd.append('--state')
        with (root/f'{model:05d}_{variant}_{label}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for v in VARIANTS:launch(1,v,'warmup',True)
    for repeat in range(3):
        for v in VARIANTS[repeat:]+VARIANTS[:repeat]:launch(1,v,f'repeat{repeat}')
    for model in (0,2999):
        for v in VARIANTS:launch(model,v,'check')
    write(root/'completed.json',{'status':'completed','calls':18,'single_source_integrations':24,'p64s_integrations':6,'p67s_integrations':6})


def metric(a:FloatArray,b:FloatArray)->dict[str,object]:
    delta=a-b;rel=float(np.linalg.norm(delta.ravel())/max(float(np.linalg.norm(b.ravel())),1e-300));peak=float(np.max(abs(delta))/max(float(np.max(abs(b))),1e-300))
    return {'relative_l2':rel,'peak_relative':peak,'max_abs':float(np.max(abs(delta))),'pass':rel<1e-8 and peak<1e-7}


def analyze(root:Path)->None:
    rows:list[dict[str,object]]=[];checks:list[dict[str,object]]=[];files=[]
    for rp in sorted(root.glob('*/*/*/result.json')):
        r=read_json(rp);assert r['status']=='completed';rows.append(r)
        for name in ('surface','state','pml_history'):
            if name in r:
                saved=mapping(r[name]);path=rp.parent/Path(str(saved['path'])).name;actual=file_record(path);assert actual['sha256']==saved['sha256'];files.append(actual)
    for model,labels in ((1,('warmup','repeat0','repeat1','repeat2')),(0,('check',)),(2999,('check',))):
        for label in labels:
            paths={v:root/f'{model:05d}'/v/label for v in VARIANTS};arrays={v:np.load(p/'velocity.npy') for v,p in paths.items()}
            records={v:read_json(p/'result.json') for v,p in paths.items()};steps={number(mapping(sequence(r['solver_records'])[0])['dt_s']) for r in records.values()};assert len(steps)==1
            for shot in range(4):
                for other in ('p6','p67s'):
                    for component in range(2):checks.append({'model':model,'label':label,'kind':'surface','shot':shot,'component':component,'reference':other,**metric(arrays['p64s'][:,shot,:,component],arrays[other][:,shot,:,component])})
            for shot in range(4):
                mirror=arrays['p64s'][:,shot,::-1].copy();mirror[:,:,0]*=-1
                checks.append({'model':model,'label':label,'kind':'mirror','shot':shot,'reference_shot':6-shot,**metric(mirror,arrays['p67s'][:,6-shot])})
            if label=='warmup':
                for filename in ('state.npy','pml_history.npy'):
                    a=np.load(paths['p64s']/filename,mmap_mode='r')
                    for other in ('p6','p67s'):
                        b=np.load(paths[other]/filename,mmap_mode='r')[:len(a)]
                        for row in range(len(a)):checks.append({'model':model,'kind':filename,'row':row,'reference':other,**metric(a[row],b[row])})
    timings={v:{key:float(np.median([number(read_json(root/'00001'/v/f'repeat{i}/result.json')[key]) for i in range(3)])) for key in ('run_seconds','total_seconds','prepare_seconds')} for v in VARIANTS}
    agree=all(c['pass'] for c in checks);out=root/'analysis';out.mkdir(exist_ok=False)
    write(out/'metrics.json',{'status':'completed' if agree else 'failed','checks':checks,'timings_median3':timings,'array_hashes':files,'rows':rows,'note':'p6 sums four sequential sources; p64s four; p67s seven with same first four. Not a seven-shot throughput comparison at equal work.'})
    assert agree,'Solver or mirror comparison failed'
    print(timings,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--analyze',action='store_true');a=p.parse_args();analyze(a.root) if a.analyze else series(a.root)
