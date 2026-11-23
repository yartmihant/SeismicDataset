"""Validate outputs and summarize Nsight traces for the p67s receiver study."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import sqlite3
import numpy as np
from numpy.typing import NDArray
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.p67s_run import ROOT,MODES,REPEATS
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.reporting import read_json,number,mapping,sequence

FloatArray=NDArray[np.float64]


def union_ns(intervals:list[tuple[int,int]])->int:
    total=0;end=0
    for begin,finish in sorted(intervals):
        total+=max(0,finish-max(begin,end));end=max(end,finish)
    return total


def compare(a:FloatArray,b:FloatArray)->dict[str,object]:
    assert a.shape==b.shape and a.shape[-1]==2 and np.isfinite(a).all() and np.isfinite(b).all()
    delta=a-b
    axes=tuple(range(a.ndim-1));err=np.sum(delta*delta,axis=axes);ref=np.sum(b*b,axis=axes)
    peak=np.max(abs(delta),axis=axes);refpeak=np.max(abs(b),axis=axes)
    l2=np.sqrt(err/np.maximum(ref,1e-300));rp=peak/np.maximum(refpeak,1e-300)
    return {'relative_l2':l2.tolist(),'max_abs_m_s':peak.tolist(),'relative_peak':rp.tolist(),'passed':bool(np.all(l2<1e-8) and np.all(rp<1e-7))}


def analyze(root:Path)->None:
    assert (root/'runs_completed.json').exists();out=root/'analysis';out.mkdir(exist_ok=False)
    verified=[];rows=[];checks=[]
    for path in sorted(root.glob('*/*/result.json')):
        r=read_json(path);assert r['status']=='completed' and r['dt_s']==1/9000 and r['steps']==9000
        for name in ('surface.npz','final_top.npy','final_velocity.npy'):
            if name in r:
                actual_file=file_record(path.parent/name);assert actual_file['sha256']==mapping(r[name])['sha256'];verified.append(actual_file)
    def surface(mode:str,label:str)->FloatArray:
        with np.load(root/mode/label/'surface.npz') as z:
            np.testing.assert_array_equal(z['time_s'],np.arange(1001)*.001)
            values=np.asarray(z['velocity_m_s'],dtype=np.float64);assert values.shape==(1001,8827,2)
            return values
    labels=['warmup']+[f'repeat{i}' for i in range(REPEATS)]
    if (root/'old_sync/profile/result.json').exists():labels+=['profile']
    for label in labels:
        ref=surface('old_sync',label)
        for mode in MODES:
            top=np.load(root/mode/label/'final_top.npy');check=compare(top,ref[-1]);assert check['passed'],(mode,label,check)
            checks.append({'kind':'final_top','mode':mode,'label':label,**check})
            if mode in ('new_sync','new_async'):
                actual=surface(mode,label)
                for shot in range(7):
                    check=compare(actual[:,shot*1261:(shot+1)*1261],ref[:,shot*1261:(shot+1)*1261]);assert check['passed'],(mode,label,shot,check)
                    checks.append({'kind':'seismogram','mode':mode,'label':label,'source':shot,**check})
    ref_final=np.load(root/'old_sync/warmup/final_velocity.npy')
    for mode in MODES[1:]:
        actual=np.load(root/mode/'warmup/final_velocity.npy')
        for shot in range(7):
            check=compare(actual[shot],ref_final[shot]);assert check['passed'],(mode,shot,check)
            checks.append({'kind':'final_full_velocity','mode':mode,'source':shot,**check})
    for mode in MODES:
        runs=[read_json(root/mode/f'repeat{i}/result.json') for i in range(REPEATS)]
        fields=('run_seconds','solve_wall_seconds','assembly_seconds','prepare_seconds','callback_wall_seconds','callback_thread_cpu_seconds','output_npz_seconds')
        row:dict[str,object]={'mode':mode}
        for field in fields:
            samples=[number(r[field]) for r in runs];row[field]=samples;row['median_'+field]=float(np.median(samples))
        row['min_run_seconds']=min(number(r['run_seconds']) for r in runs);row['max_run_seconds']=max(number(r['run_seconds']) for r in runs)
        rows.append(row)
    med={str(r['mode']):number(r['median_run_seconds']) for r in rows}
    paired={mode:[number(read_json(root/mode/f'repeat{i}/result.json')['run_seconds'])-number(read_json(root/'new_none'/f'repeat{i}/result.json')['run_seconds']) for i in range(REPEATS)] for mode in MODES[2:]}
    write(out/'metrics.json',{'status':'completed','timings':rows,'checks':checks,'verified_files':verified,
        'new_vs_old_speedup':med['old_sync']/med['new_sync'],'sync_receiver_overhead_seconds':med['new_sync']-med['new_none'],
        'async_receiver_overhead_seconds':med['new_async']-med['new_none'],'paired_deltas_vs_none_seconds':paired,
        'max_relative_l2':max(max(number(v) for v in sequence(r['relative_l2'])) for r in checks),
        'scope':'No SEG-Y. Surface in RAM during run; NPZ separately after run. Profiler runs excluded from medians.'})


def profile_summary(path:Path)->dict[str,object]:
    con=sqlite3.connect(f'file:{(path/"trace.sqlite").resolve()}?mode=ro',uri=True)
    assert con.execute('pragma quick_check').fetchone()[0]=='ok'
    strings={int(i):str(v) for i,v in con.execute('select id,value from StringIds')}
    ranges={str(text or strings.get(tid)):(int(a),int(b)) for a,b,text,tid in con.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')}
    copy_names={int(i):str(label) for i,label in con.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')}
    stages={}
    for stage in ('semgpu_prepare','semgpu_run','output_npz'):
        begin,end=ranges[stage];groups:dict[str,list[int]]={};intervals=[]
        for a,b,sid in con.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(begin,end)):
            groups.setdefault(strings[int(sid)],[]).append(int(b)-int(a));intervals.append((int(a),int(b)))
        total=sum(sum(v) for v in groups.values());kernels=[]
        for name,durations in sorted(groups.items(),key=lambda p:sum(p[1]),reverse=True):
            match=re.search(r'(\w+_kernel(?:<[^>]+>)?)',name)
            kernels.append({'name':match.group(1) if match else name,'full_name':name,'calls':len(durations),
                'seconds':sum(durations)/1e9,'share_percent':100*sum(durations)/max(total,1),'mean_us':float(np.mean(durations))/1000})
        copies=[{'kind':copy_names[int(k)],'seconds':int(t)/1e9,'calls':int(n),'bytes':int(b)} for k,t,n,b in con.execute('select copyKind,sum(end-start),count(*),sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=? group by copyKind',(begin,end))]
        m=con.execute('select coalesce(sum(end-start),0),count(*),coalesce(sum(bytes),0) from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and end<=?',(begin,end)).fetchone()
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
            intervals.extend((int(a),int(b)) for a,b in con.execute(f'select start,end from {table} where start>=? and end<=?',(begin,end)))
        api=[{'name':strings[int(sid)],'host_seconds':int(t)/1e9,'calls':int(n)} for sid,t,n in con.execute('select nameId,sum(end-start),count(*) from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? group by nameId order by sum(end-start) desc',(begin,end))]
        wall=(end-begin)/1e9;busy=union_ns(intervals)/1e9
        stages[stage]={'wall_seconds':wall,'kernel_seconds':total/1e9,'kernel_calls':sum(len(v) for v in groups.values()),'kernels':kernels,
            'memcpy':copies,'memset_seconds':int(m[0])/1e9,'memset_calls':int(m[1]),'memset_bytes':int(m[2]),'gpu_active_union_seconds':busy,
            'no_captured_gpu_activity_seconds':wall-busy,'cuda_api':api}
        if stage=='semgpu_run':
            predict=sum(len(v) for k,v in groups.items() if 'predict_nodal_state_kernel' in k)
            assert predict==(9000 if path.name=='old_sync' else 1000),(path,predict)
            assert sum(len(v) for k,v in groups.items() if 'correct_nodal_state_kernel' in k)==9000
            assert sum(len(v) for k,v in groups.items() if 'velocity_frame_kernel' in k)==(0 if path.name=='new_none' else 1001)
    diagnostics=[{'severity':int(level),'text':str(t)} for level,t in con.execute('select severity,text from DIAGNOSTIC_EVENT')]
    con.close()
    result:dict[str,object]={'status':'completed','stages':stages,'diagnostics':diagnostics,'sqlite':file_record(path/'trace.sqlite'),'trace':file_record(path/'trace.nsys-rep'),
        'caution':'Kernel sums, GPU union and host CUDA API are overlapping timelines; do not add them. No-GPU-activity is not solely receiver cost.'}
    write(path/'profile_summary.json',result);return result


def profiles(root:Path)->None:
    assert (root/'profiles/completed.json').exists()
    for mode in MODES:profile_summary(root/'profiles'/mode)


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--profiles',action='store_true');a=p.parse_args()
    if a.profiles:profiles(a.root)
    else:analyze(a.root)

if __name__=='__main__':main()
