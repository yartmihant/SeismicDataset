"""Summarize Nsight SQLite events, separating NVTX prepare/run intervals."""
from __future__ import annotations
import argparse
import csv
from dataclasses import dataclass,field
from pathlib import Path
import re
import sqlite3
import numpy as np
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_3_point_source.run import file_record


@dataclass
class KernelSamples:
    name: str
    durations_ns: list[int]=field(default_factory=list)


def union_ns(intervals: list[tuple[int,int]]) -> int:
    total=0; end=0
    for a,b in sorted(intervals):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total


def summarize(root: Path) -> None:
    database=root/'trace.sqlite'
    con=sqlite3.connect(f'file:{database.resolve()}?mode=ro',uri=True)
    assert con.execute('pragma quick_check').fetchone()[0]=='ok'
    strings={int(i):str(v) for i,v in con.execute('select id,value from StringIds')}
    ranges={str(text or strings.get(tid)): (int(a),int(b)) for a,b,text,tid in con.execute('select start,end,text,textId from NVTX_EVENTS where end is not null')}
    assert 'semgpu_prepare' in ranges and 'semgpu_run' in ranges
    copy_names={int(i):str(label) for i,label in con.execute('select id,label from ENUM_CUDA_MEMCPY_OPER')}
    stages:dict[str,object]={}
    csv_rows:list[dict[str,object]]=[]
    for name,(begin,end) in {'all_events':(0,2**63-1),**ranges}.items():
        groups:dict[str,KernelSamples]={};intervals=[]
        for a,b,sid in con.execute('select start,end,demangledName from CUPTI_ACTIVITY_KIND_KERNEL where start>=? and end<=?',(begin,end)):
            kernel=strings[int(sid)];groups.setdefault(kernel,KernelSamples(kernel)).durations_ns.append(int(b)-int(a));intervals.append((int(a),int(b)))
        total_ns=sum(sum(group.durations_ns) for group in groups.values())
        rows=[]
        for group in sorted(groups.values(),key=lambda g:sum(g.durations_ns),reverse=True):
            values=np.array(group.durations_ns,dtype=np.int64);match=re.search(r'(\w+_kernel)',group.name)
            row:dict[str,object]={'name':match.group(1) if match else group.name,'full_name':group.name,'calls':len(values),
                'total_seconds':int(values.sum())/1e9,'percent_kernel_time':float(values.sum()/total_ns*100),
                'mean_us':float(values.mean()/1000),'median_us':float(np.median(values)/1000),
                'p95_us':float(np.quantile(values,.95)/1000),'min_us':int(values.min())/1000,'max_us':int(values.max())/1000}
            rows.append(row);csv_rows.append({'stage':name,**row})
        memrows=[]
        for kind,total,count,size in con.execute('select copyKind,sum(end-start),count(*),sum(bytes) from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and end<=? group by copyKind',(begin,end)):
            memrows.append({'kind':copy_names[int(kind)],'total_seconds':int(total)/1e9,'calls':int(count),'bytes':int(size)})
        memset=con.execute('select coalesce(sum(end-start),0),count(*),coalesce(sum(bytes),0) from CUPTI_ACTIVITY_KIND_MEMSET where start>=? and end<=?',(begin,end)).fetchone()
        for table in ('CUPTI_ACTIVITY_KIND_MEMCPY','CUPTI_ACTIVITY_KIND_MEMSET'):
            intervals.extend((int(a),int(b)) for a,b in con.execute(f'select start,end from {table} where start>=? and end<=?',(begin,end)))
        api=[{'name':strings[int(sid)],'total_host_seconds':int(total)/1e9,'calls':int(count),'mean_us':float(avg)/1000} for sid,total,count,avg in con.execute('select nameId,sum(end-start),count(*),avg(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME where start>=? and end<=? group by nameId order by sum(end-start) desc',(begin,end))]
        busy=union_ns(intervals)/1e9;wall=(end-begin)/1e9 if name!='all_events' else None
        stages[name]={'nvtx_start_ns':begin if wall else None,'nvtx_end_ns':end if wall else None,'wall_seconds':wall,
                      'kernel_total_seconds':total_ns/1e9,'kernel_calls':sum(len(g.durations_ns) for g in groups.values()),'kernels':rows,
                      'memcpy':memrows,'memset':{'total_seconds':int(memset[0])/1e9,'calls':int(memset[1]),'bytes':int(memset[2])},
                      'gpu_active_union_seconds':busy,'no_captured_gpu_activity_seconds':wall-busy if wall else None,'cuda_runtime_api':api}
        if name=='semgpu_run':
            for kernel in ['material_response_kernel','predict_nodal_state_kernel','correct_nodal_state_kernel']:
                assert sum(len(g.durations_ns) for g in groups.values() if kernel in g.name)==52000
    diagnostics=[{'severity':int(level),'text':str(text)} for level,text in con.execute('select severity,text from DIAGNOSTIC_EVENT')]
    con.close()
    write(root/'profile_summary.json',{'status':'completed','sqlite':file_record(database),'report':file_record(root/'trace.nsys-rep'),'stages':stages,'diagnostics':diagnostics,
          'method':'CUDA event durations in NVTX wall intervals across all threads; fractions relative to summed kernel duration; host API and GPU durations overlap and must not be added; GPU union merges kernels/memcpy/memset intervals',
          'trace_count_check':'52000 material/predict/correct calls in run; no CPU sampling or hardware counter replay'})
    with (root/'kernels.csv').open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    print(root,'summary completed')


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args();summarize(args.root)


if __name__=='__main__':main()
