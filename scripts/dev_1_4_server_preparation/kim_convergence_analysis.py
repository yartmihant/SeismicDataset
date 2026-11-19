"""Compare saved Kim convergence traces; no solver imports or new computations of waves."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import numpy as np
from numpy.typing import NDArray
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json, number, mapping
os.environ.setdefault('MPLCONFIGDIR',str(PROJECT_ROOT/'.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class Record:
    values: NDArray[np.float64]
    x: NDArray[np.float64]
    times: NDArray[np.float64]
    meta: dict[str,object]


def load(path: Path) -> Record:
    meta=read_json(path/'result.json')
    assert meta['status']=='completed'
    file=path/'surface.npz'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==mapping(meta['surface'])['sha256']
    with np.load(file) as z:
        values=z['velocity_m_s'];x=z['coordinates_m'][:,0];times=z['time_s']
    assert values.shape==(1001,141,2) and np.isfinite(values).all()
    return Record(values,x,times,meta)


def metrics(a: Record,b: Record) -> dict[str,object]:
    assert np.allclose(a.x,b.x,atol=1e-8,rtol=0) and np.array_equal(a.times,b.times)
    rows:dict[str,object]={}
    for maskname,mask in [('all141',np.ones(141,dtype=bool)),('far20',np.abs(a.x-350)>=20-1e-8),('dataset70',np.isclose(np.mod(a.x+1e-8,10),5,atol=1e-6)),('dataset70_far20',np.isclose(np.mod(a.x+1e-8,10),5,atol=1e-6)&(np.abs(a.x-350)>=20-1e-8))]:
        for window,lo,hi in [('full',0.,1.),('early',0.,.5),('late',.5,1.)]:
            selected=(a.times>=lo)&(a.times<=hi)
            av=a.values[selected][:,mask];bv=b.values[selected][:,mask];d=av-bv
            trace_relative=np.linalg.norm(d,axis=(0,2))/np.maximum(np.linalg.norm(bv,axis=(0,2)),1e-300)
            rows[f'{maskname}/{window}']={'max_trace_relative_l2':float(trace_relative.max()),'max_trace_x_m':float(a.x[mask][np.argmax(trace_relative)]),'p95_trace_relative_l2':float(np.quantile(trace_relative,.95)),'relative_l2':float(np.linalg.norm(d))/max(float(np.linalg.norm(bv)),1e-300),
                'component_relative_l2':[float(np.linalg.norm(d[:,:,i]))/max(float(np.linalg.norm(bv[:,:,i])),1e-300) for i in (0,1)],
                'max_abs_difference':float(np.max(np.abs(d))),
                'max_difference_over_reference_peak':float(np.max(np.abs(d)))/max(float(np.max(np.abs(bv))),1e-300),
                'reference_l2':float(np.linalg.norm(bv)), 'receivers':int(mask.sum())}
    return rows


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    result:dict[str,object]={'scope':'Relative differences, not exact error; p5 is a comparison reference, not a proven converged solution','models':{}}
    models:dict[str,object]={};fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for row,model in enumerate((12404,65)):
        base=args.root/f'{model:05d}'
        records={(order,label):load(base/f'p{order}_beta{beta:.8f}_cfl{safety:g}') for order in (3,4,5) for label,beta,safety in [('zero',0.,.8),('main',1/27,.8),('half',1/27,.4)]}
        pairs:dict[str,object]={}
        for label in ('zero','main','half'):
            for order in (3,4):pairs[f'p{order}-p5/{label}']=metrics(records[order,label],records[5,label])
            pairs[f'p3-p4/{label}']=metrics(records[3,label],records[4,label])
        for order in (3,4,5):
            pairs[f'dt/p{order}']=metrics(records[order,'main'],records[order,'half'])
            pairs[f'beta/p{order}']=metrics(records[order,'main'],records[order,'zero'])
        common_k=int(number(records[5,'half'].meta['steps_per_ms']))
        if (base/f'p3_beta0.00000000_fixedk{common_k}'/'result.json').exists():
            for order in (3,4,5):
                for label,beta in [('common_zero',0.),('common_beta',1/27)]:
                    records[order,label]=records[5,'half'] if order==5 and beta else load(base/f'p{order}_beta{beta:.8f}_fixedk{common_k}')
            for label in ('common_zero','common_beta'):
                for order in (3,4): pairs[f'p{order}-p5/{label}']=metrics(records[order,label],records[5,label])
                pairs[f'p3-p4/{label}']=metrics(records[3,label],records[4,label])
            for order in (3,4,5):pairs[f'beta_common/p{order}']=metrics(records[order,'common_beta'],records[order,'common_zero'])
        models[str(model)]={'pairs':pairs,'runs':{f'p{k[0]}/{k[1]}':v.meta for k,v in records.items()}}
        for col,window in enumerate(('full','late')):
            ax=axes[row,col]
            for label,title,color in [('zero','β=0, CFL 0,8','#666666'),('main','β=1/27, CFL 0,8','#1971c2'),('half','β=1/27, CFL 0,4','#e8590c')]:
                vals=[100*number(mapping(mapping(pairs[f'p{o}-p5/{label}'])[f'dataset70_far20/{window}'])['relative_l2']) for o in (3,4)]
                ax.plot([3,4],vals,'o-',label=title,color=color)
            ax.set(xlabel='Порядок p',ylabel='Отличие от p5, % L2',xticks=[3,4],title=f'ECVB {model:05d}: '+('0–1 с' if window=='full' else '0,5–1 с'))
            ax.grid(alpha=.25);ax.legend(fontsize=8)
        ax=axes[row,2]
        vals=[100*number(mapping(mapping(pairs[f'dt/p{o}'])['dataset70_far20/full'])['relative_l2']) for o in (3,4,5)]
        ax.plot([3,4,5],vals,'o-',color='#2b8a3e');ax.set(xlabel='Порядок p',ylabel='Разность CFL 0,8 / 0,4, % L2',xticks=[3,4,5],title=f'ECVB {model:05d}: контроль dt');ax.grid(alpha=.25)
    if all('p3-p5/common_beta' in mapping(mapping(v)['pairs']) for v in models.values()):
        fig2,axs=plt.subplots(2,2,figsize=(11,8),layout='constrained')
        for row,model in enumerate((12404,65)):
            pairs=mapping(mapping(models[str(model)])['pairs'])
            for col,window in enumerate(('full','late')):
                ax=axs[row,col]
                for label,title,color in [('common_zero','β=0','#666666'),('common_beta','β=1/27','#1971c2')]:
                    for mask,style,suffix in [('dataset70','-','70 приёмников'),('dataset70_far20','--','вне 20 м')]:
                        vals=[100*number(mapping(mapping(pairs[f'p{o}-p5/{label}'])[f'{mask}/{window}'])['relative_l2']) for o in (3,4)]
                        ax.plot([3,4],vals,'o'+style,color=color,label=f'{title}, {suffix}')
                ax.set(xticks=[3,4],xlabel='Порядок p',ylabel='Отличие от p5, % L2',title=f'ECVB {model:05d}: '+('0–1 с' if window=='full' else '0,5–1 с'))
                ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig2.suptitle('Общий малый dt для всех p и обоих β внутри каждой модели')
        fig2.savefig(out/'common_dt.png',dpi=160)
    result['models']=models
    result['mask_note']='Dataset receivers x=5,15,...695, excluding |x-350|<20m; full and source-local metrics also in JSON'
    fig.suptitle('Сходимость Vx/Vy: сглаженные модели, элементы около 5 м\nПоказаны 66 из 70 приёмников, вне 20 м от точечного источника')
    fig.savefig(out/'convergence.png',dpi=160)
    (out/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('analysis saved',out,flush=True)


if __name__=='__main__':
    main()
