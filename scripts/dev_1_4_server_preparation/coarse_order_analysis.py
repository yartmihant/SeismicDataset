"""Compare p3..p5 refined with p6/p7 coarse on identical Q2 geometry."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.dev_1_4_server_preparation.kim_convergence_analysis import load,metrics
from scripts.reporting import mapping,number,read_json
from scripts.project_paths import PROJECT_ROOT
import matplotlib.pyplot as plt


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    images=PROJECT_ROOT/'img/dev_1_4_server_preparation/coarse_order_v1'/out.name;images.mkdir(parents=True,exist_ok=False)
    fig,axs=plt.subplots(2,3,figsize=(15,8),layout='constrained');models:dict[str,object]={}
    for row,m in enumerate((12404,65)):
        base=args.root/f'{m:05d}'
        native={o:load(base/('coarse' if o>=6 else 'refined')/f'p{o}_beta{1/27:.8f}_cfl0.8') for o in (3,4,5,6,7)}
        k=2*max(int(number(v.meta['steps_per_ms'])) for v in native.values())
        fixed={o:load(base/('coarse' if o>=6 else 'refined')/f'p{o}_beta{1/27:.8f}_fixedk{k}') for o in (3,4,5,6,7)}
        assert len({v.meta['dt_s'] for v in fixed.values()})==1
        pairs={f'p{o}/fixed':metrics(fixed[o],fixed[5]) for o in (3,4,6,7)}
        pairs.update({f'p{o}/native':metrics(native[o],native[5]) for o in (3,4,6,7)})
        pairs.update({f'p{o}/dt':metrics(native[o],fixed[o]) for o in (3,4,5,6,7)})
        old_k=61 if m==12404 else 129
        old=load(PROJECT_ROOT/f'data/dev_1_4_server_preparation/kim_convergence_v1/{m:05d}/p5_beta{1/27:.8f}_cfl0.4')
        geometry_change=metrics(fixed[5],old)
        models[str(m)]={'common_k':k,'pairs':pairs,'old_p5_reference_difference':geometry_change,
            'old_p5_dt_s':.001/old_k,'reference_note':'old mesh versus new exactly refined coarse mesh; difference includes geometry, discretization and unequal dt',
            'runs':{f'p{o}/{label}':v.meta for label,records in [('native',native),('fixed',fixed)] for o,v in records.items()},
            'geometry':{g:read_json(base/g/'geometry.json') for g in ('coarse','refined')}}
        orders=[3,4,6,7];labels=['p3 / 5м','p4 / 5м','p6 / 10м','p7 / 10м']
        for col,w in enumerate(('full','late')):
            ax=axs[row,col]
            for mask,title,style in [('dataset70','70 приёмников','o-'),('dataset70_far20','вне 20 м','s--')]:
                vals=[100*number(mapping(pairs[f'p{o}/fixed'][f'{mask}/{w}'])['relative_l2']) for o in orders]
                ax.plot(range(4),vals,style,label=title)
            ax.set(xticks=range(4),xticklabels=labels,ylabel='L2-разность с p5 / 5м, %',title=f'{m:05d}: '+('0–1 с' if w=='full' else '0,5–1 с'))
            ax.grid(alpha=.25);ax.legend(fontsize=8)
        ax=axs[row,2]
        all_orders=[3,4,5,6,7];times=[number(native[o].meta['total_seconds']) for o in all_orders]
        ax.bar(range(5),times,color=['#91a7ff','#748ffc','#5c7cfa','#20c997','#12b886'])
        ax.set(xticks=range(5),xticklabels=['p3/5','p4/5','p5/5','p6/10','p7/10'],ylabel='Время одного источника с подготовкой, с',title=f'{m:05d}: native CFL=0,8')
        for i,t in enumerate(times):ax.text(i,t,f'{t:.1f}',ha='center',va='bottom')
        ax.set_ylim(0,max(times)*1.15);ax.grid(axis='y',alpha=.2)
    fig.suptitle('β=1/27: p6/p7 на крупной сетке против p3/p4/p5 на её точном разбиении\nРазности — при общем малом dt; время — при индивидуальном native CFL=0,8')
    fig.savefig(images/'comparison.png',dpi=160)
    (out/'analysis.json').write_text(json.dumps({'models':models,'scope':'Two regenerated geometries, central source, 1s; p5 refined reference is not exact solution'},ensure_ascii=False,indent=2)+'\n')
    print(out,flush=True)


if __name__=='__main__':
    main()
