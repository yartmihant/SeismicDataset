"""Local report plot from compact Saturn comparison evidence."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json,sequence,mapping,number


def main() -> None:
    data=read_json(PROJECT_ROOT/'data/dev_1_5_geometry_frequency/solver_comparison_v1/analysis.json')
    out=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/solver_comparison_v1';out.mkdir(exist_ok=True,parents=True)
    rows=[mapping(r) for r in sequence(data['timings'])]
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    labels=[]
    for i,row in enumerate(rows):
        values=np.array([number(v) for v in sequence(row['run_seconds'])]);med=float(np.median(values))
        color='#315b86' if row['backend']=='tenzor' else '#bc5a34'
        ax.bar(i,med,color=color,width=.6)
        ax.scatter(i+np.linspace(-.1,.1,3),values,color='black',s=22,zorder=3)
        ax.text(i,float(values.max())+1,f'{med:.2f} с',ha='center')
        labels.append(f"{int(number(row['frequency_hz']))} Гц\n{row['backend']}")
    ax.set_xticks(range(len(rows)),labels);ax.set_ylabel('Время расчёта с регистрацией, с');ax.set_xlabel('Частота Рикера и решатель')
    ax.set_title('Saturn · A100 40 ГБ · двухслойная модель 1400×700 м\np5 / 5 м · 1 с · медиана трёх повторов; точки — повторы')
    ax.set_ylim(0,ax.get_ylim()[1]*1.15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(out/'performance.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for rowobj in sequence(data['comparisons']):
        row=mapping(rowobj)
        if row['repeat']!=0:continue
        for ax,key,label in zip(axes,['fields','surface'],['Все узлы поля','Все узлы верхней границы']):
            vals=np.array([number(v) for v in sequence(mapping(row[key])['per_sample_relative_l2'])])
            t=np.linspace(0,1,len(vals));ax.plot(t[1:],vals[1:],label=f"{int(number(row['frequency_hz']))} Гц")
            ax.set_title(label);ax.set_xlabel('Время, с');ax.set_ylabel('‖v новый − v старый‖₂ / ‖v старый‖₂');ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0));ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Разность решений на каждом сохранённом времени · первый парный повтор')
    fig.savefig(out/'agreement.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
