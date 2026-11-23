"""Saved Saturn performance for p4/p5 with sparse damping support."""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json,mapping,sequence,number
from scripts.dev_1_5_geometry_frequency.setup import write

ROOT=PROJECT_ROOT/'data/dev_1_5_geometry_frequency'
SERIES=ROOT/'p45_damping_saturn_v1'
OUT=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/p45_damping_v1'
CASE='1400x700_two_layer_30Hz'


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    sections=[];records:dict[str,object]={}
    for ax,order in zip(axes,(4,5)):
        root=SERIES/f'p{order}';analysis=read_json(root/'analysis.json');plan=read_json(root/'plan.json')
        assert analysis['status']=='completed' and analysis['all_agree'] is True
        variants=('tenzor','seismic',f'seismic_p{order}')
        names=('tenzor','seismic',f'специализированный p{order}')
        rows={str(mapping(r)['backend']):mapping(r) for r in sequence(analysis['timings'])}
        run=np.array([number(rows[b]['median_run_seconds']) for b in variants])
        lo=np.array([min(number(v) for v in sequence(rows[b]['run_seconds'])) for b in variants])
        hi=np.array([max(number(v) for v in sequence(rows[b]['run_seconds'])) for b in variants])
        ax.barh(names,run,color=['#b3bdc5','#688fa7','#286447'],xerr=np.array([run-lo,hi-run]),capsize=4)
        ax.invert_yaxis();ax.set_xlabel('run, с · медиана трёх повторов');ax.set_title(f'p{order} · A1000');ax.set_xlim(0,float(hi.max())*1.24);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for y,value in enumerate(run):ax.text(value+hi.max()*.025,y,f'{value:.3f}',va='center')
        meta=read_json(root/f'{variants[-1]}_0'/CASE/'result.json')
        max_l2=max(number(v) for c in sequence(analysis['comparisons']) for k in ('fields','surface') for v in sequence(mapping(mapping(c)[k])['relative_l2_components']))
        max_abs=max(number(v) for c in sequence(analysis['comparisons']) for k in ('fields','surface') for v in sequence(mapping(mapping(c)[k])['max_abs_m_s']))
        summary={'order':order,'median_run_seconds':dict(zip(variants,run.tolist())),'median_total_seconds':{b:rows[b]['median_total_seconds'] for b in variants},'specialized_speedup_vs_tenzor':float(run[0]/run[2]),'specialized_speedup_vs_seismic':float(run[1]/run[2]),'steps':meta['steps'],'dt_s':meta['dt_s'],'nodes':meta['field_nodes'],'receivers':meta['surface_nodes'],'damping_nodes':meta['damping_nodes'],'max_component_relative_l2':max_l2,'max_abs_difference_m_s':max_abs}
        records[f'p{order}']=summary
        table=''
        for b,label in zip(variants,names):
            row=rows[b];timings=', '.join(f'{number(v):.3f}' for v in sequence(row['run_seconds']))
            prep=float(np.median([number(v) for v in sequence(row['prepare_seconds'])]))
            table+=f'<tr><th>{label}</th><td>{timings}</td><td>{number(row["median_run_seconds"]):.3f}</td><td>{prep:.3f}</td><td>{number(row["median_total_seconds"]):.3f}</td></tr>'
        sections.append(f'<h2>Порядок {order}</h2><p>{meta["field_nodes"]} узлов, {meta["surface_nodes"]} приёмник(ов), {meta["damping_nodes"]} узлов демпфирования. Native CFL0,8: {meta["steps"]} шагов; dt={number(meta["dt_s"])*1e6:.6f} мкс. Native шаги вариантов: {html.escape(str(plan["native_dt_s"]))}. Принудительный общий dt: {plan["explicit_common_dt_s"]}.</p><p>Специализация быстрее tenzor в<b>{run[0]/run[2]:.3f} раза</b>, общего seismic — в<b>{run[1]/run[2]:.3f} раза</b>. Сокращение run: {(1-run[2]/run[0])*100:.2f}% / {(1-run[2]/run[1])*100:.2f}%.</p><table><tr><th>Решатель</th><th>Три run, с</th><th>Медиана run, с</th><th>Подготовка, с</th><th>Полное время, с</th></tr>{table}</table><p>Контроль всех сохранённых Vx/Vy относительно tenzor того же порядка и шага: максимальная компонентная L2={max_l2:.3e}, max|Δv|={max_abs:.3e} м/с.</p><p><a href="../../../data/dev_1_5_geometry_frequency/p45_damping_saturn_v1/p{order}/analysis.json">Полные метрики p{order}</a> · <a href="../../../data/dev_1_5_geometry_frequency/p45_damping_saturn_v1/p{order}/environment.json">Среда p{order}</a></p>')
    before=read_json(ROOT/'p5_saturn_v1/analysis.json');after=read_json(SERIES/'p5/analysis.json')
    old={str(mapping(r)['backend']):number(mapping(r)['median_run_seconds']) for r in sequence(before['timings'])}
    new={str(mapping(r)['backend']):number(mapping(r)['median_run_seconds']) for r in sequence(after['timings'])}
    historical=sequence(after['tenzor_before_after_damping_support']);assert len(historical)==3
    assert all(mapping(mapping(c)[k])['within_tolerance'] for c in historical for k in ('fields','surface'))
    hist_l2=max(number(v) for c in historical for k in ('fields','surface') for v in sequence(mapping(mapping(c)[k])['relative_l2_components']))
    records['p5_before_after']={'before_run_seconds':old,'after_run_seconds':new,'tenzor_max_component_relative_l2':hist_l2}
    historical_table=''.join(f'<tr><th>{b}</th><td>{old[b]:.3f}</td><td>{new[b]:.3f}</td><td>{(new[b]/old[b]-1)*100:+.2f}%</td></tr>' for b in old)
    sections.append(f'<h2>p5: сравнение с прежней серией</h2><table><tr><th>Решатель</th><th>Прежний run, с</th><th>Новый run, с</th><th>Изменение run</th></tr>{historical_table}</table><p>В исходной постановке демпфирующее поле включало все узлы, в том числе внутренние с нулевым коэффициентом. Теперь носитель — объединение узлов элементов трёх слоёв, общие узлы включаются один раз. Коэффициенты при продолжении нулём во внутреннюю область побитово совпадают с прежним полем. Узлы на стыке слоя сохраняются.</p><p>У tenzor native-библиотека и шаг до/после совпадают; полные поля и поверхности трёх повторов согласуются до относительной L2 {hist_l2:.3e}. Это изолированный контроль изменения носителя демпфирования. Малое изменение времени tenzor не следует считать доказанным ускорением или замедлением по трём повторам. Общий seismic имеет те же исходники, но отдельную сборку; специализированное семейство обновлено, поэтому изменение его времени нельзя целиком приписывать демпфированию.</p>')
    common_limit=max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:ax.set_xlim(0,common_limit)
    fig.suptitle('1400×700 м · два слоя · центральный Рикер 30 Гц · h=5 м · 1 с\nДемпфирование только в слое; усы — диапазон трёх повторов',fontsize=13)
    fig.savefig(OUT/'performance.png',dpi=160);plt.close(fig)
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>p4/p5: исправленный носитель демпфирования</title><style>body{font:16px system-ui;max-width:1350px;margin:32px auto;padding:0 24px;color:#20252b}h1{font-size:28px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}img{width:100%}a{color:#245a88}</style><h1>Полная задача p4/p5 с исправленным носителем демпфирования</h1><p>Saturn, одна A1000, FP64, CUDA12.0/SM80. Основная область1400×700 м, полная1750×875 м,61250 элементов по5 м. Один источник x=700,y=0: сила вниз, Рикер30 Гц,1 Н/м. Время1 с, β=1/27, γ=0,5, квадратичный k=0,5/ABC/175 м, свободный верх. Запись21 полного поля Vx/Vy и1001 отсчёта всех верхних GLL-узлов.</p><p>Для каждого порядка один полный прогрев каждого из трёх решателей исключён, затем три повтора с циклическим чередованием порядка в отдельных процессах. Всего24 запуска,18 измеряемых. Локальные GPU-расчёты не выполнялись.</p><img src="performance.png" alt="Время трёх CUDA-решателей при p4 и p5">'''+''.join(sections)+'''<h2>Проверка таблиц</h2><p>Локально встроенные таблицы p4/p5 совпали со стандартным ALine побитово. На Saturn стандартное построение отличается в последних битах: max|Δ|=1,78e−15 /1,11e−15. Первичный отказ строгого CPU-сравнения сохранён; повтор проверяет численный допуск rtol=atol=1e−14. Допуски сравнения полных решений не менялись.</p><h2>Границы вывода</h2><p>run включает интегрирование и обработчики регистрации. Полное время включает сборку модели, подготовку, запись и проход по амплитудам, до конечного SHA-хеширования; запуск Python не входит. Это практическое время данного опыта, не длительность отдельных CUDA-ядер.</p><p>p4 и p5 используют разные пространственные сетки GLL, число приёмников и собственный native dt. Их скорость можно сопоставлять как стоимость полного опыта, но это не доказательство одинаковой точности. Здесь сравниваются реализации одного порядка, а не пространственная сходимость p4 к p5. Контроль tenzor не является аналитическим решением. Порог согласования: компонентная L2&lt;1e−8 и max|Δv|/пик контроля&lt;1e−7. Основная очередь остаётся остановленной.</p></html>'''
    (OUT/'index.html').write_text(page)
    out=ROOT/'p45_damping_analysis_v1';out.mkdir(exist_ok=True)
    result:dict[str,object]={'status':'completed','results':records}
    if (out/'comparison.json').exists():
        assert read_json(out/'comparison.json')==result
    else:
        write(out/'comparison.json',result)


if __name__=='__main__':main()
