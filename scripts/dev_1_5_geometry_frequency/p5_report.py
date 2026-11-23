"""Render the saved full-problem p5 benchmark, without GPU work or SSH."""
from __future__ import annotations
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json, mapping, sequence, number
from scripts.dev_1_5_geometry_frequency.setup import write

ROOT=PROJECT_ROOT/'data/dev_1_5_geometry_frequency'
OUT=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/p5_comparison_v1'
LABELS={'tenzor':'tenzor','seismic':'seismic (текущий)','seismic_p5':'seismic p5'}


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    fig,axis=plt.subplots(figsize=(10,4.8),layout='constrained')
    sections=[];summary:dict[str,object]={}
    for ax,(host,title) in zip([axis],[('saturn','Saturn · A100')]):
        root=ROOT/f'p5_{host}_v1';analysis=read_json(root/'analysis.json');plan=read_json(root/'plan.json')
        assert analysis['status']=='completed' and analysis['all_agree'] is True
        timings={str(mapping(r)['backend']):mapping(r) for r in sequence(analysis['timings'])}
        values=[number(timings[b]['median_run_seconds']) for b in LABELS]
        lo=[min(number(v) for v in sequence(timings[b]['run_seconds'])) for b in LABELS]
        hi=[max(number(v) for v in sequence(timings[b]['run_seconds'])) for b in LABELS]
        ax.barh(list(LABELS.values()),values,color=['#b3bdc5','#688fa7','#286447'],xerr=np.array([np.array(values)-lo,np.array(hi)-values]),capsize=4)
        ax.invert_yaxis();ax.set_xlabel('Время run, с · медиана трёх повторов');ax.set_title(title);ax.set_xlim(0,max(hi)*1.18);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        for y,v in enumerate(values):ax.text(v+max(hi)*.025,y,f'{v:.3f}',va='center')
        base=values[0];generic=values[1];p5=values[2]
        max_l2=max(number(v) for item in sequence(analysis['comparisons']) for key in ('fields','surface') for v in sequence(mapping(mapping(item)[key])['relative_l2_components']))
        max_abs=max(number(v) for item in sequence(analysis['comparisons']) for key in ('fields','surface') for v in sequence(mapping(mapping(item)[key])['max_abs_m_s']))
        summary[host]={'median_run_seconds':dict(zip(LABELS,values)),'p5_speedup_vs_tenzor':base/p5,'p5_speedup_vs_seismic':generic/p5,'max_component_relative_l2':max_l2,'max_abs_difference_m_s':max_abs,'all_agree':True}
        rows=''
        for b in LABELS:
            row=timings[b]
            fmt=lambda key: ', '.join(f'{number(v):.3f}' for v in sequence(row[key]))
            rows+=f'<tr><th>{LABELS[b]}</th><td>{fmt("run_seconds")}</td><td>{number(row["median_run_seconds"]):.3f}</td><td>{fmt("prepare_seconds")}</td><td>{number(row["median_total_seconds"]):.3f}</td></tr>'
        env=read_json(root/'environment.json')
        sections.append(f'<h2>{title}</h2><p>Ускорение p5 относительно tenzor: <b>{base/p5:.3f}×</b>; относительно текущего seismic: <b>{generic/p5:.3f}×</b>. Сокращение run: {(1-p5/base)*100:.2f}% / {(1-p5/generic)*100:.2f}%.</p><table><thead><tr><th>Решатель</th><th>Три run, с</th><th>Медиана run, с</th><th>Три подготовки, с</th><th>Медиана полного времени, с</th></tr></thead><tbody>{rows}</tbody></table><p>Максимум относительной L2 компоненты против tenzor: {max_l2:.3e}; max|Δv|={max_abs:.3e} м/с. Проверены все сохранённые поля и поверхности трёх повторов; это численное согласование, не независимая валидация.</p><p>Native dt прогревов, с: {html.escape(str(plan["native_dt_s"]))}; принудительный общий dt: {html.escape(str(plan["explicit_common_dt_s"]))}.</p><p><a href="../../../data/dev_1_5_geometry_frequency/p5_{host}_v1/analysis.json">Полные метрики и нормы</a> · <a href="../../../data/dev_1_5_geometry_frequency/p5_{host}_v1/environment.json">Среда и происхождение</a></p><details><summary>GPU</summary><pre>{html.escape(str(env["gpu"]))}</pre></details>')
    fig.suptitle('Полная задача 1400×700 м · два слоя · 30 Гц · p5/5 м · 1 с\nПрогрев исключён; усы — минимум и максимум трёх повторов',fontsize=13)
    fig.savefig(OUT/'performance.png',dpi=160);plt.close(fig)
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Специализация p5 — полная задача</title><style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#20252b}h1{font-size:28px}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}img{width:100%}pre{white-space:pre-wrap}a{color:#245a88}</style><h1>Экспериментальный решатель p5 на полной задаче</h1><p>Двухслойная область1400×700 м; со слоем1750×875 м. FP64,1 533 876 узлов,61 250 элементов, точечная сила вниз30 Гц, p5/5 м,1 с, β=1/27, k=0,5/ABC/175 м. Регистрация21 полного поля Vx/Vy и1001 отсчёта1751 верхнего узла. Одна A1000, отдельные процессы, один полный прогрев каждого решателя и три повтора с циклическим чередованием порядка.</p><img src="performance.png" alt="Медианы времени полного цикла расчёта трёх решателей на A100">'''+''.join(sections)+'''<h2>Что измерено</h2><p>run включает интегрирование и обработчики регистрации полей/сейсмограмм. Полное время начинается перед сборкой модели и заканчивается после записи и прохода по амплитудам, но перед конечным SHA-хешированием; запуск Python не включён. Это практическая производительность данного опыта, не время отдельных CUDA-ядер. Профилировщик не применялся.</p><p>Текущий общий seismic и p5 собраны из одного снимка seismic_p5_v1. Общий seismic уже отличается от версии в прежнем сравнении и профиле; исторические35,321 с A100 нельзя приписывать этой сборке. tenzor сохранён из предыдущей изолированной среды. Сборка на Saturn: CUDA12.0/SM80. Локальная серия RTX3060 остановлена по уточнению пользователя; её неполные результаты сохранены и исключены из итоговой статистики.</p><p>Норма разности — невзвешенная L2 массива отдельно по компонентам, всем сохранённым узлам и временам. Порог: L2&lt;1e−8 и максимум разности/пик контроля&lt;1e−7. Контроль tenzor не является точным решением. Вывод относится к одному полному тесту p5; другие порядки, геометрии и режимы регистрации не исследованы. Производственная очередь не возобновлялась.</p></html>'''
    (OUT/'index.html').write_text(page)
    dest=ROOT/'p5_analysis_v1';dest.mkdir(exist_ok=True)
    write(dest/'comparison.json',{'status':'completed','hosts':summary})


if __name__=='__main__':main()
