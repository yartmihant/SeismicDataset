"""Static kernel-time report for the two recorded hardware profiles."""
from __future__ import annotations
import html
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json, mapping, number, sequence
from scripts.dev_1_5_geometry_frequency.setup import write

ROOT=PROJECT_ROOT/'data/dev_1_5_geometry_frequency'
OUT=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/profile_v1'
CASES=(('saturn','A100 40 ГБ','profile_saturn_v1'),('local','RTX 3060 12 ГБ','profile_local_v2'))
ROLES={'material_response_kernel':'Упругий отклик: деформации, напряжения, сборка внутренних сил',
       'correct_nodal_state_kernel':'Корректор Ньюмарка–Кима: U, V, A',
       'nodal_damping_kernel':'Узловое массовое демпфирование',
       'predict_nodal_state_kernel':'Предиктор U и V',
       'boundary_damping_kernel':'Граничное условие ABC',
       'nodal_force_kernel':'Точечная сила',
       'velocity_frame_kernel':'Сборка кадра приёмников Vx/Vy',
       'flag_observation_kernel':'Проверка конечности записываемого кадра'}


def table(rows: list[dict[str,object]]) -> str:
    out='<table><thead><tr><th>Ядро</th><th>Вызовы</th><th>Сумма, с</th><th>Доля, %</th><th>Среднее, мкс</th><th>Медиана, мкс</th><th>P95, мкс</th><th>Максимум, мкс</th></tr></thead><tbody>'
    for r in rows:
        out+='<tr><td><code>'+html.escape(str(r['name']))+'</code><small>'+html.escape(ROLES.get(str(r['name']),''))+'</small></td>'
        for key,fmt in [('calls','.0f'),('total_seconds','.6f'),('percent_kernel_time','.3f'),('mean_us','.3f'),('median_us','.3f'),('p95_us','.3f'),('max_us','.3f')]:out+='<td>'+format(number(r[key]),fmt)+'</td>'
        out+='</tr>'
    return out+'</tbody></table>'


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    bodies=[];records:dict[str,object]={}
    fig,axes=plt.subplots(1,2,figsize=(16,6),layout='constrained')
    for ax,(key,label,folder) in zip(axes,CASES):
        root=ROOT/folder;meta=read_json(root/'profile_summary.json');run=mapping(mapping(meta['stages'])['semgpu_run'])
        rows=[mapping(r) for r in sequence(run['kernels'])];names=[str(r['name']) for r in rows]
        values=[number(r['total_seconds']) for r in rows];y=np.arange(len(rows))
        ax.barh(y,values,color='#315b86' if key=='saturn' else '#bc5a34');ax.invert_yaxis();ax.set_yticks(y,names,fontsize=9)
        ax.set_xlim(0,max(values)*1.35);ax.set_xlabel('Суммарное время выполнения ядра, с');ax.set_title(label+' · основной цикл')
        for i,r in enumerate(rows):ax.text(values[i]+max(values)*.015,i,f"{values[i]:.3f} с · {number(r['percent_kernel_time']):.2f}%",va='center',fontsize=9)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        env=read_json(root/'environment.json')
        records[key]={'root':str(root),'run':run,'environment':env}
        rel=lambda name:html.escape(os.path.relpath(root/name,OUT))
        body=f'<h2>{label}</h2><p>Весь этап: {number(run["wall_seconds"]):.3f} с; сумма ядер: {number(run["kernel_total_seconds"]):.3f} с; объединение интервалов GPU (ядра + memcpy + memset): {number(run["gpu_active_union_seconds"]):.3f} с.</p>'
        body+=f'<p><a href="{rel("trace.nsys-rep")}">Трасса Nsight</a> · <a href="{rel("kernels.csv")}">Полная таблица CSV</a> · <a href="{rel("profile_summary.json")}">Все события по этапам (JSON)</a></p>'
        body+=table(rows)
        body+='<h3>Обмен и заполнение памяти</h3><ul>'
        for item in sequence(run['memcpy']):
            m=mapping(item);body+=f'<li>{html.escape(str(m["kind"]))}: {number(m["total_seconds"]):.6f} с, {int(number(m["calls"]))} вызовов, {number(m["bytes"])/1e6:.2f} МБ.</li>'
        ms=mapping(run['memset']);body+=f'<li>Memset: {number(ms["total_seconds"]):.6f} с, {int(number(ms["calls"]))} вызовов.</li></ul>'
        body+=f'<p>Без зарегистрированной активности GPU внутри этапа: {number(run["no_captured_gpu_activity_seconds"]):.3f} с. Это промежутки только данного процесса, включая работу CPU, ожидания и запись; это не показатель простоя всей видеокарты.</p>'
        prep=mapping(mapping(meta['stages'])['semgpu_prepare'])
        body+=f'<details><summary>Подготовка: {number(prep["wall_seconds"]):.3f} с; ядра {number(prep["kernel_total_seconds"]):.6f} с</summary>'+table([mapping(r) for r in sequence(prep['kernels'])])+'</details>'
        body+='<details><summary>CUDA API на CPU: времена перекрываются с GPU, не суммировать с таблицей ядер</summary><ul>'
        for item in sequence(run['cuda_runtime_api']):
            api=mapping(item);body+=f'<li><code>{html.escape(str(api["name"]))}</code>: {number(api["total_host_seconds"]):.6f} с; {int(number(api["calls"]))} вызовов.</li>'
        body+='</ul></details><p class="note">'+html.escape(str(env['nsys']))+'; сборка '+html.escape(str(env['cuda_build']))+'.</p>';bodies.append(body)
    fig.suptitle('1400×700 м · два слоя · Рикер 30 Гц · p5/5 м · 52 000 шагов\nДоли от суммы времени ядер; шкалы панелей различаются',fontsize=14)
    fig.savefig(OUT/'kernels.png',dpi=160);plt.close(fig)
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Профиль нового сейсмического CUDA-решателя</title><style>body{font:16px system-ui;margin:32px auto;max-width:1450px;padding:0 24px;color:#20252b}h1{font-size:28px}h2{margin-top:40px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}td,th{border-bottom:1px solid #ddd;padding:10px;text-align:right}td:first-child,th:first-child{text-align:left}small{display:block;color:#666;font-size:12px}img{width:100%}details{margin:18px 0}a{color:#245a88}.note{color:#666}</style><h1>Какие ядра занимают время</h1><p>Новый lin_elast_2d_seismic_newmark_kim_cu, FP64. Двухслойная основная область1400×700 м, полная1750×875 м; p5/5 м,30 Гц,1 с, k=0,5/ABC/175 м, β=1/27, CFL0,8. Одинаковые52 000 шагов;21 поле и1001 отсчёт1751 приёмника. 1 533 876 узлов.</p><img src="kernels.png" alt="Время всех ядер на A100 и RTX 3060"><p>Каждый интервал CUDA взят из трассы Nsight Systems; этапы выделены NVTX-метками. Доли относятся к сумме длительностей ядер, не ко времени всего процесса. Nsight Compute и аппаратные счётчики не применялись: по этим измерениям нельзя однозначно разделить ограничения FP64, памяти и атомарных операций.</p>'''+''.join(bodies)+'''<h2>Интерпретация</h2><p>Главные затраты — упругий отклик, корректор, узловое демпфирование и предиктор. Прямое GPU-время записи кадров, источника и ABC мало. cudaStreamSynchronize в CPU-отчёте включает ожидание выполнения GPU; его большой итог не является отдельной вычислительной стоимостью.</p><p>Узловое демпфирование сейчас запускается на всех узлах, в том числе с нулевым коэффициентом внутри основной области. Ограничение носителя ненулевыми узлами — кандидат для отдельного эксперимента; ускорение таким изменением здесь не измерялось.</p><p>Профилировщики и CUDA-компиляторы различаются между машинами и указаны выше. Первичная локальная трасса profile_local_v1 сохранена исторически: Nsight2024.6.2 сообщил о более новом драйвере. В итог используется повтор profile_local_v2 современным установленным профилировщиком. В локальной трассе2026.5.2 есть предупреждение о возможной неполноте NVTX. Обе заданные метки найдены с началом и концом; их длительности согласованы с таймером запускателя, все ожидаемые314044 вызова основных ядер учтены. Это по одному принятому полному профилю на машину, а не статистический тест аппаратных пределов.</p></html>'''
    (OUT/'index.html').write_text(page)
    analysis=ROOT/'profile_analysis_v1';analysis.mkdir(exist_ok=True)
    write(analysis/'comparison.json',{'status':'completed','profiles':records,'scope':'kernel timing only, no hardware-counter bottleneck diagnosis'})


if __name__=='__main__':main()
