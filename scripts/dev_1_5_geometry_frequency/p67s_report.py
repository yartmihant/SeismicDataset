"""Local HTML and scientific figures for the p67s/Nsight experiment."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.dev_1_5_geometry_frequency.p67s_run import ROOT,MODES
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.reporting import read_json,number,mapping,sequence

NAMES={'old_sync':'p6_7s · прежний приёмник','new_none':'p67s · без приёмника','new_discard':'p67s · приём без накопления','new_sync':'p67s · синхронно в RAM','new_async':'p67s · асинхронно в RAM'}


def render(root:Path,out:Path)->None:
    m=read_json(root/'analysis/metrics.json');assert m['status']=='completed'
    rows={str(r['mode']):r for r in (mapping(x) for x in sequence(m['timings']))}
    profiles={mode:read_json(root/'profiles'/mode/'profile_summary.json') for mode in MODES}
    out.mkdir(parents=True,exist_ok=False)
    med=[number(rows[mode]['median_run_seconds']) for mode in MODES]
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    for ax,indices,title in [(axes[0],list(range(5)),'Сравнение с прежним p6_7s'),(axes[1],list(range(2,5)),'Добавка относительно p67s без приёмника')]:
        delta_panel=ax is axes[1]
        values=[med[i]-(med[1] if delta_panel else 0.) for i in indices];names=[NAMES[MODES[i]] for i in indices]
        ax.barh(names,values,color='#356b88',alpha=.85)
        for j,i in enumerate(indices):
            samples=[number(v) for v in sequence(rows[MODES[i]]['run_seconds'])]
            if delta_panel:samples=[a-number(b) for a,b in zip(samples,sequence(rows['new_none']['run_seconds']),strict=True)]
            ax.scatter(samples,[j]*len(samples),s=17,color='#162734',zorder=3)
            ax.text(values[j]+max(values)*.01,j,f'{values[j]:.3f}',va='center')
        ax.invert_yaxis();ax.set_xlim(min(0.,min(values)*1.2),max(values)*1.3);ax.set_xlabel('Разность run / wait, с' if delta_panel else 'Полный model.run / wait, с');ax.set_title(title)
    fig.suptitle('A100 · 7 источников · p6/h10 · 1 с · медиана и пять повторов без профайлера')
    fig.savefig(out/'timings.png',dpi=155);plt.close(fig)
    stage={mode:mapping(mapping(profiles[mode]['stages'])['semgpu_run']) for mode in MODES}
    newkern=[mapping(x) for x in sequence(stage['new_sync']['kernels'])]
    fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
    labels=[str(x['name']) for x in newkern];values=[number(x['seconds']) for x in newkern]
    ax.barh(labels,values,color='#356b88');ax.invert_yaxis();ax.set_xlim(0,max(values)*1.23)
    for i,r in enumerate(newkern):ax.text(values[i]+max(values)*.01,i,f"{values[i]:.3f} с · {number(r['share_percent']):.1f}%",va='center')
    ax.set_xlabel('Сумма длительностей CUDA-ядер, с');ax.set_title('p67s с синхронным приёмником · отдельный Nsight-запуск')
    fig.savefig(out/'kernels.png',dpi=155);plt.close(fig)
    timing_table=''.join(f"<tr><td>{NAMES[mode]}</td><td>{number(rows[mode]['median_run_seconds']):.6f}</td><td>{number(rows[mode]['min_run_seconds']):.6f}…{number(rows[mode]['max_run_seconds']):.6f}</td><td>{number(rows[mode]['median_solve_wall_seconds']):.6f}</td><td>{number(rows[mode]['median_callback_wall_seconds']):.6f}</td><td>{number(rows[mode]['median_output_npz_seconds']):.6f}</td></tr>" for mode in MODES)
    kernel_table=''.join(f"<tr><td><code>{html.escape(str(r['name']))}</code></td><td>{r['calls']}</td><td>{number(r['seconds']):.6f}</td><td>{number(r['share_percent']):.2f}</td><td>{number(r['mean_us']):.3f}</td></tr>" for r in newkern)
    timeline=[]
    for mode in MODES:
        r=stage[mode]
        d2h=[mapping(x) for x in sequence(r['memcpy']) if 'DtoH' in str(mapping(x)['kind']) or 'Device-to-Host' in str(mapping(x)['kind'])]
        # Preserve all labels/bytes in JSON; table shows summed copies independently of label spelling.
        copies=[mapping(x) for x in sequence(r['memcpy'])]
        timeline.append(f"<tr><td>{NAMES[mode]}</td><td>{number(r['wall_seconds']):.6f}</td><td>{number(r['kernel_seconds']):.6f}</td><td>{number(r['gpu_active_union_seconds']):.6f}</td><td>{number(r['no_captured_gpu_activity_seconds']):.6f}</td><td>{sum(number(c['seconds']) for c in copies):.6f}</td></tr>")
    details=[]
    for mode in MODES:
        r=stage[mode]
        kernels=''.join(f"<tr><td><code>{html.escape(str(x['name']))}</code></td><td>{x['calls']}</td><td>{number(x['seconds']):.6f}</td><td>{number(x['share_percent']):.2f}</td></tr>" for x in (mapping(v) for v in sequence(r['kernels'])))
        api=''.join(f"<tr><td>{html.escape(str(x['name']))}</td><td>{x['calls']}</td><td>{number(x['host_seconds']):.6f}</td></tr>" for x in (mapping(v) for v in sequence(r['cuda_api'])[:10]))
        details.append(f'<details><summary>{NAMES[mode]}: все ядра, CUDA API и копирования</summary><table><tr><th>Ядро</th><th>Вызовы</th><th>Сумма, с</th><th>% ядер</th></tr>{kernels}</table><table><tr><th>CUDA API</th><th>Вызовы</th><th>Host, с</th></tr>{api}</table><pre>{html.escape(str(r["memcpy"]))}</pre><p>Memset: {number(r["memset_seconds"]):.6f} с, {r["memset_calls"]} вызовов.</p></details>')
    sync=number(m['sync_receiver_overhead_seconds']);asyn=number(m['async_receiver_overhead_seconds']);base=med[1]
    paired=mapping(m['paired_deltas_vs_none_seconds'])
    delta_table=''.join(f"<tr><td>{NAMES[mode]}</td><td>{', '.join(f'{number(v):+.4f}' for v in sequence(paired[mode]))}</td></tr>" for mode in MODES[2:])
    diagnostics=''.join(f'<li>{NAMES[mode]}: {html.escape(str(profiles[mode]["diagnostics"]))}</li>' for mode in MODES)
    checks=[mapping(c) for c in sequence(m['checks'])]
    field_l2=max(max(number(v) for v in sequence(c['relative_l2'])) for c in checks if c['kind']=='final_full_velocity')
    surface_l2=max(max(number(v) for v in sequence(c['relative_l2'])) for c in checks if c['kind']=='seismogram')
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>p67s: производительность и приёмники</title><style>body{{font:16px/1.5 system-ui;max-width:1280px;margin:35px auto;padding:0 25px;color:#182530}}h1,h2{{line-height:1.2}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;margin:14px 0}}td,th{{padding:8px;text-align:right;border-bottom:1px solid #ced4db}}td:first-child,th:first-child{{text-align:left}}code,pre{{font-size:13px;overflow-wrap:anywhere;white-space:pre-wrap}}details{{margin:16px 0}}summary{{cursor:pointer;font-weight:600}}</style>
<h1>p67s на A100: вычисления и стоимость приёмника</h1>
<p>Двухслойная1400×700 м; p6/h10, PML350 без ABC;7 источников Рикера30 Гц;1 с, β=1/27, γ=0,5. Native CFL0,8: одинаковые9000 шагов dt=1/9000 с. Полный верх:1261 узел на источник,1001 временной отсчёт Vx/Vy. После уточнения пользователя исследуется только существующий приёмник; <strong>SEG-Y в этом опыте не записывается</strong>.</p>
<p><strong>Ускорение p67s относительно p6_7s при одинаковом синхронном приёмнике: {number(m['new_vs_old_speedup']):.3f} раза.</strong> Медиана {med[0]:.3f}→{med[3]:.3f} с за семь источников, {med[3]/7:.3f} с на источник. Асинхронный вариант: {med[4]:.3f} с.</p>
<h2>Замеры без профайлера</h2><p>Одна A100GPU0; прогрев каждого режима, пять циклически переставляемых раундов.30 полных запусков. Асинхронный обработчик использует frame_buf_size=4, run(sync=False).wait(); окончание включает опустошение очереди. Время callback — сумма интервалов самого Python-обработчика, она может перекрываться с GPU и не прибавляется к run. Запись NPZ в ОС происходит отдельно после run; fsync до физического носителя не измеряется.</p>
<table><tr><th>Режим</th><th>run, с</th><th>min…max, с</th><th>Сборка+prepare+run, с</th><th>Callback, с</th><th>NPZ после run, с</th></tr>{timing_table}</table><img src="timings.png" alt="Сравнение времени пяти режимов">
<h2>Накладные расходы существующего приёмника</h2><p>Разность медиан относительно режима без приёмника: синхронный сбор {sync:+.6f} с ({100*sync/base:+.3f}%), асинхронный {asyn:+.6f} с ({100*asyn/base:+.3f}%). Это разность полного времени, а не сумма длительностей ядер приёмника. Контроль «без накопления» выполняет GPU-gather, D2H и те же проверки времени/конечности, но не копирует данные в постоянный массив поверхности.</p>
<table><tr><th>Режим минус «без приёмника»</th><th>Разности в пяти раундах, с</th></tr>{delta_table}</table><p>Разности порядка разброса повторов нельзя трактовать как устойчивое ускорение. Во всех режимах сохранены1000 тактов и штатный terminal SAVE. «Без приёмника» не означает отсутствие всех синхронизаций или всех D2H-копирований: конечное состояние и PML-history сохраняются штатным решателем.</p>
<h2>Профилирование CUDA/NVTX</h2><p>Пять отдельных профилей Nsight Systems: CUDA+NVTX, без CPU sampling/context switches. Эти времена исключены из медиан выше. Доли — от суммы времени ядер, не от полного времени процесса. Суммы GPU, копирований и host CUDA API перекрываются; складывать их нельзя. Остаток без зарегистрированной GPU-активности включает запуск ядер, CPU, ожидания и накладные расходы профайлера, а не только приёмник.</p>
<table><tr><th>Профиль</th><th>NVTX run, с</th><th>Ядра, с</th><th>Объединение GPU-интервалов, с</th><th>Без GPU-активности, с</th><th>Все memcpy, с</th></tr>{''.join(timeline)}</table>
<img src="kernels.png" alt="Время CUDA-ядер нового решателя"><table><tr><th>Ядро p67s, sync</th><th>Вызовы</th><th>Сумма, с</th><th>% ядер</th><th>Среднее, мкс</th></tr>{kernel_table}</table>
<p>material_response_kernel&lt;true&gt; обрабатывает PML, &lt;false&gt; обычные элементы. У корректора первый шаблонный флаг означает обратную массу, второй — совмещение с предиктором следующего шага. Проверено число предикторов:9000 у прежнего решателя и1000 у нового; коррекций9000, захватов поверхности1001 либо0 в режиме без приёмника.</p>{''.join(details)}
<h2>Численная проверка</h2><p>Поверхности Vx/Vy сравниваются по каждому из семи источников за весь интервал, включая близкие узлы и PML. Max relative L2 сейсмограмм={surface_l2:.6g}; конечного полного поля скорости во всех узлах={field_l2:.6g}. Допуски1e−8 по L2 и1e−7 по относительному пику соблюдены. Проверены {len(sequence(m['verified_files']))} SHA массивов. Профилированные запуски также проверены; конечный верх равен последнему кадру записи. Полных анимационных кадров нет; один конечный кадр во всех узлах сохранён только для прогревов, после окончания таймера.</p>
<h2>Изменения и воспроизводимость</h2><p>Новый snapshot seismic_p67s_v1: общая геометрия семи опытов, Рикер подготовлен заранее, коэффициенты производных p6 переносятся в shared memory, корректор использует обратную массу, промежуточные корректор/предиктор слиты. Обе версии собраны отдельно; старые исходники и результаты сохранены. Этот опыт оценивает совокупный эффект изменений; причинный вклад каждой оптимизации отдельно не измерен.</p>
<p>Запускатель p67s_run.py, анализатор p67s_analyze.py, renderer p67s_report.py; серия <code>{html.escape(str(root))}</code>. Полные NPZ/NPY и trace.nsys-rep/trace.sqlite — на Saturn, компактные JSON/отчёт — локально. Производственная очередь остановлена.</p><details><summary>Диагностика Nsight</summary><ul>{diagnostics}</ul></details></html>'''
    (out/'index.html').write_text(page)
    write(out/'manifest.json',{'files':[file_record(out/name) for name in ('index.html','timings.png','kernels.png')],'metrics':file_record(root/'analysis/metrics.json')})


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=Path('img/dev_1_5_geometry_frequency/p67s_receiver_v1'));a=p.parse_args();render(a.root,a.out)

if __name__=='__main__':main()
