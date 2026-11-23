"""Saved A100 benchmark report; no solver execution."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.dev_1_5_geometry_frequency.p6_7s_compare import ROOT, VARIANTS, result_path
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.reporting import read_json, mapping, number, sequence

LABELS=('Прежний p6: 7 сеток','Актуальный p6: 7 сеток','p6_7s: общая сетка')


def render(root:Path,out:Path)->None:
    metrics=read_json(root/'analysis/metrics.json');assert metrics['status']=='completed'
    rows=[mapping(x) for x in sequence(metrics['timings'])]
    out.mkdir(parents=True,exist_ok=False)
    med=[number(r['median_run_seconds']) for r in rows]
    wall=[number(r['median_solve_wall_seconds']) for r in rows]
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for ax,values,title in zip(axes,[med,wall],['Интегрирование и сейсмограммы','Сборка + prepare + интегрирование'],strict=True):
        ax.barh(LABELS,values,color=['#8b919a','#626f7b','#246b8e'])
        for i,v in enumerate(values):ax.text(v+.1,i,f'{v:.3f}',va='center')
        ax.invert_yaxis();ax.set_xlim(0,max(values)*1.18);ax.set_xlabel('Время семи опытов, с');ax.set_title(title)
    fig.suptitle('A100 · p6/h10 · PML350 без ABC · 1 с · медиана трёх повторов')
    fig.savefig(out/'timing.png',dpi=160);plt.close(fig)
    base=result_path(root,'frozen_p6','repeat0');new=result_path(root,'p6_7s','repeat0')
    with np.load(base/'surface.npz') as z:ref=z['velocity_m_s'][:,3*1261:4*1261];x=z['coordinates_m'][3*1261:4*1261,0]
    with np.load(new/'surface.npz') as z:actual=z['velocity_m_s'][:,3*1261:4*1261]
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for c,name in enumerate(('Vx','Vy')):
        limit=float(np.max(abs(ref[:,:,c])));delta=actual[:,:,c]-ref[:,:,c];dl=float(np.max(abs(delta)))
        for j,(value,lim,title) in enumerate([(ref[:,:,c],limit,'Прежний p6'),(actual[:,:,c],limit,'p6_7s'),(delta,max(dl,1e-30),'Разность (своя шкала)')]):
            ax=axes[c,j];im=ax.pcolormesh(x,np.arange(1001)*.001,value,cmap='RdBu_r',vmin=-lim,vmax=lim,shading='auto',rasterized=True)
            ax.invert_yaxis();ax.set_xlabel('x, м');ax.set_ylabel('Время, с');ax.set_title(name+' · '+title);fig.colorbar(im,ax=ax,label='м/с')
    fig.suptitle('Центральный источник x=700 м · весь верх с PML · линейные знаковые скорости')
    fig.savefig(out/'central_seismograms.png',dpi=145);plt.close(fig)
    pairs=[mapping(x) for x in sequence(metrics['comparisons'])]
    compare_rows=[]
    for variant,label in zip(VARIANTS,LABELS,strict=True):
        selected=[r for r in pairs if r['variant']==variant]
        l2=max(max(number(v) for v in sequence(r['relative_l2_components'])) for r in selected)
        peak=max(max(number(v) for v in sequence(r['peak_relative_components'])) for r in selected)
        absolute=max(max(number(v) for v in sequence(r['max_abs_delta_m_s'])) for r in selected)
        compare_rows.append(f'<tr><td>{label}</td><td>{l2:.6g}</td><td>{peak:.6g}</td><td>{absolute:.6g}</td></tr>')
    speed=med[0]/med[2];current_speed=med[1]/med[2]
    table=''.join(f'<tr><td>{label}</td><td>{med[i]:.6f}</td><td>{wall[i]:.6f}</td><td>{med[i]/7:.6f}</td></tr>' for i,label in enumerate(LABELS))
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>p6_7s — сравнение на A100</title>
<style>body{{font:17px/1.55 system-ui;max-width:1200px;margin:40px auto;padding:0 24px;color:#19232c}}h1,h2{{line-height:1.2}}table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}td,th{{padding:9px;border-bottom:1px solid #ccd3da;text-align:right}}td:first-child,th:first-child{{text-align:left}}img{{max-width:100%}}code{{overflow-wrap:anywhere}}</style>
<h1>p6_7s: семь независимых опытов на общей сетке</h1>
<p>Двухслойная модель 1400×700 м; p6, элементы10 м; PML350 м, dmax173,088611 с⁻¹, без ABC и граничного пространства; Рикер30 Гц, источники x175…1225 через175 м. Время1 с, β=1/27, γ=0,5. Только Vx/Vy:1001×1261×2 на источник, весь верх с PML.</p>
<p><strong>p6_7s быстрее прежнего пакета в {speed:.3f} раза, актуального p6 — в {current_speed:.3f} раза.</strong> Эквивалентное время на источник: {med[2]/7:.3f} с. Это семь независимых состояний, не суперпозиция.</p>
<h2>Измерения</h2><p>Одна A100-PCIE-40GB GPU0; по одному прогреву, затем три повтора на вариант с циклическим порядком. Все родные шаги при CFL0,8 совпали: dt=1/9000 с, 9000 шагов. run включает приёмники и перенос сейсмограмм в RAM, исключает prepare и запись NPZ. Последний столбец — деление времени пакета на7, не время одиночного запуска.</p>
<table><tr><th>Решатель</th><th>Медиана run, с</th><th>Сборка+prepare+run, с</th><th>run/7, с</th></tr>{table}</table><img src="timing.png" alt="Медианные времена трёх вариантов">
<h2>Совпадение решений</h2><p>Максимумы по семи источникам, обеим компонентам, прогреву и трём повторам. Полная запись0–1 с и все1261 узлов на источник, без исключения близких трасс. L2 и пик — безразмерные отношения, не проценты. Критерии: L2&lt;1e−8, относительный пик&lt;1e−7; все проверки пройдены. Дополнительно сравнение с сохранённым историческим пакетом, проверены {len(sequence(metrics['verified_sha256']))} SHA256.</p>
<table><tr><th>Относительно прежнего p6 в том же раунде</th><th>Относительная L2</th><th>Относительный пик разности</th><th>max|Δv|, м/с</th></tr>{''.join(compare_rows)}</table><img src="central_seismograms.png" alt="Центральная сейсмограмма и разность">
<h2>Что изменилось в реализации</h2><p>Одна сетка795691 узел/22050 элементов вместо семи копий5569837 узлов/154350 элементов. Геометрия, материалы, масса и связность общие; u/v/a и PML-history независимы. На PML-элементах упругий и PML-отклик объединены в одном ядре; обычные элементы обрабатываются отдельно. Верх перенумерован первым, связность сохраняет физические координаты. Граничного пространства нет.</p>
<p>Новый интерфейс принимает только11000 и31100; сила встроена. prepare вычисляет9000 значений аналитического Рикера и один раз переносит их на GPU. Прежний p6 интерполирует таблицу с шагом1 мкс и передаёт текущую силу каждый шаг. Поэтому измерен совокупный эффект специализации, а не изолированный выигрыш единственного ядра. Это не Nsight-профилирование.</p>
<h2>Ограничения и воспроизводимость</h2><p>Новый источник жёстко фиксирует30 Гц, t0=0,05, Fy=−1 Н/м, семь узлов и геометрию верха. Для других частот/позиций нужна пересборка. История PML и поверхности конечны; новые полные поля не записывались. Согласование реализаций не доказывает пространственную сходимость или устойчивость за пределами1 с.</p>
<p>Снимок: docs/runtime_sources/seismic_p6_7s_v1. Текущий p6 собран из того же снимка; прежний p6 использует неизменённую прежнюю среду. Скрипты p6_7s_compare.py/build_p6_7s.sh/p6_7s_report.py. Полные NPZ на Saturn в <code>{html.escape(str(root))}</code>; JSON и отчёт скопированы локально. Основная очередь остаётся остановленной.</p></html>'''
    (out/'index.html').write_text(page)
    write(out/'manifest.json',{'files':[file_record(out/n) for n in ('index.html','timing.png','central_seismograms.png')],
        'source':file_record(root/'analysis/metrics.json')})


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=Path('img/dev_1_5_geometry_frequency/p6_7s_comparison_v2'));a=p.parse_args();render(a.root,a.out)

if __name__=='__main__':main()
