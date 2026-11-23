"""Warmup, rotated timings, full-state equivalence and mirror-shot validation."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_6_p64s_production.run import SNAPSHOT
from scripts.reporting import read_json,mapping,number,sequence
ROOT=Path('data/dev_1_6_p64s_production/comparison_v1')
VARIANTS=('p6','p64s','p67s')


def series(root:Path)->None:
    root.mkdir(parents=True,exist_ok=False)
    write(root/'plan.json',{'models':[1,0,2999],'variants':list(VARIANTS),'benchmark_model':1,'warmups':1,'rotated_repeats':3,'shots':'p6 four sequential; p64s four simultaneous; p67s seven simultaneous; sources100:200:1300','receiver':'140 points x5:10:1395, Vx/Vy,1001 times,FP64; async buffer4; no frames','snapshot':file_record(SNAPSHOT),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,name,uuid,utilization.gpu','--format=csv'],text=True)})
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    def launch(model:int,variant:str,label:str,state:bool=False)->None:
        out=root/f'{model:05d}'/variant/label;cmd=[sys.executable,'-m','scripts.dev_1_6_p64s_production.run','--model',str(model),'--variant',variant,'--out',str(out)]
        if state:cmd.append('--state')
        with (root/f'{model:05d}_{variant}_{label}.log').open('x') as log:subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    for v in VARIANTS:launch(1,v,'warmup',True)
    for repeat in range(3):
        for v in VARIANTS[repeat:]+VARIANTS[:repeat]:launch(1,v,f'repeat{repeat}')
    for model in (0,2999):
        for v in VARIANTS:launch(model,v,'check')
    write(root/'completed.json',{'status':'completed','calls':18,'single_source_integrations':24,'p64s_integrations':6,'p67s_integrations':6})


def extra(root:Path)->None:
    assert (root/'completed.json').exists()
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    for v in VARIANTS:
        out=root/'02994'/v/'check'
        with (root/f'02994_{v}_check.log').open('x') as log:subprocess.run([sys.executable,'-m','scripts.dev_1_6_p64s_production.run','--model','2994','--variant',v,'--out',str(out)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    write(root/'extra_completed.json',{'status':'completed','model':2994,'reason':'lowest maximum Vp in first3000, largest anticipated native dt','additional_calls':3})


def metric(a:FloatArray,b:FloatArray)->dict[str,object]:
    delta=a-b;rel=float(np.linalg.norm(delta.ravel())/max(float(np.linalg.norm(b.ravel())),1e-300));peak=float(np.max(abs(delta))/max(float(np.max(abs(b))),1e-300))
    return {'relative_l2':rel,'peak_relative':peak,'max_abs':float(np.max(abs(delta))),'pass':rel<1e-8 and peak<1e-7}


def analyze(root:Path)->None:
    rows:list[dict[str,object]]=[];checks:list[dict[str,object]]=[];files=[]
    for rp in sorted(root.glob('*/*/*/result.json')):
        r=read_json(rp);assert r['status']=='completed';rows.append(r)
        for name in ('surface','state','pml_history'):
            if name in r:
                saved=mapping(r[name]);path=rp.parent/Path(str(saved['path'])).name;actual=file_record(path);assert actual['sha256']==saved['sha256'];files.append(actual)
    for model,labels in ((1,('warmup','repeat0','repeat1','repeat2')),(0,('check',)),(2999,('check',)),(2994,('check',))):
        for label in labels:
            paths={v:root/f'{model:05d}'/v/label for v in VARIANTS};arrays={v:np.load(p/'velocity.npy') for v,p in paths.items()}
            records={v:read_json(p/'result.json') for v,p in paths.items()};steps={number(mapping(sequence(r['solver_records'])[0])['dt_s']) for r in records.values()};assert len(steps)==1
            for shot in range(4):
                for other in ('p6','p67s'):
                    for component in range(2):checks.append({'model':model,'label':label,'kind':'surface','shot':shot,'component':component,'reference':other,**metric(arrays['p64s'][:,shot,:,component],arrays[other][:,shot,:,component])})
            for shot in range(4):
                mirror=arrays['p64s'][:,shot,::-1].copy();mirror[:,:,0]*=-1
                checks.append({'model':model,'label':label,'kind':'mirror','shot':shot,'reference_shot':6-shot,**metric(mirror,arrays['p67s'][:,6-shot])})
            if label=='warmup':
                for filename in ('state.npy','pml_history.npy'):
                    a=np.load(paths['p64s']/filename,mmap_mode='r')
                    for other in ('p6','p67s'):
                        b=np.load(paths[other]/filename,mmap_mode='r')[:len(a)]
                        for row in range(len(a)):checks.append({'model':model,'kind':filename,'row':row,'reference':other,**metric(a[row],b[row])})
    timings={v:{key:float(np.median([number(read_json(root/'00001'/v/f'repeat{i}/result.json')[key]) for i in range(3)])) for key in ('run_seconds','total_seconds','prepare_seconds')} for v in VARIANTS}
    agree=all(c['pass'] for c in checks);out=root/'analysis';out.mkdir(exist_ok=False)
    write(out/'metrics.json',{'status':'completed' if agree else 'failed','checks':checks,'timings_median3':timings,'array_hashes':files,'rows':rows,'note':'p6 sums four sequential sources; p64s four; p67s seven with same first four. Not a seven-shot throughput comparison at equal work.'})
    assert agree,'Solver or mirror comparison failed'
    print(timings,flush=True)

def report(root:Path,out:Path)->None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    result=read_json(root/'analysis/metrics.json');assert result['status']=='completed'
    timings=mapping(result['timings_median3']);out.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),layout='constrained')
    names=['p6: четыре последовательно','p64s: четыре совместно','p67s: семь совместно']
    for ax,key,title in zip(axes,('run_seconds','total_seconds'),('Интегрирование и приёмник','Подготовка, расчёт,\nпроверка и запись')):
        values=[number(mapping(timings[v])[key]) for v in VARIANTS]
        bars=ax.barh(names,values,color=['#667788','#268865','#467ab5']);ax.invert_yaxis();ax.set(xlabel='Время, с',title=title,xlim=(0,max(values)*1.3));ax.bar_label(bars,fmt='%.3f',padding=4);ax.grid(axis='x',alpha=.2)
    fig.suptitle('A100, EFVB00001: медиана трёх повторов после прогрева')
    fig.savefig(out/'timings.png',dpi=150);plt.close(fig)
    checks=[mapping(c) for c in sequence(result['checks'])]
    maxima={kind:max(number(c['relative_l2']) for c in checks if c['kind']==kind) for kind in ('surface','mirror','state.npy','pml_history.npy')}
    p6=number(mapping(timings['p6'])['run_seconds']);four=number(mapping(timings['p64s'])['run_seconds']);seven=number(mapping(timings['p67s'])['run_seconds'])
    text=f"""# p64s: точность, скорость и первый чанк EFVB

[Подробная постановка](../../../docs/dev_1_6_p64s_production_spec.txt). Основная область1400×700 м, p6/h10, PML350 без ABC, Рикер30Гц/1Н/м вниз,1с,β1/27,γ0.5,native CFL0.8.140 приёмников x5:10:1395,1001 отсчёт Vx/Vy,FP64. p64s непосредственно рассчитывает только четыре источника100/300/500/700 м на полной сетке. Источники900/1100/1300 не вычисляются и не сохраняются; для плоской модели их можно восстановить зеркально, с изменением знака Vx.

## Точность

Выполнены21 полных запуска обёртки на EFVB00000/00001/02999/02994; внутри семи p6-запусков выполнены28 одиночных интегрирований. Для EFVB00001 —прогрев и3 чередующихся повтора каждого решателя. Всего220 проверок и27 SHA массивов. Оба компонента каждой сейсмограммы сравниваются отдельно; прогрев дополнительно сравнивает все24 строки U/V/A и16 строк памяти PML. Максимальные относительные L2:

- Сейсмограммы p64s против p6/p67s: {maxima['surface']:.6g}.
- Зеркальные пары, включая центральный источник: {maxima['mirror']:.6g}.
- Конечное состояние: {maxima['state.npy']:.6g}.
- Память PML: {maxima['pml_history.npy']:.6g}.

Результаты согласованы в пределах округления, побитовое равенство не требуется. Это эквивалентность реализаций, не независимая валидация дифференциальной задачи и не новая сеточная сходимость. Родной шаг совпал у всех решателей: EFVB0 —10000 шагов, EFVB1 —9000, EFVB2999 —8000, EFVB2994 —4000. Последняя модель имеет наименьший maxVp среди первых3000; контроль покрывает наиболее крупный ожидаемый dt. Все терминальные U/V/A и память PML конечны.

## Производительность

![Время A100](timings.png)

| Вариант | Источников | Run, с | Полный замер, с |
|---|---:|---:|---:|
"""
    for v,shots in zip(VARIANTS,(4,4,7)):
        t=mapping(timings[v]);text+=f"| {v} | {shots} | {number(t['run_seconds']):.6f} | {number(t['total_seconds']):.6f} |\n"
    text+=f"""
Для требуемых четырёх источников p64s быстрее четырёх одиночных p6 в{p6/four:.3f} раза. Относительно полного пакета семи источников p67s —в{seven/four:.3f} раза, но это разный объём работы. На непосредственно рассчитанный источник приходится{four/4:.3f} с у p64s и{seven/7:.3f} с у p67s: основной выигрыш относительно p67s связан с исключением трёх зеркальных опытов, а не с ускорением одного состояния. Приёмник асинхронный, его очередь полностью дожидается завершения; SEG-Y и полевых кадров нет. Времена полного замера включают проверку и сохранение, прогревы с дополнительными полными состояниями исключены из медиан.

## Производственный запуск

CPU-аудит всех3000 EFVB прошёл: плоские слои, положительный объёмный модуль, корректные узловые источники и140 приёмников. Перед каждой моделью исходные VP/VS сверяются с SHA этого аудита. Новый отдельный чанк —EFVB00000–02999, четыре источника, массив velocity.npy[1001,4,140,2] и input.npz с исходными VP/VS. Чистые скорости занимают25.06ГиБ, с входами и метаданными около25.3ГиБ; резервируется10МиБ на модель. Общий SQLite DiskGuard сохраняет20ГиБ неснижаемого остатка и20ГиБ рабочего бюджета, учитывает обе GPU и останавливает выдачу при ошибке. Failed/stale автоматически не снимаются, результаты не перезаписываются, следующий чанк не включается.

Фактическое состояние запуска и оценка завершения фиксируются в [снимке очереди](../../../data/dev_1_6_p64s_production/chunk01_v1/launch_snapshot.json); это снимок, а не живой статус. Корень на Saturn: /home/antonov/SeismicDataset/data/dev_1_6_p64s_production/chunk01_v1. Производственные массивы в Git не включаются.

[Метрики сравнения](../../../data/dev_1_6_p64s_production/comparison_v1/analysis/metrics.json) · [Аудит3000 моделей](../../../data/dev_1_6_p64s_production/cpu_v1/check.json) · [Результат очистки](../../../data/dev_1_6_p64s_production/cleanup_v1/completed.json).
"""
    base=root.parent
    if (base/'chunk01_v1/launch_snapshot.json').exists():
        clean=read_json(base/'cleanup_v1/completed.json');pilot_record=read_json(base/'dual_gpu_pilot_v1/completed.json');live=read_json(base/'chunk01_v1/launch_snapshot.json')
        text+=f"\nУдалено{number(clean['planned_allocated_bytes'])/2**30:.2f}ГиБ старых файлов; после очистки свободно{number(clean['available_after_bytes'])/2**30:.2f}ГиБ. Метаданные старых очередей архивированы и проверены до удаления; исходники, галереи и последние визуальные серии сохранены. Пилот обеих A100:6 моделей за{number(pilot_record['wall_seconds']):.3f}с.\n\nЧанк запущен{live['started_utc']}. Снимок{live['as_of_utc']}: {number(live['completed']):.0f} завершены,{number(live['active']):.0f} активны,{number(live['failed']):.0f} ошибок. Среднее полное время{number(live['mean_model_total_seconds']):.3f}с на модель; прогноз окончания{live['estimated_finish_utc']} (UTC), если скорость сохранится. Проекция файлов чанка{number(live['projected_chunk_bytes'])/2**30:.2f}ГиБ. Это не утверждение о завершении всех3000.\n"
    (out/'README.md').write_text(text)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--analyze',action='store_true');p.add_argument('--extra',action='store_true');p.add_argument('--report',type=Path);a=p.parse_args();report(a.root,a.report) if a.report else (analyze(a.root) if a.analyze else (extra(a.root) if a.extra else series(a.root)))
