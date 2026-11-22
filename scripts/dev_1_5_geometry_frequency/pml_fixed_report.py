"""Validate fixed/generic equivalence and separate order/time errors for PML350."""
from __future__ import annotations
import argparse
from pathlib import Path
import html
import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import ROOT, CASES, CONTROLS, BATCH_ROOT, SOURCE_POSITIONS, SURFACE_BATCH_ROOT
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.reporting import read_json, mapping, number

FloatArray=NDArray[np.float64]


def surface(path:Path)->FloatArray:
    with np.load(path/'common_surface.npz') as z:
        np.testing.assert_array_equal(z['x_m'],np.arange(1401))
        np.testing.assert_array_equal(z['time_s'],np.arange(1001)*.001)
        return np.asarray(z['velocity_m_s'],dtype=np.float64)


def equivalence(a:FloatArray,b:FloatArray)->dict[str,object]:
    sums=np.zeros(2); refs=np.zeros(2); peaks=np.zeros(2); refpeaks=np.zeros(2)
    for av,bv in zip(a,b,strict=True):
        delta=av-bv
        sums+=np.sum(delta*delta,axis=0);refs+=np.sum(bv*bv,axis=0)
        peaks=np.maximum(peaks,np.max(abs(delta),axis=0));refpeaks=np.maximum(refpeaks,np.max(abs(bv),axis=0))
    rel=np.sqrt(sums/np.maximum(refs,1e-300)); peakrel=peaks/np.maximum(refpeaks,1e-300)
    return {'relative_l2_components':rel.tolist(),'max_abs_delta_m_s':peaks.tolist(),'peak_relative_components':peakrel.tolist(),'within_tolerance':bool(np.all(rel<1e-8) and np.all(peakrel<1e-7))}


def error(a:FloatArray,b:FloatArray)->dict[str,object]:
    t=np.arange(1001)*.001; x=np.arange(1401)
    dt=a-b;result:dict[str,object]={}
    for name,maskx in [('interior',(x>0)&(x<1400)),('outside20m',(x>0)&(x<1400)&(abs(x-700)>20))]:
        for win,start in [('full',0.),('after015',.15),('late',.5)]:
            delta=dt[t>=start][:,maskx];ref=b[t>=start][:,maskx]
            energy=float(.001*np.sum(delta**2));reference=float(.001*np.sum(ref**2))
            result[name+'_'+win]={'energy_proxy':energy,'relative_l2':float(np.sqrt(energy/max(reference,1e-300))),'peak_vector_m_s':float(np.max(np.linalg.norm(delta,axis=-1))),'relative_l2_components':np.sqrt(np.sum(delta**2,axis=(0,1))/np.maximum(np.sum(ref**2,axis=(0,1)),1e-300)).tolist()}
    center=abs(x-700)<=200;arrival=.05+np.sqrt(700**2+(x-700)**2)/4087
    gate=center[None,:]&(abs(t[:,None]-arrival[None,:])<=.03)
    delta=dt[gate];ref=b[gate]
    result['first_PP']={'relative_l2':float(np.linalg.norm(delta)/np.linalg.norm(ref)),'peak_delta_to_reference_peak':float(np.max(np.linalg.norm(delta,axis=1))/np.max(np.linalg.norm(ref,axis=1)))}
    return result


def analyze(root:Path)->None:
    assert read_json(root/'runs_completed.json')['status']=='completed'
    out=root/'analysis';out.mkdir(exist_ok=False)
    verified=[]
    for path in sorted(root.glob('*/*/*/result.json')):
        r=read_json(path);assert r['status']=='completed' and r['boundary_elements']==0 and r['ABC'] is False
        for name in ('velocity.npy','nodes.npy','surface.npz','common_surface.npz'):
            assert file_record(path.parent/name)['sha256']==mapping(r[name])['sha256']
            verified.append(str((path.parent/name).relative_to(root)))
    assert len(verified)==43*4
    pairs=[];timings=[];cases:dict[str,object]={}
    refroot=root/CONTROLS[-1].label/'fixed'
    common_ref=surface(refroot/'common_dt');fine_ref=surface(refroot/'half_common_dt')
    for case in CASES:
        for backend in ('generic','fixed'):
            rows=[read_json(root/case.label/backend/f'repeat{i}/result.json') for i in range(3)]
            times=[number(r['run_seconds']) for r in rows]
            timings.append({'case':case.label,'backend':backend,'dt_s':rows[0]['dt_s'],'steps':rows[0]['steps'],'median_run_seconds':float(np.median(times)),'run_seconds':times,'prepare_seconds':[r['prepare_seconds'] for r in rows],'nodes':rows[0]['nodes_count']})
        for i in range(3):
            generic=root/case.label/'generic'/f'repeat{i}';fixed=root/case.label/'fixed'/f'repeat{i}'
            gr=read_json(generic/'result.json');fr=read_json(fixed/'result.json')
            assert gr['dt_s']==fr['dt_s'] and gr['steps']==fr['steps']
            np.testing.assert_array_equal(np.load(generic/'nodes.npy'),np.load(fixed/'nodes.npy'))
            field=equivalence(np.load(fixed/'velocity.npy',mmap_mode='r'),np.load(generic/'velocity.npy',mmap_mode='r'))
            with np.load(fixed/'surface.npz') as a,np.load(generic/'surface.npz') as b:
                surf=equivalence(a['velocity_m_s'],b['velocity_m_s'])
            pairs.append({'case':case.label,'repeat':i,'fields':field,'surface':surf})
        a=surface(root/case.label/'fixed/repeat0');half=surface(root/case.label/'fixed/half_dt');common=surface(root/case.label/'fixed/common_dt')
        r=read_json(root/case.label/'fixed/repeat0/result.json')
        cases[case.label]={'native_vs_fine_control':error(a,fine_ref),'common_dt_vs_p9_common':error(common,common_ref),'native_vs_half_dt':error(a,half),'half_dt_vs_fine_control':error(half,fine_ref),'pml_peaks_native':r['pml_peak_m_s'],'core_kinetic_energy_native':r['core_kinetic_energy'],'native_reproducibility':equivalence(surface(root/case.label/'fixed/repeat1'),a)}
    order_pairs=[]
    for i,a_case in enumerate(CASES):
        for b_case in CASES[i+1:]:
            order_pairs.append({'a':a_case.label,'reference_b':b_case.label,'native_steps':error(surface(root/a_case.label/'fixed/repeat0'),surface(root/b_case.label/'fixed/repeat0')),'common_step':error(surface(root/a_case.label/'fixed/common_dt'),surface(root/b_case.label/'fixed/common_dt'))})
    controls={'p8_vs_p9_common':error(surface(root/CONTROLS[0].label/'fixed/common_dt'),common_ref),'p9_common_vs_half':error(common_ref,fine_ref)}
    agree=all(mapping(p[k])['within_tolerance'] for p in pairs for k in ('fields','surface'))
    write(out/'metrics.json',{'status':'completed','timings':timings,'generic_fixed_comparisons':pairs,'generic_fixed_agree':agree,'cases':cases,'order_pairs':order_pairs,'controls':controls,'step_plan':read_json(root/'step_plan.json'),'verified_array_files':verified,'reference':'p9/h5 common_dt/2; controls p8/h5 at common_dt and p9 half-step; not exact PDE truth','sampling':'SEM interpolation to same x=0..1400 in1m steps; use strict interior, report outside20m separately; integration uniform1m*1ms','first_PP':'central+-200m; +-0.03s around .05+sqrt(700^2+(x-700)^2)/4087; raw diagnostic window, not phase isolation','timing':'median3 full runs after warmup; includes surface and21 full field transfers; volume differs with p/h, retained for full-task benchmark','boundary':'noABC and no1D boundary stratum, all edges natural free; PML350 quadratic fixedmax173.088611'})
    assert agree,'Fixed and generic disagree: inspect metrics before accepting timing'


def cast_list(value:object)->list[object]:
    if not isinstance(value,list):raise TypeError('Expected JSON list')
    return value


def render(root:Path,output:Path)->None:
    m=read_json(root/'analysis/metrics.json');output.mkdir(parents=True,exist_ok=False)
    rows=m['timings'];assert isinstance(rows,list)
    labels=[c.label for c in CASES]
    generic=[];fixed=[];table=[]
    for label in labels:
        gr=next(mapping(r) for r in rows if mapping(r)['case']==label and mapping(r)['backend']=='generic')
        fr=next(mapping(r) for r in rows if mapping(r)['case']==label and mapping(r)['backend']=='fixed')
        gt,ft=number(gr['median_run_seconds']),number(fr['median_run_seconds']);generic.append(gt);fixed.append(ft)
        case=mapping(mapping(m['cases'])[label])
        native=mapping(mapping(case['native_vs_fine_control'])['outside20m_full'])
        common=mapping(mapping(case['common_dt_vs_p9_common'])['outside20m_full'])
        pp=mapping(mapping(case['native_vs_fine_control'])['first_PP'])
        table.append(f'<tr><td>{html.escape(label)}</td><td>{number(fr["dt_s"])*1e6:.3f}</td><td>{gt:.4f}</td><td>{ft:.4f}</td><td>{gt/ft:.3f}</td><td>{number(native["relative_l2"])*100:.6g}</td><td>{number(common["relative_l2"])*100:.6g}</td><td>{number(pp["peak_delta_to_reference_peak"])*100:.6g}</td></tr>')
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4)
    ax.bar(x-.18,generic,.36,label='Общий seismic');ax.bar(x+.18,fixed,.36,label='Фиксированный порядок')
    ax.set(xticks=x,xticklabels=labels,ylabel='Медиана run, с',title='A100: PML350 без ABC, родной CFL0,8, 1 с');ax.legend();fig.tight_layout();fig.savefig(output/'timings.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key,title in zip(axes,('native_vs_fine_control','common_dt_vs_p9_common'),('Родные dt против p9/5 м на малом шаге','Общий dt: различие пространственных порядков'),strict=True):
        vals=[number(mapping(mapping(mapping(mapping(m['cases'])[label])[key])['outside20m_full'])['relative_l2'])*100 for label in labels]
        ax.bar(labels,vals);ax.set_yscale('log');ax.set(ylabel='Относительная L2-разность, %',title=title)
    fig.suptitle('Верх основной области вне20 м от источника, 0–1 с');fig.tight_layout();fig.savefig(output/'accuracy.png',dpi=150);plt.close(fig)
    body='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>PML350: порядок, точность, время</title><style>body{font:17px sans-serif;max-width:1250px;margin:35px auto;padding:15px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccc}img{max-width:100%}</style><h1>PML350 без ABC: фиксированные порядки на A100</h1><p>Двухслойная1400×700 м, центральный Рикер30 Гц вниз,1 с, β=1/27. Нет одномерных граничных элементов. Native CFL0,8 полной PML-модели; p4/p5 h5 м, p6/p7 h10 м.</p><p>Время — медиана трёх повторов после прогрева, включает запись1001 поверхностного отсчёта и21 полного поля. Объём полей различается с порядком и сеткой. Сравнение общего и фиксированного решателя выполнено при одинаковом dt внутри каждой пары.</p><table><tr><th>Вариант</th><th>dt, мкс</th><th>Общий, с</th><th>Фикс., с</th><th>Ускорение</th><th>L2 родной dt, %</th><th>L2 общий dt, %</th><th>Первое PP: пик Δ / пик контроля, %</th></tr>'''+''.join(table)+'''</table><img src="timings.png"><img src="accuracy.png"><p>Точность: одинаковые точки x=0..1400 м через1 м, полиномиальная SEM-интерполяция. Основные числа: строго внутри области, вне20 м от источника. Родные шаги сравниваются с p9/5 м при половине общего шага; общие — с p9/5 м при том же dt. Контроль не является точным PDE-решением. Неопределённость контроля оценена p8/p9 и dt/2; их метрики, попарные сравнения и окна0,15–1/0,5–1 с сохранены в JSON.</p><p>Первое PP: центральная полоса±200 м, окно±0,03 с по гиперболе; нормировка на пик этого окна, без модового разложения.</p><p><a href="../../../data/dev_1_5_geometry_frequency/pml_fixed_orders_v1/analysis/metrics.json">Все метрики и проверки</a></p></html>'''
    controls=mapping(m['controls'])
    spatial=number(mapping(mapping(controls['p8_vs_p9_common'])['outside20m_full'])['relative_l2'])*100
    temporal=number(mapping(mapping(controls['p9_common_vs_half'])['outside20m_full'])['relative_l2'])*100
    comparisons=m['generic_fixed_comparisons'];assert isinstance(comparisons,list)
    max_equivalence=max(number(value) for pair in comparisons for key in ('fields','surface') for value in cast_list(mapping(mapping(pair)[key])['relative_l2_components']))
    body=body.replace('</table>',f'</table><p>Проверка контроля: L2 p8/5 м против p9/5 м при общем dt — <b>{spatial:.6g}%</b>; p9 при общем dt против dt/2 — <b>{temporal:.6g}%</b>. Те же внутренние приёмники вне20 м, вся1 с. Максимальная относительная L2-разность компонент fixed/generic по всем полям и сейсмограммам трёх повторов: <b>{max_equivalence:.3g}</b>.</p>')
    (output/'index.html').write_text(body)
    write(output/'render.json',{'status':'completed','metrics':file_record(root/'analysis/metrics.json'),'files':{p.name:file_record(p) for p in output.iterdir() if p.is_file()}})


def review(root:Path,output:Path)->None:
    m=read_json(root/'analysis/metrics.json')
    labels=[c.label for c in CASES]
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,key,title in zip(axes,('native_vs_fine_control','common_dt_vs_p9_common'),('Родные dt против p9/5 м на малом шаге','Общий dt: различие порядков'),strict=True):
        values=[number(mapping(mapping(mapping(mapping(m['cases'])[label])[key])['outside20m_full'])['relative_l2'])*100 for label in labels]
        if key=='native_vs_fine_control':
            ax.bar(labels,values);ax.set_ylim(0,max(values)*1.2)
            for i,value in enumerate(values):ax.text(i,value+max(values)*.02,f'{value:.5f}',ha='center')
        else:
            ax.plot(labels,values,'o',markersize=8);ax.set_yscale('log');ax.set_ylim(min(values)*.5,max(values)*2)
            for i,value in enumerate(values):ax.annotate(f'{value:.3g}',(i,value),xytext=(0,12),textcoords='offset points',ha='center')
        ax.set(ylabel='Относительная L2-разность, %',title=title);ax.grid(axis='y',alpha=.2)
    fig.suptitle('Верх основной области вне20 м от источника,0–1 с');fig.tight_layout();fig.savefig(output/'accuracy_reviewed.png',dpi=150);plt.close(fig)
    rows=[]
    for label in labels:
        c=mapping(mapping(m['cases'])[label])
        native=number(mapping(mapping(c['native_vs_fine_control'])['interior_full'])['relative_l2'])*100
        common=number(mapping(mapping(c['common_dt_vs_p9_common'])['interior_full'])['relative_l2'])*100
        rows.append(f'<tr><td>{label}</td><td>{native:.5f}%</td><td>{common:.5f}%</td></tr>')
    near='''<h2>Что меняется рядом с точечным источником</h2><p>В основной таблице исключены20 м около источника. Если взять весь верх строго внутри0&lt;x&lt;1400, включая источник, за0–1 с, разность становится существенно больше:</p><table><tr><th>Вариант</th><th>L2, родной шаг против малого p9</th><th>L2, общий шаг против p9</th></tr>'''+''.join(rows)+'''</table><p>У самого контроля p8/p9 эта полная норма различается на0,64576%. Пространственная сходимость непосредственно у источника не установлена. Малые значения вне20 м не распространяются на ближайшие трассы; позднее окно0,15–1 с всех внутренних приёмников также сохранено в JSON.</p><h2>Вывод для выбранной двухслойной модели</h2><p>p6/10 м быстрее p4/5 м в1,72 раза при практически одинаковой удалённой точности на родном dt. p7/10 м занимает почти то же время, что p4/5 м, но уменьшает удалённую L2-разность примерно в1,78 раза. p5/5 м даёт наименьшую разность на родном dt, однако медленнее p7/10 м в2,11 раза. На общем шаге p7/10 м и p5/5 м очень близки к p9; основная удалённая разность рабочих конфигураций определяется временной дискретизацией. В окне первого PP пространственная разность не монотонна по порядку; все значения приведены в основной таблице и JSON. Эти выводы относятся к данной плоской двухслойной модели, не заменяют проверку сложных границ EFVB/ECVB.</p><p>Все43 запуска завершены; проверены172 SHA массивов. Поверхность и полные поля совпадают на общих временах. Поля и память PML конечны, кинетическая энергия основной области к концу записи убывает. Это наблюдение за1 с, не доказательство долгосрочной устойчивости.</p>'''
    body=(output/'index.html').read_text().replace('accuracy.png','accuracy_reviewed.png').replace('</html>',near+'</html>')
    (output/'reviewed.html').write_text(body)
    write(output/'review.json',{'status':'completed','source_metrics':file_record(root/'analysis/metrics.json'),'files':{n:file_record(output/n) for n in ('accuracy_reviewed.png','reviewed.html')}})

def batch_analyze(root:Path)->None:
    done=read_json(root/'runs_completed.json');assert done['status']=='completed'
    save_fields=number(read_json(root/'plan.json')['field_frames_per_source'])>0
    comparison_keys=('fields','surface') if save_fields else ('surface',)
    out=root/'analysis';out.mkdir(exist_ok=False)
    runroot=root/'p6_h10/fixed';verified=[]
    for path in sorted(runroot.glob('*/result.json')):
        record=read_json(path);assert record['status']=='completed' and record['boundary_elements']==0
        assert record['frames']==(21 if save_fields else 0)
        if save_fields:assert record['surface_field_exact'] is True
        else:
            assert record['field_receiver_registered'] is False
            assert not any((path.parent/name).exists() for name in ('velocity.npy','nodes.npy','common_surface.npz'))
        for name in (('velocity.npy','nodes.npy','surface.npz','common_surface.npz') if save_fields else ('surface.npz',)):
            assert file_record(path.parent/name)['sha256']==mapping(record[name])['sha256']
            verified.append(str((path.parent/name).relative_to(root)))
    assert len(verified)==(128 if save_fields else 32)
    comparisons=[];rounds=[]
    for repeat in range(3):
        label=f'repeat{repeat}';packed=runroot/f'batch7_{label}';pr=read_json(packed/'result.json');n=int(number(pr['nodes_per_copy']));top=int(number(pr['surface_nodes_per_copy']))
        assert pr['copies']==7 and pr['source_x_m']==list(SOURCE_POSITIONS)
        pv=np.load(packed/'velocity.npy',mmap_mode='r') if save_fields else None;pn=np.load(packed/'nodes.npy',mmap_mode='r') if save_fields else None
        with np.load(packed/'surface.npz') as pz:
            ps=pz['velocity_m_s'];pc=pz['coordinates_m'];pt=pz['time_s']
            singles=[]
            for i,x in enumerate(SOURCE_POSITIONS):
                path=runroot/f'x{int(x)}_{label}';single=read_json(path/'result.json');singles.append(single)
                assert single['copies']==1 and single['source_x_m']==[x] and single['dt_s']==pr['dt_s'] and single['steps']==pr['steps']
                assert single['native']==pr['native'] and single['snapshot']==pr['snapshot'] and single['runner']==pr['runner']
                fields:dict[str,object]={}
                if save_fields:
                    assert pv is not None and pn is not None
                    np.testing.assert_array_equal(pn[i*n:(i+1)*n],np.load(path/'nodes.npy'))
                    fields=equivalence(pv[:,i*n:(i+1)*n],np.load(path/'velocity.npy',mmap_mode='r'))
                with np.load(path/'surface.npz') as sz:
                    np.testing.assert_array_equal(pc[i*top:(i+1)*top],sz['coordinates_m']);np.testing.assert_array_equal(pt,sz['time_s'])
                    surf=equivalence(ps[:,i*top:(i+1)*top],sz['velocity_m_s'])
                comparisons.append({'repeat':repeat,'source_x_m':x,'surface':surf,**({'fields':fields} if save_fields else {})})
            sequential=sum(number(s['run_seconds']) for s in singles);packed_run=number(pr['run_seconds'])
            sequential_wall=sum(number(s['solve_wall_seconds']) for s in singles);packed_wall=number(pr['solve_wall_seconds'])
            rounds.append({'repeat':repeat,'single_sources_run_seconds':[s['run_seconds'] for s in singles],'single_sources_assembly_seconds':[s['assembly_seconds'] for s in singles],'single_sources_prepare_seconds':[s['prepare_seconds'] for s in singles],'sequential_sum_run_seconds':sequential,'batch_run_seconds':packed_run,'run_speedup':sequential/packed_run,'sequential_sum_solve_wall_seconds':sequential_wall,'batch_solve_wall_seconds':packed_wall,'solve_wall_speedup':sequential_wall/packed_wall,'batch_assembly_seconds':pr['assembly_seconds'],'batch_prepare_seconds':pr['prepare_seconds']})
    agree=all(mapping(c[key])['within_tolerance'] for c in comparisons for key in comparison_keys)
    seq=float(np.median([number(r['sequential_sum_run_seconds']) for r in rounds]));packed_time=float(np.median([number(r['batch_run_seconds']) for r in rounds]))
    seq_wall=float(np.median([number(r['sequential_sum_solve_wall_seconds']) for r in rounds]));pack_wall=float(np.median([number(r['batch_solve_wall_seconds']) for r in rounds]))
    prior_comparisons=[];prior_sha=[];prior_metrics:dict[str,object]={}
    if not save_fields:
        oldroot=BATCH_ROOT/'p6_h10/fixed'
        prior_metrics=read_json(BATCH_ROOT/'analysis/metrics.json')
        for path in sorted(runroot.glob('*/result.json')):
            previous=oldroot/path.parent.name
            old=read_json(previous/'result.json');new=read_json(path)
            assert old['dt_s']==new['dt_s'] and old['source_x_m']==new['source_x_m'] and old['native']==new['native'] and old['snapshot']==new['snapshot']
            assert file_record(previous/'surface.npz')['sha256']==mapping(old['surface.npz'])['sha256']
            prior_sha.append(str(previous/'surface.npz'))
            with np.load(path.parent/'surface.npz') as av,np.load(previous/'surface.npz') as bv:
                np.testing.assert_array_equal(av['time_s'],bv['time_s']);np.testing.assert_array_equal(av['coordinates_m'],bv['coordinates_m'])
                diff=equivalence(av['velocity_m_s'],bv['velocity_m_s'])
            prior_comparisons.append({'run':path.parent.name,'surface':diff})
        agree=agree and all(mapping(c['surface'])['within_tolerance'] for c in prior_comparisons)
    write(out/'metrics.json',{'status':'completed','source_x_m':list(SOURCE_POSITIONS),'copies':7,'order':6,'h_m':10,'boundary_elements':0,'ABC':False,'PML_thickness_m':350,'step_plan':read_json(root/'step_plan.json'),'rounds':rounds,'median_sequential_run_seconds':seq,'median_batch_run_seconds':packed_time,'run_speedup':seq/packed_time,'run_time_reduction_fraction':1-packed_time/seq,'equivalent_batch_seconds_per_source':packed_time/7,'median_sequential_solve_wall_seconds':seq_wall,'median_batch_solve_wall_seconds':pack_wall,'solve_wall_speedup':seq_wall/pack_wall,'comparisons':comparisons,'all_agree':agree,'verified_array_files':verified,'output_mode':'fields_and_surface' if save_fields else 'surface_only','field_frames_per_source':21 if save_fields else 0,'with_fields_baseline':{k:prior_metrics[k] for k in ('median_sequential_run_seconds','median_batch_run_seconds','median_sequential_solve_wall_seconds','median_batch_solve_wall_seconds')} if prior_metrics else None,'with_fields_baseline_metrics':file_record(BATCH_ROOT/'analysis/metrics.json') if prior_metrics else None,'vs_with_fields':prior_comparisons,'verified_prior_surface_files':prior_sha,'sampling':'1001 full-top nodal samples per source; no mixing between copies; full fields optional as explicitly recorded' ,'timing':'median3 sums of7 sequential run times vs median3 package run times; 8 complete warmup launches excluded. Solve wall includes assembly/preparation/run, excludes process startup and postprocessing/hashing.','physical_interpretation':'7 independent disconnected meshes in one SGModel; not7 sources superposed in one physical field','native_reference':'same frozen fixed-p6 snapshot as prior single-source study; production stopped'})
    assert agree,'Packed and single-source results differ beyond tolerance'


def batch_render(root:Path,output:Path)->None:
    m=read_json(root/'analysis/metrics.json')
    if m.get('output_mode')=='surface_only':surface_batch_render(root,output);return
    output.mkdir(parents=True,exist_ok=False)
    seq=number(m['median_sequential_run_seconds']);packed=number(m['median_batch_run_seconds'])
    fig,ax=plt.subplots(figsize=(9,5));bars=ax.bar(['7 источников последовательно','7 независимых копий в одной модели'],[seq,packed],color=['#777777','#287ba5'])
    ax.set(ylabel='Медиана времени run всей серии, с',title='A100 · p6/10 м · PML350 без ABC · 1400×700 м · 30 Гц · 1 с',ylim=(0,max(seq,packed)*1.2))
    for bar,value in zip(bars,[seq,packed],strict=True):ax.text(bar.get_x()+bar.get_width()/2,value+max(seq,packed)*.02,f'{value:.3f} с',ha='center')
    fig.tight_layout();fig.savefig(output/'timings.png',dpi=150);plt.close(fig)
    comparisons=cast_list(m['comparisons']);maximum=max(number(v) for row in comparisons for key in ('fields','surface') for v in cast_list(mapping(mapping(row)[key])['relative_l2_components']))
    dt=number(mapping(m['step_plan'])['batch_dt_s'])*1e6
    rows=[]
    for row in cast_list(m['rounds']):
        r=mapping(row);rows.append(f'<tr><td>{r["repeat"]}</td><td>{number(r["sequential_sum_run_seconds"]):.4f}</td><td>{number(r["batch_run_seconds"]):.4f}</td><td>{number(r["run_speedup"]):.3f}</td></tr>')
    body=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>p6: семь источников в одной модели</title><style>body{{font:17px sans-serif;max-width:1100px;margin:35px auto;padding:15px}}td,th{{padding:9px;border-bottom:1px solid #ccc}}table{{border-collapse:collapse;width:100%}}img{{max-width:100%}}</style><h1>Семь независимых источников: пакет против последовательной серии</h1><p>Двухслойная1400×700 м, p6/h10 м, центральная и смещённые точечные силы вниз x=175,350,525,700,875,1050,1225 м. Рикер30 Гц,1 с, PML350 м с квадратичным dmax173,088611 с⁻¹, без ABC и без граничных элементов, β=1/27. Использованы7 несвязанных копий сетки с одинаковыми координатами и разными номерами узлов в одном SGModel. Это отдельные поля, не суммарная волна семи источников.</p><p>Родной шаг CFL0,8 одиночной сетки и пакета совпал: <b>{dt:.6f} мкс</b>. В каждой копии сохраняются1001 отсчёт Vx/Vy всех узлов верха и21 полное поле. Поэтому объём полезного результата одинаков.</p><p>Медианы трёх повторов после полного прогрева: последовательно <b>{seq:.4f} с</b>; пакет <b>{packed:.4f} с</b>; ускорение <b>{number(m['run_speedup']):.3f} раза</b>. Эквивалентное время пакета на источник <b>{number(m['equivalent_batch_seconds_per_source']):.4f} с</b>. Последовательное время — сумма семи измерений, а не7× центральный источник.</p><img src="timings.png"><table><tr><th>Повтор</th><th>Сумма7 одиночных run, с</th><th>Пакет run, с</th><th>Ускорение</th></tr>{''.join(rows)}</table><p>С учётом построения модели и prepare, до окончания run: последовательная сумма {number(m['median_sequential_solve_wall_seconds']):.4f} с, пакет {number(m['median_batch_solve_wall_seconds']):.4f} с, ускорение {number(m['solve_wall_speedup']):.3f}. Это не включает запуск Python, постобработку и хеширование. Дисковая запись использует системный кэш; тест не является бенчмарком накопителя.</p><p>Все21 пары «источник×повтор» проверены по полям и сейсмограммам. Максимальная относительная L2 компонент: <b>{maximum:.3g}</b>. Все массивы конечны, поверхность совпадает с полями на общих временах. Проверены128 SHA файлов32 запусков. Изолированность копий дополнительно проверена по индексам элементов и источников.</p><p><a href="../../../data/dev_1_5_geometry_frequency/pml_p6_sources7_v1/analysis/metrics.json">Полные метрики</a></p></html>'''
    (output/'index.html').write_text(body)
    write(output/'render.json',{'status':'completed','source_metrics':file_record(root/'analysis/metrics.json'),'files':{n:file_record(output/n) for n in ('timings.png','index.html')}})

def surface_batch_render(root:Path,output:Path)->None:
    m=read_json(root/'analysis/metrics.json');output.mkdir(parents=True,exist_ok=False)
    old=mapping(m['with_fields_baseline'])
    keys=('median_sequential_run_seconds','median_batch_run_seconds')
    before=[number(old[k]) for k in keys];after=[number(m[k]) for k in keys]
    fig,ax=plt.subplots(figsize=(10,5));x=np.arange(2)
    ax.bar(x-.18,before,.36,label='Сейсмограммы +21 кадр',color='#888888')
    ax.bar(x+.18,after,.36,label='Только сейсмограммы',color='#287ba5')
    for i in range(2):
        ax.text(i-.18,before[i]+.3,f'{before[i]:.3f}',ha='center');ax.text(i+.18,after[i]+.3,f'{after[i]:.3f}',ha='center')
    ax.set(xticks=x,xticklabels=['7 источников последовательно','Пакет7 в одной модели'],ylabel='Медиана run всей серии, с',ylim=(0,max(*before,*after)*1.25),title='A100 · p6/10 м · 1400×700 м · 30 Гц · 1 с · PML350 без ABC')
    ax.legend();fig.tight_layout();fig.savefig(output/'timings.png',dpi=150);plt.close(fig)
    maximum=max(number(v) for c in cast_list(m['comparisons']) for v in cast_list(mapping(mapping(c)['surface'])['relative_l2_components']))
    prior_max=max(number(v) for c in cast_list(m['vs_with_fields']) for v in cast_list(mapping(mapping(c)['surface'])['relative_l2_components']))
    rows=[]
    for row in cast_list(m['rounds']):
        r=mapping(row);rows.append(f'<tr><td>{r["repeat"]}</td><td>{number(r["sequential_sum_run_seconds"]):.4f}</td><td>{number(r["batch_run_seconds"]):.4f}</td><td>{number(r["run_speedup"]):.3f}</td></tr>')
    body=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>p6: только сейсмограммы</title><style>body{{font:17px sans-serif;max-width:1100px;margin:35px auto;padding:15px}}td,th{{padding:9px;border-bottom:1px solid #ccc}}table{{border-collapse:collapse;width:100%}}img{{max-width:100%}}</style><h1>Семь источников: только сейсмограммы</h1><p>Двухслойная1400×700 м, p6/h10 м, PML350 м без ABC и без граничных элементов, β=1/27,30 Гц,1 с. Источники x=175,350,525,700,875,1050,1225 м. Native CFL0,8, dt={number(mapping(m['step_plan'])['batch_dt_s'])*1e6:.6f} мкс в одиночной и пакетной модели. Пакет содержит7 несвязанных копий, поля не суммируются.</p><p>Полный полевой приёмник не зарегистрирован: отсутствуют GPU→CPU-передачи кадров, обработчик полей и velocity.npy. Сохраняется только surface.npz:1001×1261×2 float64 на источник, все узлы верхней грани, включая PML. Число приёмников не уменьшалось. Изменён только режим регистрации.</p><p>Медианы трёх повторов после прогрева: сумма семи одиночных <b>{after[0]:.4f} с</b>, один пакет <b>{after[1]:.4f} с</b>, ускорение пакета <b>{number(m['run_speedup']):.3f} раза</b>, эквивалентно <b>{number(m['equivalent_batch_seconds_per_source']):.4f} с/источник</b>.</p><p>По сравнению с предыдущей серией с кадрами: сумма одиночных {before[0]:.4f}→{after[0]:.4f} с ({100*(1-after[0]/before[0]):.2f}% экономии); пакет {before[1]:.4f}→{after[1]:.4f} с ({100*(1-after[1]/before[1]):.2f}% экономии). Режимы с/без кадров измерены в отдельных последовательных сериях, а не в чередующихся парах; небольшие различия включают вариацию времени между сериями.</p><img src="timings.png"><table><tr><th>Повтор</th><th>Сумма7 одиночных run, с</th><th>Пакет run, с</th><th>Ускорение</th></tr>{''.join(rows)}</table><p>С построением модели и prepare, до окончания run: {number(m['median_sequential_solve_wall_seconds']):.4f} с последовательно и {number(m['median_batch_solve_wall_seconds']):.4f} с пакетом. Run включает регистрацию сейсмограмм в память; сериализация surface.npz, хеширование и запуск Python измерениями run не охватываются.</p><p>Проверена21 пара «источник×повтор» пакета и одиночных, максимальная относительная L2 компонент <b>{maximum:.3g}</b>. Все32 новые сейсмограммы также сравнивались с соответствующими сохранёнными запусками с кадрами: максимум <b>{prior_max:.3g}</b>. Проверены32 новых и32 прежних SHA поверхностных файлов. Полный полевой приёмник отсутствовал; прежние поля сохранены неизменными. Новые сейсмограммы и конечная память PML конечны; новых наблюдений поля во всех узлах нет.</p><p><a href="../../../data/dev_1_5_geometry_frequency/pml_p6_sources7_surface_v1/analysis/metrics.json">Все метрики и проверки</a></p></html>'''
    (output/'index.html').write_text(body)
    write(output/'render.json',{'status':'completed','source_metrics':file_record(root/'analysis/metrics.json'),'files':{n:file_record(output/n) for n in ('timings.png','index.html')}})

def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--analyze',action='store_true');p.add_argument('--render',type=Path);p.add_argument('--review',type=Path);p.add_argument('--surface-only',action='store_true');p.add_argument('--batch-analyze',action='store_true');p.add_argument('--batch-render',type=Path);a=p.parse_args()
    if (a.batch_analyze or a.batch_render) and a.root==ROOT:a.root=SURFACE_BATCH_ROOT if a.surface_only else BATCH_ROOT
    if a.batch_analyze:batch_analyze(a.root)
    if a.batch_render:batch_render(a.root,a.batch_render)
    if a.analyze:analyze(a.root)
    if a.render:render(a.root,a.render)
    if a.review:review(a.root,a.review)


if __name__=='__main__':main()
