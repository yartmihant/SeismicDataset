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
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import ROOT, CASES, CONTROLS
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
    (output/'index.html').write_text(body)
    write(output/'render.json',{'status':'completed','metrics':file_record(root/'analysis/metrics.json'),'files':{p.name:file_record(p) for p in output.iterdir() if p.is_file()}})


def main()->None:
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--analyze',action='store_true');p.add_argument('--render',type=Path);a=p.parse_args()
    if a.analyze:analyze(a.root)
    if a.render:render(a.root,a.render)


if __name__=='__main__':main()
