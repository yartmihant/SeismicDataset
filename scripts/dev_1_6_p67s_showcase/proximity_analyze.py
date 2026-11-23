"""Quantify source/PML back-action against phase-matched expanded references."""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_6_p67s_showcase.proximity_mesh import ROOT,DISTANCES
from scripts.reporting import read_json,number,mapping

@dataclass
class Surface:
    t:FloatArray
    x:FloatArray
    w:FloatArray
    v:FloatArray


def load(root:Path,name:str)->Surface:
    r=read_json(root/name/'result.json');assert r['status']=='completed'
    assert file_record(root/name/'surface.npz')['sha256']==mapping(r['surface'])['sha256']
    with np.load(root/name/'surface.npz') as a:
        s=Surface(a['time_s'].copy(),a['x_m'].copy(),a['weights_m'].copy(),a['velocity_m_s'].copy())
    assert np.isfinite(s.v).all()
    return s


def aligned(s:Surface,x:FloatArray,shift:float=0)->FloatArray:
    target=x+shift;i=np.searchsorted(s.x,target);i=np.clip(i,0,len(s.x)-1)
    prev=np.maximum(i-1,0);i=np.where(abs(s.x[prev]-target)<abs(s.x[i]-target),prev,i)
    np.testing.assert_allclose(s.x[i],target,atol=1e-8,rtol=0)
    return s.v[:,i]


def norm(v:FloatArray,w:FloatArray)->float:
    return float(np.sum(v*v*w[None,:,None])*.001)


def check(a:FloatArray,b:FloatArray,w:FloatArray)->dict[str,float]:
    return {'relative_l2':float(np.sqrt(norm(a-b,w)/norm(b,w))), 'max_abs_m_s':float(np.max(abs(a-b)))}


def analyze(root:Path,out:Path,analysis_name:str='analysis_v1')->None:
    out.mkdir(parents=True,exist_ok=False);metrics_out=root/analysis_name;metrics_out.mkdir(exist_ok=False)
    refs={i:load(root,f'R{i}') for i in (0,5)};homes={i:load(root,f'H{i}') for i in (0,5)}
    controls:dict[str,object]={}
    for i in (0,5):
        ref=refs[i];b=load(root,f'B{i}');half=load(root,f'R{i}_half')
        controls[f'larger_{i}']=check(ref.v,aligned(b,ref.x),ref.w)
        controls[f'dt_{i}']=check(ref.v,aligned(half,ref.x),ref.w)
        offsets=ref.x-(700+i);gate=(abs(ref.t[:,None]-(.05+np.sqrt(320**2+offsets**2)/2400)[None,:])<=.025)&((offsets>=20)&(offsets<=200))[None,:]
        hp=float(np.max(np.linalg.norm((ref.v-homes[i].v)*gate[:,:,None],axis=2)))
        controls[f'larger_to_horizon_peak_{i}']=float(np.max(np.linalg.norm(ref.v-aligned(b,ref.x),axis=2))/hp)
        controls[f'dt_to_horizon_peak_{i}']=float(np.max(np.linalg.norm(ref.v-aligned(half,ref.x),axis=2))/hp)
    direct=load(root,'direct5');sel=direct.x<=1400
    controls['translation_source5']=check(direct.v[:,sel],aligned(refs[5],direct.x[sel],700),direct.w[sel])
    rows:list[dict[str,object]]=[];curves:list[list[float]]=[]
    fig,axes=plt.subplots(2,3,figsize=(15,8),sharex=True,sharey=True,layout='constrained')
    image_handle=None
    fig_trace,trace_axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout='constrained')
    for d in DISTANCES:
        name=f'P{d:g}';s=load(root,name);phase=int(d%10);shift=700+phase-d;r=aligned(refs[phase],s.x,shift);h=aligned(homes[phase],s.x,shift);e=s.v-r
        offsets=s.x-d;aperture=(offsets>=20)&(offsets<=200)
        arrival=.05+np.sqrt(320**2+offsets**2)/2400
        gate=(abs(s.t[:,None]-arrival[None,:])<=.025)&aperture[None,:]
        horizon=(r-h)*gate[:,:,None];he=norm(horizon,s.w);hp=float(np.max(np.linalg.norm(horizon,axis=2)));assert he>0 and hp>0
        entry:dict[str,object]={'distance_m':d,'horizon_energy_proxy':he,'horizon_peak_m_s':hp,'run_seconds':read_json(root/name/'result.json')['run_seconds']}
        metrics:list[dict[str,object]]=[]
        for margin in (0,10,25,50,100,175):
            for near in (False,True):
                mask=(s.x>margin)&(s.x<1400-margin)
                if near:mask&=abs(offsets)>=20
                for start in (0.,.5):
                    ts=s.t>=start;ev=e[ts][:,mask];rv=r[ts][:,mask];w=s.w[mask]
                    m:dict[str,object]={'receiver_margin_m':margin,'exclude_source20':near,'window_s':[start,1.], 'error_to_horizon_peak':float(np.max(np.linalg.norm(ev,axis=2))/hp),'error_to_horizon_energy':norm(ev,w)/he,'relative_l2':float(np.sqrt(norm(ev,w)/norm(rv,w))),'component_max_abs_m_s':np.max(abs(ev),axis=(0,1)).tolist()}
                    metrics.append(m)
                    if margin==0 and not near and start==0:
                        curves.append([d,number(m['error_to_horizon_peak']),number(m['error_to_horizon_energy']),number(m['relative_l2'])])
        entry['metrics']=metrics
        # Equal right-offset aperture gives an additional comparison independent of receiver distance to left PML.
        entry['right_aperture_error_peak_ratio']=float(np.max(np.linalg.norm(e[:,aperture],axis=2))/hp)
        entry['right_aperture_error_energy_ratio']=norm(e[:,aperture],s.w[aperture])/he
        if d in (0,35,175,700):
            l=load(root,f'L{d:g}');le=aligned(l,s.x)-r
            entry['left_only_peak_ratio']=float(np.max(np.linalg.norm(le[:,1:-1],axis=2))/hp)
            entry['left_only_energy_ratio']=norm(le[:,1:-1],s.w[1:-1])/he
            entry['other_sides_peak_ratio']=float(np.max(np.linalg.norm((s.v-aligned(l,s.x))[:,1:-1],axis=2))/hp)
        if d in (0,5,175):
            half=load(root,f'P{d:g}_half');rh=load(root,f'R{phase}_half');eh=half.v-aligned(rh,s.x,shift)
            entry['half_dt_boundary_peak_ratio']=float(np.max(np.linalg.norm(eh[:,1:-1],axis=2))/hp)
            entry['half_dt_boundary_energy_ratio']=norm(eh[:,1:-1],s.w[1:-1])/he
            entry['boundary_error_dt_change_to_horizon_peak']=float(np.max(np.linalg.norm((e-eh)[:,1:-1],axis=2))/hp)
            controls[f'production_dt_{d:g}']=check(s.v,half.v,s.w)
        # Per-receiver traces allow inspecting errors near PML separately from central receivers.
        np.savez(metrics_out/f'{name}_metrics.npz',x_m=s.x,error_peak_m_s=np.max(np.linalg.norm(e,axis=2),axis=0),reference_peak_m_s=np.max(np.linalg.norm(r,axis=2),axis=0),error_time_peak_m_s=np.max(np.linalg.norm(e[:,1:-1],axis=2),axis=1),time_s=s.t)
        if d in (0,5,35,100,175,700):
            ax=axes.flat[(0,5,35,100,175,700).index(int(d))]
            image_handle=ax.pcolormesh(s.x,s.t,e[:,:,1]/hp,cmap='RdBu_r',vmin=-.001,vmax=.001,rasterized=True,shading='auto')
            ax.set_title(f'Источник {d:g} м');ax.set_xlabel('x, м');ax.set_ylabel('t, с');ax.invert_yaxis() if not ax.yaxis_inverted() else None
        if d in (0,5,35,175,700):
            ix=int(np.argmin(abs(s.x-(d+100))));assert abs(s.x[ix]-(d+100))<1e-8
            trace_axes[0].plot(s.t,e[:,ix,1]/hp,label=f'd={d:g} м')
            if d==175:
                trace_axes[1].plot(s.t,r[:,ix,1]/hp,label='Полная запись эталона')
                trace_axes[1].plot(s.t,(r-h)[:,ix,1]/hp,label='Эталон − однородная среда')
                trace_axes[1].axvspan(float(arrival[ix]-.025),float(arrival[ix]+.025),alpha=.2,color='gray',label='Окно первого PP')
        rows.append(entry)
    trace_axes[0].set(ylabel='ΔVy / пик первого отражения',title='Приёмник в 100 м справа от источника: влияние всех PML')
    trace_axes[1].set(xlabel='t, с',ylabel='Vy / пик первого отражения',title='Первое отражение (d=175 м); прямая волна вне вертикальной шкалы',ylim=(-1.2,1.2))
    for ax_trace in trace_axes:ax_trace.legend(fontsize=8);ax_trace.grid(alpha=.2)
    fig_trace.savefig(out/'relative_offset_traces.png',dpi=150);plt.close(fig_trace)
    if image_handle is not None:fig.colorbar(image_handle,ax=axes.ravel().tolist(),label='ΔVy / пик первого отражения; насыщение ±0.1%')
    fig.suptitle('Ошибка сейсмограммы: PML350 − расширенный эталон, EFVB00001')
    fig.savefig(out/'difference_seismograms.png',dpi=150);plt.close(fig)
    c=np.array(curves);fig,axes2=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for ax,col,label in zip(axes2,(1,2),('Пик ошибки / пик первого отражения','Энергия ошибки / энергия первого отражения')):
        ax.semilogy(c[:,0],c[:,col],'o-',label='Все внутренние приёмники')
        ax.semilogy(c[:,0],[number(row['right_aperture_error_peak_ratio' if col==1 else 'right_aperture_error_energy_ratio']) for row in rows],'s--',label='Справа от источника 20–200 м')
        for level in ((.001,.01,.05) if col==1 else (1e-6,1e-4,1e-2)):ax.axhline(level,color='gray',alpha=.4,lw=.7)
        ax.set(xlabel='Источник → левый PML, м',ylabel=label);ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'distance_errors.png',dpi=160);plt.close(fig)
    # Requirements apply to both peak and integrated signal-energy proxy.
    thresholds:dict[str,object]={}
    for margin in (0,10,25,50,100,175):
        for peak,energy in ((.001,1e-6),(.01,1e-4),(.05,1e-2)):
            valid=[]
            for row in rows:
                ms=row['metrics'];assert isinstance(ms,list)
                m=next(m for m in ms if m['receiver_margin_m']==margin and not m['exclude_source20'] and m['window_s']==[0.,1.])
                if number(m['error_to_horizon_peak'])<=peak and number(m['error_to_horizon_energy'])<=energy:valid.append(row['distance_m'])
            thresholds[f'margin{margin}_peak{peak}_energy{energy}']=valid
    write(metrics_out/'analysis.json',{'status':'completed','controls':controls,'rows':rows,'passing_distances_m':thresholds,'definition':'error = PML - translated layered reference, first reflection = layered - homogeneous in first PP gate; energy = sum(dt*w*(Vx²+Vy²)), not mechanical energy','inputs':[file_record(p) for p in sorted(root.glob('*/surface.npz'))]})
    lines=['# Источник у PML350: p6, EFVB00001','','Параметры и ограничения: [постановка](../../../docs/dev_1_6_pml_proximity_spec.txt). Все значения относятся к Vx/Vy внутри основной области за 0–1 с. Энергия — интеграл квадрата скорости по времени и координате, а не поток механической энергии. Нормировка — первое отражение от горизонта в фиксированной правой апертуре 20–200 м.','','![Зависимость от расстояния](distance_errors.png)','','![Разности сейсмограмм](difference_seismograms.png)','','![Трассы и первое отражение](relative_offset_traces.png)','','| Источник–PML, м | Пик ошибки / первое отражение, % | Энергия ошибки / первое отражение, % | L2 ошибки / полной записи, % |','|---:|---:|---:|---:|']
    lines.extend(f'| {r[0]:g} | {100*r[1]:.6g} | {100*r[2]:.6g} | {100*r[3]:.6g} |' for r in curves)
    lines+=['','Метрики всей основной поверхности включают вклад боковых и нижнего PML. Контроли left_only отделяют левую границу; проверки большей области, прямого переноса источника и dt/2 находятся в analysis.json. Это не универсальный предел для других материалов и частот.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--out',type=Path,default=Path('img/dev_1_6_p67s_showcase/proximity_v1'));p.add_argument('--analysis-name',default='analysis_v1');a=p.parse_args();analyze(a.root,a.out,a.analysis_name)
