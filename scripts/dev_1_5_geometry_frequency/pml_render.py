"""Comparable PML movies, seismograms and reference-subtracted diagnostics."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import shutil
import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize,SymLogNorm
from scripts.dev_1_5_geometry_frequency.pml_setup import ROOT,THICK_ROOT,THICKNESSES,W,H,VP,VS,ALPHA
from scripts.dev_1_5_geometry_frequency.pml_analyze import MODES,traces,core_indices,core_traces,thickness_paths
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_3_point_source.run import file_record
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json,mapping,number,sequence

OUT=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/pml_homogeneous_v1'
LABELS={'old':'Массовое демпфирование + ABC','pml':'PML, внешний край свободный','pml_abc':'PML + ABC','reference':'Расширенная область: вырез'}
COLORS={'old':'#aa493c','pml':'#24724e','pml_abc':'#386c9f','reference':'#333333'}
FloatArray=NDArray[np.float64]


def norm(log: bool) -> Normalize:
    return SymLogNorm(linthresh=1e-14,vmin=-1e-10,vmax=1e-10,base=10) if log else Normalize(vmin=-1e-10,vmax=1e-10)


def movies(order: int) -> None:
    encoder=shutil.which('ffmpeg')
    if encoder is None:
        binaries=list((PROJECT_ROOT/'.runtime_ch15_ffmpeg/imageio_ffmpeg/binaries').glob('ffmpeg-*'))
        if len(binaries)!=1:raise RuntimeError('Missing existing project ffmpeg')
        encoder=str(binaries[0])
    matplotlib.rcParams['animation.ffmpeg_path']=encoder
    base=ROOT/f'p{order}';nodes=np.round(np.load(base/'old/nodes.npy'),8)
    xs=np.unique(nodes[:,0]);ys=np.unique(nodes[:,1]);assert len(xs)*len(ys)==len(nodes)
    grid=nodes.reshape(len(ys),len(xs),2);assert np.all(grid[:,:,0]==xs[None,:]) and np.all(grid[:,:,1]==ys[:,None])
    gx=np.linspace(xs[0],xs[-1],701);gy=np.linspace(ys[0],ys[-1],351)
    ix=np.clip(np.searchsorted(xs,gx,side='right')-1,0,len(xs)-2);iy=np.clip(np.searchsorted(ys,gy,side='right')-1,0,len(ys)-2)
    wx=(gx-xs[ix])/(xs[ix+1]-xs[ix]);wy=(gy-ys[iy])/(ys[iy+1]-ys[iy])
    def raster(frame: FloatArray) -> FloatArray:
        v=frame.reshape(len(ys),len(xs),2)
        low=v[iy[:,None],ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None],ix[None,:]+1]*wx[None,:,None]
        high=v[iy[:,None]+1,ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None]+1,ix[None,:]+1]*wx[None,:,None]
        return low*(1-wy)[:,None,None]+high*wy[:,None,None]
    fields={mode:np.load(base/mode/'velocity.npy',mmap_mode='r') for mode in MODES}
    for logarithmic in (False,True):
        name='symlog' if logarithmic else 'linear';fig,axes=plt.subplots(4,2,figsize=(14,14),layout='constrained');artists=[]
        for row,mode in enumerate(MODES):
            shown=raster(fields[mode][0]);row_artists=[]
            for c,ax in enumerate(axes[row]):
                artist=ax.imshow(shown[:,:,c],origin='lower',extent=(-175,1575,-875,0),cmap='RdBu_r',norm=norm(logarithmic),aspect='equal',interpolation='nearest')
                row_artists.append(artist)
                ax.axvline(0,color='k',ls='--',lw=.6);ax.axvline(W,color='k',ls='--',lw=.6);ax.axhline(-H,color='k',ls='--',lw=.6)
                ax.set(title=LABELS[mode]+' · '+('Vx' if c==0 else 'Vy'),xlabel='x, м',ylabel='y, м')
            artists.append(row_artists)
        fig.colorbar(artists[0][0],ax=list(axes.ravel()),shrink=.75,label='Vx, Vy, м/с · '+('SymLog, линейный порог 1e−14' if logarithmic else 'линейная шкала'),extend='both')
        title=fig.suptitle('');writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-crf','20','-pix_fmt','yuv420p','-threads','2'])
        path=OUT/f'p{order}_{name}.mp4'
        with writer.saving(fig,str(path),dpi=100):
            for i in range(21):
                for row,mode in enumerate(MODES):
                    shown=raster(fields[mode][i])
                    for c in range(2):artists[row][c].set_data(shown[:,:,c])
                title.set_text(f'p{order} · h=5 м · однородная1400×700 м · Рикер30 Гц · t={i*.05:.2f} с\nОбщая фиксированная шкала ±1e−10 м/с; пунктир — физическая область')
                writer.grab_frame()
                if i==20:fig.savefig(OUT/f'p{order}_{name}_final.png',dpi=110)
        plt.close(fig)


def figures(order: int) -> None:
    base=ROOT/f'p{order}';metrics=mapping(mapping(read_json(ROOT/'analysis/metrics.json')['orders'])[f'p{order}'])
    with np.load(ROOT/'analysis'/f'p{order}_curves.npz') as z:
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        peak=float(z['reference_kinetic_j_per_m'].max())
        for mode in MODES:
            axes[0].semilogy(z['time_s'],np.maximum(z[mode+'_kinetic_j_per_m']/peak,1e-30),label=LABELS[mode],color=COLORS[mode])
            if mode!='reference':axes[1].semilogy(z['surface_time_s'],np.maximum(z[mode+'_surface_error_l2_m_s_sqrt_m'],1e-30),label=LABELS[mode],color=COLORS[mode])
        axes[0].set(xlabel='t, с',ylabel='K / max(K_reference)',title='Кинетическая энергия в физической области',ylim=(1e-12,2))
        peak_error=max(float(z[m+'_surface_error_l2_m_s_sqrt_m'].max()) for m in MODES[:3])
        axes[1].set(xlabel='t, с',ylabel='Норма ошибки Vx/Vy, м^(3/2)/с',title='Поверхность: разность с расширенной областью',xlim=(.1,1),ylim=(peak_error*1e-7,peak_error*2))
        for ax in axes:ax.axvspan(.5,1,color='#dddddd',alpha=.3);ax.grid(alpha=.25);ax.legend(fontsize=8)
        fig.suptitle(f'p{order} · физическая область0≤x≤1400, −700≤y≤0 · серое окно0,5–1 с')
        fig.savefig(OUT/f'p{order}_decay.png',dpi=150);plt.close(fig)
    _,rx,reference=traces(base/'reference')
    for difference in (False,True):
        modes=MODES[:3] if difference else MODES
        fig,axes=plt.subplots(len(modes),2,figsize=(14,3.6*len(modes)),layout='constrained')
        for row,mode in enumerate(modes):
            times,x,v=traces(base/mode)
            if difference:v=v-reference
            keep=(rx[:,0]>=-1e-8)&(rx[:,0]<=W+1e-8) if difference else np.ones(len(rx),dtype=bool)
            for c,ax in enumerate(axes[row]):
                artist=ax.pcolormesh(x[keep,0],times,v[:,keep,c],shading='nearest',cmap='RdBu_r',norm=norm(True),rasterized=True)
                ax.invert_yaxis();ax.set(xlabel='x приёмника, м',ylabel='t, с',title=LABELS[mode]+' · '+('ΔV' if difference else 'V')+('x' if c==0 else 'y'))
                ax.axhline(.5,color='k',ls=':',lw=.6)
        colorbar=fig.colorbar(artist,ax=list(axes.ravel()),shrink=.7,label='м/с · SymLog ±1e−10, линейный порог1e−14',extend='both')
        fig.suptitle(f'p{order} · '+('Ошибка относительно расширенной области, только физический верх' if difference else 'Сейсмограммы всех верхних узлов, включая полосу175 м'))
        stem=f'p{order}_'+('surface_error' if difference else 'surface')
        fig.savefig(OUT/(stem+'.png'),dpi=140)
        for ax in axes.ravel():ax.collections[0].set_norm(norm(False))
        colorbar.set_label('м/с · линейная шкала ±1e−10')
        fig.savefig(OUT/(stem+'_linear.png'),dpi=140);plt.close(fig)


def report() -> None:
    metrics=read_json(ROOT/'analysis/metrics.json');sections=[]
    for order in (4,5):
        record=mapping(mapping(metrics['orders'])[f'p{order}']);modes=mapping(record['modes']);rows=''
        for mode in MODES[:3]:
            row=mapping(modes[mode]);late=mapping(row['late_surface'])
            reduction=1. if mode=='old' else number(row['late_surface_error_reduction_factor_vs_old'])
            rows+=f'<tr><th>{html.escape(LABELS[mode])}</th><td>{number(late["error_weighted_l2"]):.6e}</td><td>{reduction:.3f}×</td><td>{number(row["kinetic_final_j_per_m"]):.6e}</td><td>{number(row["run_seconds"]):.3f}</td></tr>'
        temporal=''
        if order==5:
            tests=mapping(record['time_step_check']);pml=mapping(tests['pml']);delta=number(mapping(pml['late_coarse_fine'])['error_weighted_l2']);residual=number(mapping(mapping(modes['pml'])['late_surface'])['error_weighted_l2'])
            temporal=f'<p>Контроль шага PML: разность dt и dt/2 в окне0,5–1 с равна {delta/residual*100:.3f}% нормы основной PML-ошибки относительно контрольной области. Это проверка чувствительности по времени; контрольная область рассчитана только на основном общем dt.</p>'
        dt=number(mapping(modes['old'])['dt_s'])
        sections.append(f'<section><h2>p{order}: dt={dt*1e6:.6f} мкс, {mapping(modes["old"])["steps"]} шагов</h2><table><tr><th>Граница</th><th>L2 ошибки поверхности,0,5–1 с</th><th>Гашение ошибки относительно старого слоя</th><th>Кинетическая энергия при1 с, Дж/м</th><th>run, с</th></tr>{rows}</table>{temporal}<img src="p{order}_decay.png" alt="Остаточная кинетическая энергия и норма отражённого поля"><h3>Линейная шкала ±1e−10 м/с</h3><video controls preload="none" poster="p{order}_linear_final.png" src="p{order}_linear.mp4"></video><h3>Чувствительная знаковая логарифмическая шкала</h3><video controls preload="none" poster="p{order}_symlog_final.png" src="p{order}_symlog.mp4"></video><p><a href="p{order}_surface.png">Полные сейсмограммы</a> · <a href="p{order}_surface_error.png">Сейсмограммы разности с расширенной областью</a> · <a href="p{order}_surface_linear.png">Линейные сейсмограммы</a> · <a href="p{order}_surface_error_linear.png">Линейная разность</a></p><a href="p{order}_surface_error.png"><img src="p{order}_surface_error.png" loading="lazy" alt="Сейсмограммы ошибки"></a></section>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>PML против массового демпфирования</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#20252b}h1{font-size:28px}h2{margin-top:42px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}img,video{max-width:100%;display:block;margin:20px auto}video{max-height:1000px}a{color:#245a88}</style><h1>PML в однородной среде</h1><p>Основная область1400×700 м, полная1750×875 м; источник x=700,y=0, сила вниз1 Н/м, Рикер30 Гц,t0=0,05 с. Vp=4087 м/с, Vs=2431,62036 м/с, ρ=1000 кг/м³. Элементы5 м, p4/p5, β=1/27, γ=0,5,1 с. Все расчёты — общий seismic CUDA с поддержкой PML.</p><p>Старый способ: квадратичный массовый слой175 м,k=0,5,αmax=173,088611 с⁻¹, ABC по трём внешним граням. PML: те же175 м, dx(x),dy(y) квадратичны и возрастают от нуля до173,088611 с⁻¹; массового демпфирования нет. Проверены свободный внешний край PML и добавление ABC. Верх всегда свободный; в углах действуют оба направления PML. Амплитуда профиля не оптимизировалась.</p><p>Шаг получен native CFL0,8 в старой постановке и явно передан PML, поскольку текущая связанная система с памятью не поддерживает встроенный CFL. Внутри порядка шаг общий; дополнительно p5 старого слоя и PML повторены с dt/2. Порядки не сравниваются как эквивалентные по точности.</p><p>Контроль: однородная область6400×3200 м, x=−2500…3900,y=−3200…0, без поглощения, тот же h/p/dt. В анимации показан общий вырез. Минимальное геометрическое время возврата от её внешних границ в вырез —1,352 с, больше времени расчёта. Это численное решение в расширенной области, не аналитический эталон. Ранние записи сопоставлены отдельно.</p><p>Главная метрика — пространственно взвешенная GLL L2 разности Vx/Vy с контрольной областью за0,5–1 с, только x=0…1400. Дополнительно учитывается кинетическая энергия в основной области; это не полная упругая энергия. Физическое остаточное поле отделено от отражений вычитанием контроля. Поля сохранены в21 момент через0,05 с, поверхность —1001 момент через0,001 с; анимации5 кадров/с. Фиксированный цвет может насыщаться, исходные значения не обрезаны.</p><p>Оба порядка выполнялись на разных A100 параллельно; времена приведены справочно, это не серия статистического бенчмарка. Основная производственная очередь не возобновлялась.</p>'''+''.join(sections)+'''<p><a href="../../../data/dev_1_5_geometry_frequency/pml_homogeneous_saturn_v1/analysis/metrics.json">Все численные метрики и контроль шага</a></p></html>'''
    (OUT/'index.html').write_text(page)
    files={p.name:file_record(p) for p in OUT.iterdir() if p.is_file() and p.name!='render.json'}
    write(OUT/'render.json',{'status':'completed','files':files,'frames_per_movie':21,'fps':5,'movie_count':4,'linear_limit_m_s':1e-10,'symlog_linthresh_m_s':1e-14,'source_metrics':file_record(ROOT/'analysis/metrics.json')})


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--thickness',action='store_true');parser.add_argument('--thickness-review',action='store_true');parser.add_argument('--layered',action='store_true');parser.add_argument('--layered-review',action='store_true');parser.add_argument('--first-reflection',action='store_true');parser.add_argument('--timing',action='store_true');args=parser.parse_args()
    if args.timing:render_timing();return
    if args.first_reflection:render_first_reflection();return
    if args.layered_review:review_layered();return
    if args.layered:render_layered();return
    if args.thickness_review:review_thickness();return
    if args.thickness:render_thickness();return
    OUT.mkdir(parents=True,exist_ok=False)
    for order in (4,5):movies(order);figures(order)
    report();print('RENDER COMPLETED',OUT,flush=True)


THICK_OUT=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/pml_thickness_v1'
THICK_LABELS=('L175',)+tuple(f'L{v}' for v in THICKNESSES)


def thickness_movies(order: int,fixed: bool=False) -> None:
    labels=('L175',)+tuple(f'F{v}' for v in THICKNESSES) if fixed else THICK_LABELS
    prefix=f'p{order}'+('_fixed' if fixed else '')
    cases=mapping(mapping(mapping(read_json(THICK_ROOT/'analysis/metrics.json')['orders'])[f'p{order}'])['cases'])
    encoder=shutil.which('ffmpeg')
    if encoder is None:
        encoder=str(next((PROJECT_ROOT/'.runtime_ch15_ffmpeg/imageio_ffmpeg/binaries').glob('ffmpeg-*')))
    matplotlib.rcParams['animation.ffmpeg_path']=encoder
    paths=dict(thickness_paths(order))
    grids:dict[str,tuple[FloatArray,FloatArray,NDArray[np.int64],NDArray[np.int64],FloatArray,FloatArray]]={}
    for label in labels:
        nodes=np.round(np.load(paths[label]/'nodes.npy'),8)
        xs=np.unique(nodes[:,0]);ys=np.unique(nodes[:,1]);assert len(xs)*len(ys)==len(nodes)
        gx=np.linspace(xs[0],xs[-1],601);gy=np.linspace(ys[0],ys[-1],301)
        ix=np.clip(np.searchsorted(xs,gx,side='right')-1,0,len(xs)-2);iy=np.clip(np.searchsorted(ys,gy,side='right')-1,0,len(ys)-2)
        grids[label]=(xs,ys,ix,iy,(gx-xs[ix])/(xs[ix+1]-xs[ix]),(gy-ys[iy])/(ys[iy+1]-ys[iy]))
    fields={label:np.load(paths[label]/'velocity.npy',mmap_mode='r') for label in labels}
    def raster(label: str,i: int) -> FloatArray:
        xs,ys,ix,iy,wx,wy=grids[label]
        v=fields[label][i].reshape(len(ys),len(xs),2)
        lo=v[iy[:,None],ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None],ix[None,:]+1]*wx[None,:,None]
        hi=v[iy[:,None]+1,ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None]+1,ix[None,:]+1]*wx[None,:,None]
        return lo*(1-wy)[:,None,None]+hi*wy[:,None,None]
    for logarithmic in (False,True):
        name='symlog' if logarithmic else 'linear';fig,axes=plt.subplots(3,4,figsize=(18,12),layout='constrained');artists=[]
        for j,label in enumerate(labels):
            pair=[]
            for c in range(2):
                ax=axes[j//2,(j%2)*2+c]
                a=ax.imshow(raster(label,0)[:,:,c],origin='lower',extent=(grids[label][0][0],grids[label][0][-1],grids[label][1][0],grids[label][1][-1]),cmap='RdBu_r',norm=norm(logarithmic),aspect='equal')
                growth=bool(mapping(cases[label])['strong_growth'])
                ax.set(title=f'{label[1:]} м · '+('РОСТ · ' if growth else '')+('Vx' if c==0 else 'Vy'),xlabel='x, м',ylabel='y, м',xlim=(-175,1575),ylim=(-875,0));pair.append(a)
                if growth:ax.title.set_color('#b00020')
                ax.axvline(0,color='k',ls=':',lw=.5);ax.axvline(W,color='k',ls=':',lw=.5);ax.axhline(-H,color='k',ls=':',lw=.5)
            artists.append(pair)
        fig.colorbar(artists[0][0],ax=list(axes.ravel()),shrink=.7,label='м/с · '+('SymLog, порог 1e−14' if logarithmic else 'линейная шкала'),extend='both')
        title=fig.suptitle('');writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-crf','20','-pix_fmt','yuv420p','-threads','2'])
        with writer.saving(fig,str(THICK_OUT/f'{prefix}_{name}.mp4'),dpi=100):
            for i in range(21):
                for j,label in enumerate(labels):
                    v=raster(label,i)
                    for c in range(2):artists[j][c].set_data(v[:,:,c])
                title.set_text(f'PML + ABC · p{order} · t={i*.05:.2f} с · '+('dmax=173,09 с⁻¹' if fixed else 'dmax·L=const')+'\nПоказана вся область с PML; красные подписи — обнаруженный рост; шкала ±1e−10 м/с')
                writer.grab_frame()
                if i==20:fig.savefig(THICK_OUT/f'{prefix}_{name}_final.png',dpi=110)
        plt.close(fig)


def thickness_figures(order: int,fixed: bool=False) -> None:
    labels=('L175',)+tuple(f'F{v}' for v in THICKNESSES) if fixed else THICK_LABELS
    prefix=f'p{order}'+('_fixed' if fixed else '')
    cases=mapping(mapping(mapping(read_json(THICK_ROOT/'analysis/metrics.json')['orders'])[f'p{order}'])['cases'])
    paths=dict(thickness_paths(order));t,x,ref=core_traces(paths['reference'])
    for difference in (False,True):
        fig,axes=plt.subplots(6,2,figsize=(14,18),layout='constrained');images=[]
        for row,label in enumerate(labels):
            _,_,v=core_traces(paths[label])
            if difference:v=v-ref
            for c in range(2):
                ax=axes[row,c];a=ax.pcolormesh(x[:,0],t,v[:,:,c],shading='nearest',cmap='RdBu_r',norm=norm(True),rasterized=True);images.append(a)
                growth=bool(mapping(cases[label])['strong_growth'])
                ax.invert_yaxis();ax.set(title=f'{label[1:]} м · '+('РОСТ · ' if growth else '')+('ΔV' if difference else 'V')+('x' if c==0 else 'y'),xlabel='x, м',ylabel='t, с')
                ax.axhline(.5,color='k',ls=':',lw=.6)
        cb=fig.colorbar(images[-1],ax=list(axes.ravel()),shrink=.7,label='м/с · SymLog ±1e−10, порог 1e−14',extend='both')
        fig.suptitle(f'p{order} · PML + ABC · '+('разность с расширенным контролем' if difference else 'поверхность основной области'))
        stem=f'{prefix}_surface'+('_error' if difference else '')
        fig.savefig(THICK_OUT/(stem+'.png'),dpi=120)
        for a in images:a.set_norm(norm(False))
        cb.set_label('м/с · линейная шкала ±1e−10');fig.savefig(THICK_OUT/(stem+'_linear.png'),dpi=120);plt.close(fig)
    with np.load(THICK_ROOT/'analysis'/f'p{order}_curves.npz') as z:
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        peak=float(z['reference_kinetic'].max())
        for label in ('old',)+labels:
            name='Старый слой 175 м' if label=='old' else label[1:]+' м'
            if not bool(mapping(cases[label])['strong_growth']):axes[0].semilogy(z['surface_time_s'],np.maximum(z[label+'_surface_error'],1e-30),label=name)
            axes[1].semilogy(z['field_time_s'],np.maximum(z[label+'_full_peak'],1e-30),label=name)
        axes[1].semilogy(z['field_time_s'],np.maximum(z['reference_full_peak'],1e-30),'k--',label='Контроль')
        peak_error=max(float(z[label+'_surface_error'].max()) for label in ('old',)+labels if not bool(mapping(cases[label])['strong_growth']))
        axes[0].set(xlim=(.1,1),ylim=(peak_error*1e-6,peak_error*2),xlabel='t, с',ylabel='GLL-норма ΔVx/ΔVy',title='Ошибка поверхности: без случаев с сильным ростом')
        axes[1].set(ylim=(1e-15,None),xlabel='t, с',ylabel='max |Vx,Vy|, м/с',title='Пик во всей области, включая PML')
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8);ax.axvline(.5,color='k',ls=':',lw=.6)
        fig.suptitle(f'p{order} · квадратичный PML, ABC · '+('dmax=173,09' if fixed else 'dmax·L=const'));fig.savefig(THICK_OUT/f'{prefix}_decay.png',dpi=140);plt.close(fig)


def render_thickness() -> None:
    THICK_OUT.mkdir(parents=True,exist_ok=False);metrics=read_json(THICK_ROOT/'analysis/metrics.json');sections=[]
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for order,fixed in ((4,False),(5,False),(5,True)):
        labels=('L175',)+tuple(f'F{v}' for v in THICKNESSES) if fixed else THICK_LABELS
        prefix=f'p{order}'+('_fixed' if fixed else '')
        description=f'p{order} · '+('dmax=173,09' if fixed else 'dmax·L=const')
        entry=mapping(mapping(metrics['orders'])[f'p{order}']);cases=mapping(entry['cases']);rows=''
        Ls=[175,*THICKNESSES]
        for ax,window in zip(axes[:2],('full','late')):
            ax.semilogy(Ls,[np.nan if bool(mapping(cases[label])['strong_growth']) else number(mapping(cases[label])[window+'_error_ratio_to_L175']) for label in labels],'-o',label=description)
            ax.set(xlabel='Толщина PML, м',ylabel='J / J при 175 м',title='Ошибка за '+('0–1 с' if window=='full' else '0,5–1 с'));ax.axhline(1,color='k',ls=':',lw=.7)
        axes[2].plot(Ls,[np.nan if bool(mapping(cases[label])['strong_growth']) else number(mapping(cases[label])['run_seconds']) for label in labels],'-o',label=description)
        for label in labels:
            row=mapping(cases[label]);state='РОСТ' if bool(row['strong_growth']) else 'нет сильного роста за1 с';rows+=f'<tr><th>{row["thickness_m"]} м / {row["rows"]}</th><td>{state}</td><td>{number(row["late_full_peak_m_s"]):.3e}</td><td>{number(row["dmax_s_inv"]):.2f}</td><td>{number(row["full_error_ratio_to_L175"]):.3g}</td><td>{number(row["late_error_ratio_to_L175"]):.3g}</td><td>{number(row["full_error_reduction_vs_old"]):.2f}×</td><td>{number(row["run_seconds"]):.3f}</td></tr>'
        time_note=''
        if order==5 and not fixed:time_note=f'<p>Контроль 10 м: разность dt/dt/2 составляет {100*number(mapping(entry["time_step_check_L10"])["full_difference_over_L10_error"]):.4f}% нормы ошибки этого варианта за0–1 с. Максимум при dt/2 достигает {number(mapping(entry["time_step_check_L10"])["late_full_peak_fine_m_s"]):.3e} м/с: уменьшение шага не устранило рост.</p>'
        sections.append(f'<section><h2>{description}</h2><table><tr><th>Толщина / ряды</th><th>Полное поле</th><th>Поздний пик, м/с</th><th>dmax, с⁻¹</th><th>J0–1 / J175</th><th>J0,5–1 / J175</th><th>Уменьшение J0–1 против старого слоя</th><th>run, с</th></tr>{rows}</table>{time_note}<img src="{prefix}_decay.png"><h3>Полные поля, включая PML</h3><video controls preload="none" poster="{prefix}_linear_final.png" src="{prefix}_linear.mp4"></video><video controls preload="none" poster="{prefix}_symlog_final.png" src="{prefix}_symlog.mp4"></video><p><a href="{prefix}_surface.png">Сейсмограммы</a> · <a href="{prefix}_surface_linear.png">Линейные</a> · <a href="{prefix}_surface_error.png">Разность с контролем</a> · <a href="{prefix}_surface_error_linear.png">Линейная разность</a></p><img src="{prefix}_surface_error.png" loading="lazy"></section>')
    axes[2].set(xlabel='Толщина PML, м',ylabel='run, с',title='Время с записью полей; одиночные опыты')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.savefig(THICK_OUT/'tradeoff.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for thickness in (175,*THICKNESSES):
        distance=np.linspace(0,thickness,200);ax.plot(distance,ALPHA*175/thickness*(distance/thickness)**2,label=f'{thickness} м')
    ax.set(xlabel='Расстояние от входа в PML, м',ylabel='d, с⁻¹',title='Квадратичные профили с одинаковым интегралом');ax.grid(alpha=.25);ax.legend();fig.savefig(THICK_OUT/'profiles.png',dpi=140);plt.close(fig)
    for order,fixed in ((4,False),(5,False),(5,True)):thickness_movies(order,fixed);thickness_figures(order,fixed)
    page="""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Толщина PML</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#20252b}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}img,video{max-width:100%;display:block;margin:24px auto}a{color:#245a88}</style><h1>Толщина PML: качество и стоимость</h1><p>Однородная основная область1400×700 м, Vp4087 м/с, Vs2431,62036 м/с, плотность1000 кг/м³. Рикер30 Гц, центральная сила вниз1 Н/м,1 с. p4/p5, h5 м, β1/27. Квадратичный PML с внешним ABC, свободный верх. Толщина175/100/50/25/15/10 м; внешний размер области уменьшается, основная область неизменна.</p><p>dmax=173,0886108476381·175/L. В первой серии интеграл профиля сохраняется. После обнаружения роста добавлен контроль p5 всех пяти толщин с фиксированным dmax=173,088611 с⁻¹. Во всех вариантах одного порядка одинаковый dt: p4 1/36000 с, p5 1/53000 с. У p5 10 м дополнительно dt/2. Значения175 м, старый массовый слой k0,5+ABC и расширенный контроль6400×3200 м повторно используются из предыдущего исследования; native-библиотека та же.</p><p>Метрика J — пространственно взвешенная GLL-норма разности Vx/Vy с расширенным контролем на физическом верхе0…1400 м; общий множитель временного шага регистрации опущен. Основное окно0–1 с включает ранние возвраты тонких слоёв, дополнительно0,2–1 и0,5–1 с. Отношение J/J175 меньше1 означает улучшение. Кинетическая энергия не является полной упругой энергией. Поля21 кадр через0,05 с; поверхность1001 отсчёт через0,001 с. Анимации показывают полные поля, включая PML; внешние границы различаются. Сейсмограммы и нормы ошибки ограничены общей физической областью.</p><p>Времена включают запись полных полей, объём которой уменьшается вместе с толщиной. Это одиночные прогоны, без статистического бенчмарка; исторические времена175 м не пересняты. Основная очередь остановлена.</p><p><b>Тонкие слои с усилением dmax показали численный рост, начинающийся внутри PML.</b> Признак РОСТ: максимум полного поля за0,5–1 с превышает в10 раз максимум контрольного поля за весь опыт. Это индикатор сильного роста, не доказательство устойчивости остальных случаев на больших временах. Случаи с ростом исключены из графика качества/времени и из левого графика ошибки; они показаны в таблицах, на правом графике пика и в анимациях. Цвет насыщается при выходе за шкалу, поэтому для оценки роста нужны численные пики.</p><img src="tradeoff.png"><img src="profiles.png">"""+''.join(sections)+"""<p><a href="../../../data/dev_1_5_geometry_frequency/pml_thickness_saturn_v1/analysis/metrics.json">Полные метрики</a> · <a href="../pml_homogeneous_v1/index.html">Предыдущий тест175 м</a></p></html>"""
    (THICK_OUT/'index.html').write_text(page)
    write(THICK_OUT/'render.json',{'status':'completed','files':{p.name:file_record(p) for p in THICK_OUT.iterdir() if p.is_file()},'frames_per_movie':21,'fps':5,'movie_count':6,'source_metrics':file_record(THICK_ROOT/'analysis/metrics.json')})
    print('THICKNESS RENDER COMPLETED',flush=True)


def review_thickness() -> None:
    notes:dict[str,object]={}
    for order in (4,5):
        with np.load(THICK_ROOT/'analysis'/f'p{order}_curves.npz') as z:
            values=z['L100_full_peak']
            notes[f'p{order}']={'peak_t060_m_s':float(values[12]),'peak_t100_m_s':float(values[-1]),'growth_ratio_060_to_100':float(values[-1]/values[12]),'monotone_060_to_100':bool(np.all(np.diff(values[12:])>0))}
    p5=mapping(notes['p5'])
    notice=f'<aside style="border:2px solid #b06920;padding:16px;background:#fff8e8"><b>Уточнение по100 м: хороший результат за1 с ещё не означает устойчивость.</b><p>У100 м с усиленным профилем максимум полного поля при p5 монотонно растёт от {number(p5["peak_t060_m_s"]):.3e} м/с в0,6 с до {number(p5["peak_t100_m_s"]):.3e} м/с в1 с. Такой же тренд есть у p4. Порог грубого роста РОСТ в таблице здесь ещё не превышен, но поздний рост уже наблюдается. Для производственной замены175 м этот вариант пока не подтверждён. Слой100 м экономит около23% времени p5 и даёт в2,07 раза большую позднюю ошибку, но требует дополнительной проверки профиля и устойчивости.</p><p>При50 м и тоньше с усилением рост уже сильный; dt/2 на10 м его не устраняет. При прежнем dmax173,09 с⁻¹ тонкие варианты обходятся без сильного роста за1 с, но отражают значительно больше. Причина неустойчивости реализации и оптимальная амплитуда этим сравнением не установлены.</p></aside>'
    source=THICK_OUT/'index.html';page=source.read_text();anchor='<h1>Толщина PML: качество и стоимость</h1>';assert anchor in page
    target=THICK_OUT/'reviewed.html'
    with target.open('x') as f:f.write(page.replace(anchor,anchor+notice,1))
    write(THICK_OUT/'reviewed.json',{'status':'completed','source':file_record(source),'reviewed':file_record(target),'late_growth_L100':notes,'note':'Original rendering preserved; review adds observed weak late growth below the gross-growth threshold.'})


def render_layered() -> None:
    from scripts.dev_1_5_geometry_frequency.pml_layered import ROOT as root
    out=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/pml_layered_cached_v1';out.mkdir(parents=True,exist_ok=False)
    metrics=read_json(root/'analysis/metrics.json');cases=mapping(metrics['cases'])
    with np.load(root/'analysis/curves.npz') as z:data={k:z[k] for k in z.files}
    t=data['time_s'];x=data['x_m'];physical=data['physical'];ft=data['field_time_s']
    labels=sorted((label for label in cases if label.startswith(('L','F'))),key=lambda label:(label[0],number(mapping(cases[label])['thickness_m'])))
    fig,axes=plt.subplots(1,3,figsize=(17,5),layout='constrained')
    ds=[];l2=[];pk=[];times=[]
    for label in labels:
        row=mapping(cases[label]);ds.append(number(row['thickness_m']));rat=np.asarray(mapping(mapping(row['ratios_error_to_physical'])['all'])['full'],dtype=float)
        l2.append(rat[0]);pk.append(number(mapping(mapping(row['central_pp_window_ratios'])['5'])['full_surface_both_peak_ratio']));times.append(number(row['run_seconds']))
        if row['monotone_late_pml_growth'] and not row['strong_growth']:
            axes[0].scatter(ds[-1],l2[-1],marker='x',c='red',s=90,zorder=5);axes[1].scatter(ds[-1],pk[-1],marker='x',c='red',s=90,zorder=5)
    visible=np.array([not mapping(cases[label])['strong_growth'] for label in labels])
    for family,title in [('L','L: постоянный интеграл'),('F','F: постоянный максимум')]:
        family_mask=np.array([label.startswith(family) for label in labels]);good=visible&family_mask
        axes[0].semilogy(np.array(ds)[good],np.array(l2)[good],'o-',label=title);axes[1].semilogy(np.array(ds)[good],np.array(pk)[good],'o-',label=title);axes[2].plot(np.array(ds)[family_mask],np.array(times)[family_mask],'o-',label=title)
    axes[2].legend(fontsize=8)
    for ax,title in zip(axes,['L2 ошибки / L2 отражений интерфейса','Глобальный пик ошибки / пик пятого PP-окна','Время интегрирования, с']):
        ax.set(xlabel='Толщина PML, м',title=title);ax.grid(alpha=.3)
    for ax in axes[:2]:ax.axhline(.01,color='green',ls='--',label='100× слабее');ax.axhline(.001,color='gray',ls=':',label='1000× слабее');ax.legend()
    fig.suptitle('p5, h5 м, двухслойная1400×700 м,30 Гц; красный × — поздний рост; сильный рост исключён из графиков ошибки')
    fig.savefig(out/'tradeoff.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    for label in labels:
        if label not in ('L10','L50','L100','L175','L350','L700','F350','F500','F700','F1000','F1200'):continue
        axes[0].semilogy(ft,np.maximum(data[label+'_pml_peak'],1e-30),label=label)
        if not mapping(cases[label])['strong_growth']:
            axes[1].semilogy(t,np.maximum(np.max(data[label+'_surface_error_envelope'],axis=1),1e-30),label=label)
    axes[1].semilogy(t,np.maximum(np.max(abs(physical),axis=(1,2)),1e-30),'k',lw=2,label='Физические отражения')
    for ax,title in zip(axes,['Полный пик внутри PML','Поверхность: ошибка и отражения интерфейса']):ax.set(xlabel='Время, с',ylabel='м/с',title=title,ylim=(1e-16,None));ax.grid(alpha=.3);ax.legend(ncol=2,fontsize=8)
    fig.savefig(out/'decay.png',dpi=150);plt.close(fig)
    # Same linear color scale: magnify boundary errors 100 times to visualize the criterion.
    show=['physical']+[label for label in ('L175','F350','F500','F700','F1000','F1200') if label in cases]
    fig,axes=plt.subplots(len(show),2,figsize=(14,3*len(show)),layout='constrained')
    limit=float(abs(physical).max())
    for i,label in enumerate(show):
        values=physical if label=='physical' else 100*data[label+'_error']
        title='Вклад интерфейса' if label=='physical' else f'{label}: ошибка ×100'
        for c,component in enumerate(('Vx','Vy')):
            im=axes[i,c].pcolormesh(x,t,values[:,:,c],cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest',rasterized=True)
            axes[i,c].set(title=f'{title}, {component}',xlabel='x, м',ylabel='Время, с',ylim=(1,0));fig.colorbar(im,ax=axes[i,c],label='м/с (ошибка усилена ×100)')
    fig.savefig(out/'seismograms.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,12),layout='constrained')
    for i,target in enumerate((800,1000,1200,1400)):
        j=int(np.argmin(abs(x-target)))
        for c,component in enumerate(('Vx','Vy')):
            ax=axes[i,c];ax.plot(t,physical[:,j,c],color='black',label='Отражения интерфейса')
            for label in ('L175','F500','F1200'):
                ax.plot(t,100*data[label+'_error'][:,j,c],label=f'{label}, ошибка ×100',alpha=.75)
            ax.set(title=f'{component}, x={x[j]:.1f} м',xlabel='Время, с',ylabel='м/с');ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.savefig(out/'traces.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(5,1,figsize=(12,14),layout='constrained')
    j=int(np.argmin(abs(x-700)))
    for bounce,ax in enumerate(axes,1):
        center=.05+700*bounce/4087;mask=(t>=center-.02)&(t<=center+.02)
        denom=float(abs(physical[mask,j,1]).max())
        ax.plot(t[mask],physical[mask,j,1]/denom,color='black',lw=2,label='Вклад интерфейса / пик окна')
        for label in ('L175','F500','F1200'):
            ax.plot(t[mask],100*data[label+'_error'][mask,j,1]/denom,label=f'{label}, ошибка ×100 / пик окна')
        ax.set(title=f'Vy в центре, окно около PP{bounce}; пик интерфейса {denom:.3e} м/с',xlabel='Время, с',ylabel='Относительная амплитуда',ylim=(-1.2,1.2));ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.savefig(out/'echo_windows.png',dpi=150);plt.close(fig)
    rows=[]
    for label,value in cases.items():
        row=mapping(value);rat=np.asarray(mapping(mapping(row['ratios_error_to_physical'])['all'])['full'],dtype=float)
        growth='сильный' if row['strong_growth'] else 'поздний монотонный' if row['monotone_late_pml_growth'] else 'не выявлен этим критерием'
        rows.append(f'<tr><td>{label}</td><td>{number(row["thickness_m"]):g}</td><td>{rat[0]:.5g}</td><td>{rat[1]:.5g}</td><td>{number(mapping(mapping(row["central_pp_window_ratios"])["5"])["full_surface_both_peak_ratio"]):.5g}</td><td>{number(row["run_seconds"]):.2f}</td><td>{growth}</td><td>{"да" if row["passes_100x_all_five_pp_global_peak"] else "нет"}</td></tr>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Двухслойная модель: толщина PML</title><style>body{font:16px system-ui;max-width:1400px;margin:30px auto;padding:0 24px}img{max-width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}table{border-collapse:collapse;width:100%}</style><h1>Толщина PML относительно физических отражений</h1><p>Новая версия seismic с кэшированием коэффициентов. Только p5/h5 м, центральный Рикер30 Гц,1 Н/м,1 с, β=1/27. Основная область1400×700 м, интерфейс y=−350 м; Vp4087/3963, Vs2431,62036/2357,84473 м/с, плотность1000 кг/м³. Квадратичный PML: варианты L имеют dmax=173,088611·175/L, варианты F — фиксированный максимум173,088611 с⁻¹; внешний ABC, свободный верх. Общий dt=1/53000 с.</p><p>Физическое отражённое поле = расширенная двухслойная модель минус расширенная однородная. Это весь вклад интерфейса, включая конверсию и кратные волны, а не разложение на чистые фазы. Ошибка = кандидат минус расширенная двухслойная модель: сюда входят отражения от входа/внешнего края PML и возможный численный рост. Контроль6400×3200 м, интерфейс не масштабируется; геометрический возврат внешнего края позже1,39 с.</p><p>Общий критерий: ошибка как минимум в100 раз слабее физического отражённого поля по взвешенной L2 и пику за0–1 с на x0…1400. Более строгий итоговый критерий требует, чтобы глобальный максимум ошибки на всей поверхности, обе компоненты за всю секунду, был в100 раз меньше пика каждого из пяти PP-окон Vy в центральной полосе ±200 м. Это консервативное сравнение слабеющих отражений, а не только самого сильного. Компоненты, пространственные полосы, отдельные временные окна и окна около пяти кратных PP-приходов сохранены в JSON. Критерий общей нормы не гарантирует100× относительно каждой слабой кратной волны; это отдельно видно на последних графиках. Красные отметки означают поздний рост полного поля: хорошей поверхностной метрики недостаточно. Сильный рост исключён из графиков качества, но сохранён в таблице и диагностике полного поля. Отсутствие флага не доказывает длительную устойчивость.</p><table><tr><th>Вариант</th><th>L, м</th><th>L2 ошибки / отражений</th><th>Пик ошибки / главных отражений</th><th>Пик ошибки / PP5-окна</th><th>run, с</th><th>Рост PML</th><th>100× для пяти окон</th></tr>'''+''.join(rows)+'''</table><img src="tradeoff.png"><img src="decay.png"><h2>Линейные сейсмограммы</h2><p>Для сравнения с порогом100× ошибки усилены ровно в100 раз; шкала у всех панелей общая. Превышения насыщают цвет. Это диагностическое усиление, исходные массивы не изменены.</p><img src="seismograms.png"><img src="traces.png"><h2>Слабые последующие отражения</h2><p>Окна ±20 мс около геометрических кратных PP-приходов на центральной трассе. Возможна смесь фаз. Каждое окно нормировано на собственный пик физического вклада; ошибка усилена100×. Кривые вне ±1,2 обрезаны: они не выполняют показанный порог.</p><img src="echo_windows.png"><p><a href="../../../data/dev_1_5_geometry_frequency/pml_layered_cached_saturn_v1/analysis/metrics.json">Полные численные метрики</a></p></html>'''
    (out/'index.html').write_text(page)
    write(out/'render.json',{'status':'completed','source':file_record(root/'analysis/metrics.json'),'files':{p.name:file_record(p) for p in out.iterdir() if p.is_file()}})
    print('LAYERED RENDER COMPLETED',flush=True)


def review_layered() -> None:
    from scripts.dev_1_5_geometry_frequency.pml_layered import ROOT as root
    out=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/pml_layered_cached_v1'
    metrics=read_json(root/'analysis/metrics.json');cases=mapping(metrics['cases'])
    candidates=[]
    for label,value in cases.items():
        row=mapping(value);ratios=mapping(row['central_pp_window_ratios'])
        if label.startswith('F') and not row['strong_growth'] and not row['monotone_late_pml_growth'] and all(number(mapping(ratios[str(n)])['full_surface_both_peak_ratio'])<=.01 for n in range(1,5)):
            candidates.append((number(row['thickness_m']),label))
    _,chosen=min(candidates);chosen_row=mapping(cases[chosen]);chosen_pp=mapping(chosen_row['central_pp_window_ratios'])
    assert chosen=='F550'
    with np.load(root/'analysis/curves.npz') as z:
        t=z['time_s'];x=z['x_m'];physical=z['physical'];error=z[chosen+'_error'];largest_error=z['F1200_error']
    fig,axes=plt.subplots(5,1,figsize=(12,14),layout='constrained');j=int(np.argmin(abs(x-700)))
    for bounce,ax in enumerate(axes,1):
        center=.05+700*bounce/4087;mask=(t>=center-.02)&(t<=center+.02);denom=float(abs(physical[mask,j,1]).max())
        ax.plot(t[mask],physical[mask,j,1]/denom,color='black',lw=2,label='Физическое отражённое поле / пик окна')
        ax.plot(t[mask],100*error[mask,j,1]/denom,label='PML550 м: ошибка ×100 / пик окна')
        ax.plot(t[mask],100*largest_error[mask,j,1]/denom,label='PML1200 м: ошибка ×100 / пик окна',ls='--')
        ax.set(title=f'Центральная Vy: окно{bounce}, пик {denom:.3e} м/с',xlabel='Время, с',ylabel='Относительная амплитуда',ylim=(-1.2,1.2));ax.grid(alpha=.3);ax.legend(fontsize=8)
    fig.savefig(out/'selected_echo_windows.png',dpi=150);plt.close(fig)
    factor4=1/number(mapping(chosen_pp['4'])['full_surface_both_peak_ratio']);factor5=1/number(mapping(chosen_pp['5'])['full_surface_both_peak_ratio'])
    largest=mapping(cases['F1200']);largest_factor5=1/number(mapping(mapping(largest['central_pp_window_ratios'])['5'])['full_surface_both_peak_ratio'])
    rows=[]
    for label in ('L175','F350','F500','F550','F700','F1000','F1200'):
        row=mapping(cases[label]);rr=mapping(row['central_pp_window_ratios']);factors=[1/number(mapping(rr[str(n)])['full_surface_both_peak_ratio']) for n in (3,4,5)]
        rows.append(f'<tr><td>{label}</td><td>{number(row["run_seconds"]):.1f}</td>'+''.join(f'<td>{v:.3g}×</td>' for v in factors)+'</tr>')
    verdict=f'<section style="border-left:4px solid #315e84;padding:8px 20px;background:#f4f7fa"><h2>Вывод по слабым отражениям</h2><p><b>550 м (110 рядов по5 м), квадратичный профиль с dmax173,088611 с⁻¹ и ABC</b> — наименьший проверенный вариант с запасом≥100× относительно каждого из первых четырёх окон. Для четвёртого окна запас {factor4:.1f}×; у500 м — лишь79,7×. Порог заключён между500 и550 м, точный минимум не найден.</p><p><b>Для всех пяти окон условие100× не выполнено ни в одном опыте.</b> У550 м запас относительно пятого окна {factor5:.1f}×, у1200 м — {largest_factor5:.1f}×. После500–700 м выигрыш от толщины становится небольшим; причина остаточной ошибки и её зависимость от dt/профиля этим экспериментом не установлены.1200 м нельзя объявлять достаточным решением строгого критерия.</p><p>Это сравнение глобальной ошибки на всей поверхности и по обеим компонентам за1 с с пиком Vy каждого окна в полосе±200 м. Оно консервативнее сопоставления в совпадающих окнах. Окна привязаны к геометрическим PP-приходам и могут содержать другие фазы. Значения ниже1× означают, что ошибка сильнее данной слабой волны.</p><table><tr><th>Вариант</th><th>run, с</th><th>Запас к окну3</th><th>К окну4</th><th>К окну5</th></tr>'+''.join(rows)+'</table><p>При сохранении интеграла профиля (L) утолщение само по себе не обеспечивает нужного подавления: уL700 глобальная ошибка всё ещё в941 раз больше пика пятого окна. Варианты F сохраняют максимум173,09 с⁻¹, поэтому интегральное поглощение возрастает с толщиной. Вывод относится только к данной модели, p5/h5 м,30 Гц,1 с и выбранному шагу.</p><img src="selected_echo_windows.png"><p>На графиках выше каждое центральное окно нормировано на собственный пик; ошибка усилена100×. Значения за ±1,2 обрезаются. Полные амплитуды, все пространственные полосы и отношения хранятся в JSON.</p></section>'
    source=out/'index.html';anchor='<h1>Толщина PML относительно физических отражений</h1>';page=source.read_text();assert anchor in page
    target=out/'reviewed.html'
    with target.open('x') as f:f.write(page.replace(anchor,anchor+verdict,1))
    write(out/'reviewed.json',{'status':'completed','source':file_record(source),'reviewed':file_record(target),'selected_plot':file_record(out/'selected_echo_windows.png'),'minimum_tested_four_windows_m':550,'threshold_bracket_four_windows_m':[500,550],'all_five_threshold_achieved':False,'max_tested_thickness_m':1200,'source_metrics':file_record(root/'analysis/metrics.json')})
    print('LAYERED REVIEW COMPLETED',flush=True)


def render_first_reflection() -> None:
    from scripts.dev_1_5_geometry_frequency.pml_layered import ROOT as root
    source=root/'first_reflection_v1';out=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/pml_first_reflection_v1';out.mkdir(parents=True,exist_ok=False)
    metrics=read_json(source/'metrics.json');cases=mapping(metrics['cases'])
    names={'abc_only':'Только ABC, без добавленной полосы','old':'Массовый слой175 м + ABC','L100':'PML100 м + ABC','L125':'PML125 м + ABC','L150':'PML150 м + ABC','L175':'PML175 м + ABC','F250':'PML250 м, фикс. максимум + ABC','F350':'PML350 м, фикс. максимум + ABC'}
    chosen=list(names)
    fig,axes=plt.subplots(1,2,figsize=(15,6),layout='constrained');y=np.arange(len(chosen))
    for ax,key,title in zip(axes,('energy_error_to_first','peak_vector_error_to_first'),('Энергия ошибки / энергия первого отражения','Пик ошибки / пик первого отражения')):
        vals=[number(mapping(cases[label])[key]) for label in chosen]
        ax.barh(y,vals,color=['#a54c3d' if label in ('abc_only','old') else '#376f9c' for label in chosen]);ax.set_yticks(y,[names[label] for label in chosen]);ax.invert_yaxis();ax.set_xscale('log');ax.set(xlabel='Безразмерное отношение; меньше — лучше',title=title);ax.axvline(1.,color='black',ls='--');ax.grid(axis='x',alpha=.25)
        for yy,value in zip(y,vals):ax.text(value*1.12,yy,f'{value:.3g}',va='center',fontsize=9)
        ax.set_xlim(min(vals)/3,max(vals)*8)
    fig.suptitle('Первое отражение горизонта: p5 / 5 м / 30 Гц; приёмники внутри0<x<1400 м')
    fig.savefig(out/'comparison.png',dpi=150);plt.close(fig)
    with np.load(source/'curves.npz') as z:data={key:z[key] for key in z.files}
    t=data['time_s'];x=data['x_m'];primary=data['first_reflection'];limit=float(np.abs(primary).max())
    labels=('first','abc_only','old','L100','L150','L175','F250')
    fig,axes=plt.subplots(len(labels),2,figsize=(14,3*len(labels)),layout='constrained')
    for i,label in enumerate(labels):
        values=primary if label=='first' else data[label+'_error']
        title='Первое отражение горизонта' if label=='first' else names[label]+': искусственная ошибка'
        for c,component in enumerate(('Vx','Vy')):
            im=axes[i,c].pcolormesh(x,t,values[:,:,c],cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest',rasterized=True)
            axes[i,c].set(title=f'{title}, {component}',xlabel='x, м',ylabel='Время, с',ylim=(1,0));fig.colorbar(im,ax=axes[i,c],label='м/с; одна линейная шкала, без усиления')
    fig.savefig(out/'seismograms.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(4,2,figsize=(14,11),layout='constrained')
    for i,label in enumerate(('L100','L150','L175','F250')):
        for c,component in enumerate(('Vx','Vy')):
            im=axes[i,c].pcolormesh(x,t,100*data[label+'_error'][:,:,c],cmap='RdBu_r',vmin=-limit,vmax=limit,shading='nearest',rasterized=True)
            axes[i,c].set(title=f'{names[label]}, {component}: ошибка ×100',xlabel='x, м',ylabel='Время, с',ylim=(1,0));fig.colorbar(im,ax=axes[i,c],label='м/с после усиления ×100')
    fig.savefig(out/'pml_errors_x100.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(3,1,figsize=(12,9),layout='constrained');j=int(np.argmin(abs(x-700)))
    for ax,label in zip(axes,('abc_only','old','L175')):
        ax.plot(t,primary[:,j,1],color='black',lw=2,label='Первое отражение горизонта');ax.plot(t,data[label+'_error'][:,j,1],label='Искусственная ошибка',alpha=.8)
        ax.set(title=names[label]+'; центральная Vy',xlabel='Время, с',ylabel='м/с',ylim=(-1.2*limit,1.2*limit));ax.grid(alpha=.3);ax.legend()
    fig.savefig(out/'central_traces.png',dpi=150);plt.close(fig)
    rows=[]
    def sort_key(label: str) -> tuple[int,float]:
        return (0 if label=='abc_only' else 1 if label=='old' else 2 if label.startswith('L') else 3,number(mapping(cases[label])['thickness_m']))
    for label in sorted(cases,key=sort_key):
        row=mapping(cases[label]);growth='сильный рост' if row['strong_growth'] else 'поздний рост' if row['late_growth_flag'] else 'нет флага роста'
        rows.append(f'<tr><td>{names.get(label,label)}</td><td>{number(row["thickness_m"]):g}</td><td>{number(row["dmax_s_inv"]):.2f}</td><td>{number(row["energy_error_to_first"]):.5g}</td><td>{number(row["peak_vector_error_to_first"]):.5g}</td><td>{number(row["run_seconds"]):.2f}</td><td>{growth}</td></tr>')
    old=mapping(cases['old']);pml=mapping(cases['L175']);abc=mapping(cases['abc_only'])
    page=f'''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Первое отражение: PML, массовый слой и ABC</title><style>body{{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 24px;color:#20262c}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}</style><h1>Гашение относительно первого отражения горизонта</h1><p><b>Исправление критерия по прямому уточнению пользователя:</b> сравнение ведётся с первым ярким отражением. Прежние рекомендации550 м и350 м, основанные на слабых поздних приходах, к этой задаче не относятся. Исторические результаты сохранены, но не используются как требование сохранять все кратные волны.</p><p>Двухслойная1400×700 м, интерфейс y−350 м, p5/h5 м, центральная сила Рикера вниз30 Гц/1 Н/м, β1/27,1 с, общий dt1/53000. Все PML имеют внешний ABC, верх свободный. Новый контроль: ABC прямо на боковых и нижней границах основной области, без дополнительной полосы, без массового демпфирования и PML. Старый контроль: массовый квадратичный слой175 м,k0,5,dmax173,088611+ABC.</p><p>Первый PP-приход выделяется из разности расширенных двухслойного и однородного контролей: t(x)=0,05+sqrt(700²+(x−700)²)/4087, окно±0,03 с. Последующие кратные приходы не входят в знаменатель. Числитель — разность кандидата и расширенного двухслойного контроля за всю секунду, включая все искусственные возвраты и численные ошибки. Использованы все{int(number(metrics['receiver_count']))} GLL-приёмников строго внутри основной области, без PML и узлов x0/1400; фильтрация не применялась.</p><p>Энергетическая мера:0,001·Σ wGLL(Vx²+Vy²), это сигнальная энергия, не полный механический поток. Пиковая мера: максимум sqrt(Vx²+Vy²). Пик первого отражения {number(metrics['first_peak_vector_m_s']):.6e} м/с. По отношению пиков: только ABC {number(abc['peak_vector_error_to_first']):.4g}; старый слой {number(old['peak_vector_error_to_first']):.4g}; PML175 м {number(pml['peak_vector_error_to_first']):.4g}. Общая норма, локальные отношения на отдельных приёмниках и диагностика роста сохранены в JSON.</p><img src="comparison.png"><table><tr><th>Вариант</th><th>Толщина, м</th><th>dmax, с⁻¹</th><th>Eош/Eперв</th><th>Aош/Aперв</th><th>run, с</th><th>Поле</th></tr>{''.join(rows)}</table><p>L: постоянный интеграл профиля,dmax=173,088611·175/L. F: фиксированный максимум173,088611. Варианты с численным ростом не являются допустимым способом поглощения независимо от поверхностных отношений. Отсутствие флага за1 с не доказывает длительную устойчивость. Тайминги одиночные, включают запись полей разного размера.</p><h2>Линейные сейсмограммы без усиления ошибки</h2><p>Верхний ряд — первый импульс горизонта. Остальные — искусственная ошибка. Шкала одинакова, значения за пределами насыщают цвет.</p><img src="seismograms.png"><h2>Детали PML</h2><p>Только на следующем изображении ошибка усилена100×; шкала остаётся той же.</p><img src="pml_errors_x100.png"><h2>Центральная трасса</h2><p>Первое отражение и искусственная ошибка без усиления. Пределы одинаковые; большие значения ABC выходят за рамку.</p><img src="central_traces.png"><p><a href="../../../data/dev_1_5_geometry_frequency/pml_layered_cached_saturn_v1/first_reflection_v1/metrics.json">Численные метрики</a></p></html>'''
    (out/'index.html').write_text(page)
    write(out/'render.json',{'status':'completed','source_metrics':file_record(source/'metrics.json'),'files':{p.name:file_record(p) for p in out.iterdir() if p.is_file()}})
    print('FIRST REFLECTION RENDER COMPLETED',flush=True)


def render_timing() -> None:
    from scripts.dev_1_5_geometry_frequency.pml_layered import TIMING_ROOT as root
    source=root/'analysis/metrics.json';metrics=read_json(source);cases=mapping(metrics['cases'])
    out=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/pml_timing_native_cfl_v1';out.mkdir(parents=True,exist_ok=False)
    labels=('abc_only','L100','L175','F250','P175_noabc');names=('Без PML, ABC','PML100 м + ABC','PML175 м + ABC','PML250 м + ABC','PML175 м без ABC')
    fig,ax=plt.subplots(figsize=(11,6),layout='constrained');x=np.arange(len(labels))
    for j,order in enumerate((4,5)):
        timing_rows=[mapping(cases[f'p{order}_{label}']) for label in labels];values=np.array([number(r['median_run_seconds']) for r in timing_rows]);lo=values-np.array([number(r['min_run_seconds']) for r in timing_rows]);hi=np.array([number(r['max_run_seconds']) for r in timing_rows])-values
        xpos=x+(j-.5)*.36;ax.bar(xpos,values,width=.34,label=f'p{order}',color=('#377ba3','#d18a43')[j],yerr=np.stack((lo,hi)),capsize=4)
        for xx,value in zip(xpos,values):ax.text(xx,value+.6,f'{value:.2f}',ha='center')
    ax.set(xticks=x,xticklabels=names,ylabel='Время model.run, с',title='Saturn A100 · двухслойная1400×700 м · h5 м ·30 Гц ·1 с');ax.legend();ax.grid(axis='y',alpha=.25)
    ax.set_ylim(0,ax.get_ylim()[1]*1.12);fig.suptitle('Медиана трёх повторов; усики — минимум/максимум; запись полей включена')
    fig.savefig(out/'timings.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for j,order in enumerate((4,5)):
        for label,name in zip(labels[1:],names[1:]):
            r=mapping(cases[f'p{order}_{label}']);t=np.asarray(r['field_time_s'],dtype=float)
            axes[j,0].semilogy(t,np.maximum(np.asarray(r['pml_peak_m_s'],dtype=float),1e-30),label=name)
            axes[j,0].semilogy(t,np.maximum(np.asarray(r['half_dt_pml_peak_m_s'],dtype=float),1e-30),ls='--',alpha=.7)
            axes[j,1].semilogy(t[t>=.3],np.maximum(np.asarray(r['core_kinetic_j_per_m'],dtype=float)[t>=.3],1e-40),label=name)
        axes[j,0].set(title=f'p{order}: максимум |Vx,Vy| внутри PML',xlabel='t, с',ylabel='м/с',xlim=(.3,1),ylim=(1e-15,1e-8));axes[j,0].legend();axes[j,0].grid(alpha=.25)
        axes[j,1].set(title=f'p{order}: кинетическая энергия основной области',xlabel='t, с',ylabel='Дж/м',xlim=(.3,1));axes[j,1].legend();axes[j,1].grid(alpha=.25)
    fig.suptitle('Сплошная — шаг из оценки CFL0,8; пунктир на левом графике — dt/2')
    fig.savefig(out/'field_diagnostics.png',dpi=150);plt.close(fig)
    rows=[]
    for order in (4,5):
        for label,name in zip(labels,names):
            r=mapping(cases[f'p{order}_{label}']);temporal=mapping(mapping(r['temporal_surface'])['after_015'])
            growth='сильный рост' if r['strong_growth_flag'] else 'монотонный поздний рост PML' if r['monotone_late_pml_growth_flag'] else 'нет флага роста за1 с'
            rows.append(f'<tr><td>p{order}</td><td>{name}</td><td>{number(r["native_dt_s"])*1e6:.6f}</td><td>{int(number(r["steps"]))}</td><td>{number(r["median_run_seconds"]):.3f}</td><td>{number(r["min_run_seconds"]):.3f}–{number(r["max_run_seconds"]):.3f}</td><td>{number(r["median_total_seconds"]):.3f}</td><td>{number(temporal["relative_l2"])*100:.6g}%</td><td>{number(temporal["max_vector_delta_to_first_horizon"])*100:.6g}%</td><td>{number(r["temporal_core_fields_relative_l2"])*100:.6g}%</td><td>{growth}</td></tr>')
    page='''<!doctype html><html lang="ru"><meta charset="utf-8"><title>Время PML p4/p5 и проверка шага</title><style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px;color:#20262c}img{max-width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:nth-child(2){text-align:left}</style><h1>Время исполнения и чувствительность к шагу</h1><p>Общий lin_elast_2d_seismic_newmark_kim_cu, текущие исходники побайтово совпадают со снимком seismic_pml_cached_v1. Двухслойная1400×700 м, интерфейс−350 м, p4/p5,h5 м, Рикер30 Гц вниз в(700,0),1 Н/м,1 с, β1/27. Верх свободный; на боках и дне ABC, кроме дополнительного контроля PML175 м без ABC со свободным внешним краем. Нет массового демпфирования. Без PML область не расширяется. Максимумы квадратичного PML:100 м302,905 с⁻¹;175 и250 м173,089 с⁻¹. Это варианты прежнего сравнения, а не серия с постоянным максимумом для всех толщин.</p><p><b>Выбор шага:</b> без PML — native CFL0,8. С активным PML решатель всё ещё отвергает cfl_safety; отказ воспроизведён при обоих порядках. С прямого разрешения пользователя шаг оценён решателем на каждой той же сетке с материалом и соответствующим оформлением внешней границы, но без PML, затем передан явно в полную модель. Следовательно, это не спектральная оценка связанной системы с памятью. У контроля без ABC спектральная оценка не содержит демпфирования внешней границы; изменение времени может включать изменение числа шагов, а не только исключение работы ABC. Такт0,001 с, длительность1 с.</p><p>Все50 полных запусков последовательно на одной A100 GPU0:10 прогревов,30 измерений,10 контролей половинного шага. Время run включает синхронное интегрирование и регистрацию21 полного поля/1001 сейсмограммы; исключает сборку, prepare и конечное SHA-хеширование. Показана медиана трёх повторов, диапазон — минимум/максимум. Полное время включает сборку/prepare и сохранение поверхности до конечного хеширования. CFL-probe для PML выполняется отдельно и не входит в run.</p><img src="timings.png"><table><tr><th>Порядок</th><th>Вариант</th><th>dt, мкс</th><th>Шагов</th><th>run, с</th><th>Диапазон, с</th><th>Полное, с</th><th>Δdt L2 поверхности</th><th>Пик Δdt / первое отражение</th><th>Δdt L2 полей</th><th>Диагностика</th></tr>'''+''.join(rows)+'''</table><p>Δdt означает разность расчёта с основным шагом и dt/2, не точную ошибку. Поверхностные нормы взвешены GLL, только0&lt;x&lt;1400 м и вне20 м от источника, время0,15–1 с; поля сравниваются по массе основной области на21 кадре за0–1 с. Масштаб первого отражения взят из прежнего расширенного двухслойного контроля. При каждом порядке сравниваются одинаковые узлы; p4 и p5 напрямую не вычитаются. Конечность данных, воспроизведение повторов, совпадение поверхности/полей и SHA проверены. Отсутствие флагов роста за1 с не доказывает длительной устойчивости PML.</p><img src="field_diagnostics.png"><p><a href="../../../data/dev_1_5_geometry_frequency/pml_timing_native_cfl_v1/analysis/metrics.json">Все метрики, шаги, спектры, повторяемость и диагностика</a></p></html>'''
    (out/'index.html').write_text(page)
    write(out/'render.json',{'status':'completed','source_metrics':file_record(source),'files':{p.name:file_record(p) for p in out.iterdir() if p.is_file()}})
    print('TIMING RENDER COMPLETED',flush=True)


if __name__=='__main__':main()
