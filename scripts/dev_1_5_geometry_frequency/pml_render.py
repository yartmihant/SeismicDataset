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
    parser=argparse.ArgumentParser();parser.add_argument('--thickness',action='store_true');args=parser.parse_args()
    if args.thickness:render_thickness();return
    OUT.mkdir(parents=True,exist_ok=False)
    for order in (4,5):movies(order);figures(order)
    report();print('RENDER COMPLETED',OUT,flush=True)


THICK_OUT=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/pml_thickness_v1'
THICK_LABELS=('L175',)+tuple(f'L{v}' for v in THICKNESSES)


def thickness_movies(order: int) -> None:
    encoder=shutil.which('ffmpeg')
    if encoder is None:
        encoder=str(next((PROJECT_ROOT/'.runtime_ch15_ffmpeg/imageio_ffmpeg/binaries').glob('ffmpeg-*')))
    matplotlib.rcParams['animation.ffmpeg_path']=encoder
    paths=dict(thickness_paths(order));indices={label:core_indices(np.load(paths[label]/'nodes.npy')) for label in THICK_LABELS}
    nodes=np.round(np.load(paths['L175']/'nodes.npy')[indices['L175']],8)
    xs=np.unique(nodes[:,0]);ys=np.unique(nodes[:,1]);assert len(xs)*len(ys)==len(nodes)
    gx=np.linspace(0,W,601);gy=np.linspace(-H,0,301)
    ix=np.clip(np.searchsorted(xs,gx,side='right')-1,0,len(xs)-2);iy=np.clip(np.searchsorted(ys,gy,side='right')-1,0,len(ys)-2)
    wx=(gx-xs[ix])/(xs[ix+1]-xs[ix]);wy=(gy-ys[iy])/(ys[iy+1]-ys[iy])
    fields={label:np.load(paths[label]/'velocity.npy',mmap_mode='r') for label in THICK_LABELS}
    def raster(label: str,i: int) -> FloatArray:
        v=fields[label][i,indices[label]].reshape(len(ys),len(xs),2)
        lo=v[iy[:,None],ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None],ix[None,:]+1]*wx[None,:,None]
        hi=v[iy[:,None]+1,ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None]+1,ix[None,:]+1]*wx[None,:,None]
        return lo*(1-wy)[:,None,None]+hi*wy[:,None,None]
    for logarithmic in (False,True):
        name='symlog' if logarithmic else 'linear';fig,axes=plt.subplots(3,4,figsize=(18,12),layout='constrained');artists=[]
        for j,label in enumerate(THICK_LABELS):
            pair=[]
            for c in range(2):
                ax=axes[j//2,(j%2)*2+c]
                a=ax.imshow(raster(label,0)[:,:,c],origin='lower',extent=(0,W,-H,0),cmap='RdBu_r',norm=norm(logarithmic),aspect='equal')
                ax.set(title=f'{label[1:]} м · p{order} · '+('Vx' if c==0 else 'Vy'),xlabel='x, м',ylabel='y, м');pair.append(a)
            artists.append(pair)
        fig.colorbar(artists[0][0],ax=list(axes.ravel()),shrink=.7,label='м/с · '+('SymLog, порог 1e−14' if logarithmic else 'линейная шкала'),extend='both')
        title=fig.suptitle('');writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-crf','20','-pix_fmt','yuv420p','-threads','2'])
        with writer.saving(fig,str(THICK_OUT/f'p{order}_{name}.mp4'),dpi=100):
            for i in range(21):
                for j,label in enumerate(THICK_LABELS):
                    v=raster(label,i)
                    for c in range(2):artists[j][c].set_data(v[:,:,c])
                title.set_text(f'PML + ABC · p{order} · t={i*.05:.2f} с · физическая область 1400×700 м\nТолщина меняется, интеграл профиля постоянен; общая шкала ±1e−10 м/с')
                writer.grab_frame()
                if i==20:fig.savefig(THICK_OUT/f'p{order}_{name}_final.png',dpi=110)
        plt.close(fig)


def thickness_figures(order: int) -> None:
    paths=dict(thickness_paths(order));t,x,ref=core_traces(paths['reference'])
    for difference in (False,True):
        fig,axes=plt.subplots(6,2,figsize=(14,18),layout='constrained');images=[]
        for row,label in enumerate(THICK_LABELS):
            _,_,v=core_traces(paths[label])
            if difference:v=v-ref
            for c in range(2):
                ax=axes[row,c];a=ax.pcolormesh(x[:,0],t,v[:,:,c],shading='nearest',cmap='RdBu_r',norm=norm(True),rasterized=True);images.append(a)
                ax.invert_yaxis();ax.set(title=f'{label[1:]} м · '+('ΔV' if difference else 'V')+('x' if c==0 else 'y'),xlabel='x, м',ylabel='t, с')
                ax.axhline(.5,color='k',ls=':',lw=.6)
        cb=fig.colorbar(images[-1],ax=list(axes.ravel()),shrink=.7,label='м/с · SymLog ±1e−10, порог 1e−14',extend='both')
        fig.suptitle(f'p{order} · PML + ABC · '+('разность с расширенным контролем' if difference else 'поверхность основной области'))
        stem=f'p{order}_surface'+('_error' if difference else '')
        fig.savefig(THICK_OUT/(stem+'.png'),dpi=120)
        for a in images:a.set_norm(norm(False))
        cb.set_label('м/с · линейная шкала ±1e−10');fig.savefig(THICK_OUT/(stem+'_linear.png'),dpi=120);plt.close(fig)
    with np.load(THICK_ROOT/'analysis'/f'p{order}_curves.npz') as z:
        fig,axes=plt.subplots(1,2,figsize=(14,5),layout='constrained')
        peak=float(z['reference_kinetic'].max())
        for label in ('old',)+THICK_LABELS:
            name='Старый слой 175 м' if label=='old' else label[1:]+' м'
            axes[0].semilogy(z['surface_time_s'],np.maximum(z[label+'_surface_error'],1e-30),label=name)
            axes[1].semilogy(z['field_time_s'],np.maximum(z[label+'_kinetic']/peak,1e-30),label=name)
        axes[1].semilogy(z['field_time_s'],np.maximum(z['reference_kinetic']/peak,1e-30),'k--',label='Контроль')
        peak_error=max(float(z[label+'_surface_error'].max()) for label in ('old',)+THICK_LABELS)
        axes[0].set(xlim=(.1,1),ylim=(peak_error*1e-6,peak_error*2),xlabel='t, с',ylabel='GLL-норма ΔVx/ΔVy',title='Ошибка поверхности: весь приход отражений')
        axes[1].set(ylim=(1e-12,2),xlabel='t, с',ylabel='K / max K контроля',title='Кинетическая энергия основной области')
        for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=8);ax.axvline(.5,color='k',ls=':',lw=.6)
        fig.suptitle(f'p{order} · квадратичный PML, ABC, dmax·L = const');fig.savefig(THICK_OUT/f'p{order}_decay.png',dpi=140);plt.close(fig)


def render_thickness() -> None:
    THICK_OUT.mkdir(parents=True,exist_ok=False);metrics=read_json(THICK_ROOT/'analysis/metrics.json');sections=[]
    fig,axes=plt.subplots(1,3,figsize=(16,5),layout='constrained')
    for order in (4,5):
        entry=mapping(mapping(metrics['orders'])[f'p{order}']);cases=mapping(entry['cases']);rows=''
        Ls=[175,*THICKNESSES]
        for ax,window in zip(axes[:2],('full','late')):
            ax.semilogy(Ls,[number(mapping(cases[label])[window+'_error_ratio_to_L175']) for label in THICK_LABELS],'-o',label=f'p{order}')
            ax.set(xlabel='Толщина PML, м',ylabel='J / J при 175 м',title='Ошибка за '+('0–1 с' if window=='full' else '0,5–1 с'));ax.axhline(1,color='k',ls=':',lw=.7)
        axes[2].plot(Ls,[number(mapping(cases[label])['run_seconds']) for label in THICK_LABELS],'-o',label=f'p{order}')
        for label in THICK_LABELS:
            row=mapping(cases[label]);rows+=f'<tr><th>{row["thickness_m"]} м / {row["rows"]}</th><td>{number(row["dmax_s_inv"]):.2f}</td><td>{number(row["full_error_ratio_to_L175"]):.3f}</td><td>{number(row["late_error_ratio_to_L175"]):.3f}</td><td>{number(row["full_error_reduction_vs_old"]):.2f}×</td><td>{number(row["run_seconds"]):.3f}</td></tr>'
        time_note=''
        if order==5:time_note=f'<p>Контроль 10 м: разность dt/dt/2 составляет {100*number(mapping(entry["time_step_check_L10"])["full_difference_over_L10_error"]):.4f}% нормы ошибки этого варианта за0–1 с.</p>'
        sections.append(f'<section><h2>p{order}</h2><table><tr><th>Толщина / ряды</th><th>dmax, с⁻¹</th><th>J0–1 / J175</th><th>J0,5–1 / J175</th><th>Уменьшение J0–1 против старого слоя</th><th>run, с</th></tr>{rows}</table>{time_note}<img src="p{order}_decay.png"><h3>Поля: общая физическая область</h3><video controls preload="none" poster="p{order}_linear_final.png" src="p{order}_linear.mp4"></video><video controls preload="none" poster="p{order}_symlog_final.png" src="p{order}_symlog.mp4"></video><p><a href="p{order}_surface.png">Сейсмограммы</a> · <a href="p{order}_surface_linear.png">Линейные</a> · <a href="p{order}_surface_error.png">Разность с контролем</a> · <a href="p{order}_surface_error_linear.png">Линейная разность</a></p><img src="p{order}_surface_error.png" loading="lazy"></section>')
    axes[2].set(xlabel='Толщина PML, м',ylabel='run, с',title='Время с записью полей; одиночные опыты')
    for ax in axes:ax.grid(alpha=.25);ax.legend()
    fig.savefig(THICK_OUT/'tradeoff.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for thickness in (175,*THICKNESSES):
        distance=np.linspace(0,thickness,200);ax.plot(distance,ALPHA*175/thickness*(distance/thickness)**2,label=f'{thickness} м')
    ax.set(xlabel='Расстояние от входа в PML, м',ylabel='d, с⁻¹',title='Квадратичные профили с одинаковым интегралом');ax.grid(alpha=.25);ax.legend();fig.savefig(THICK_OUT/'profiles.png',dpi=140);plt.close(fig)
    for order in (4,5):thickness_movies(order);thickness_figures(order)
    page="""<!doctype html><html lang="ru"><meta charset="utf-8"><title>Толщина PML</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#20252b}table{border-collapse:collapse;width:100%}td,th{padding:10px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}img,video{max-width:100%;display:block;margin:24px auto}a{color:#245a88}</style><h1>Толщина PML: качество и стоимость</h1><p>Однородная основная область1400×700 м, Vp4087 м/с, Vs2431,62036 м/с, плотность1000 кг/м³. Рикер30 Гц, центральная сила вниз1 Н/м,1 с. p4/p5, h5 м, β1/27. Квадратичный PML с внешним ABC, свободный верх. Толщина175/100/50/25/15/10 м; внешний размер области уменьшается, основная область неизменна.</p><p>dmax=173,0886108476381·175/L. Интеграл профиля сохраняется; фиксированный коэффициент при обрезании слоя не проверялся. Во всех вариантах одного порядка одинаковый dt: p4 1/36000 с, p5 1/53000 с. У p5 10 м дополнительно dt/2. Значения175 м, старый массовый слой k0,5+ABC и расширенный контроль6400×3200 м повторно используются из предыдущего исследования; native-библиотека та же.</p><p>Метрика J — пространственно взвешенная GLL-норма разности Vx/Vy с расширенным контролем на физическом верхе0…1400 м; общий множитель временного шага регистрации опущен. Основное окно0–1 с включает ранние возвраты тонких слоёв, дополнительно0,2–1 и0,5–1 с. Отношение J/J175 меньше1 означает улучшение. Кинетическая энергия не является полной упругой энергией. Поля21 кадр через0,05 с; поверхность1001 отсчёт через0,001 с. Визуализация только общей физической области, исходные массивы включают PML.</p><p>Времена включают запись полных полей, объём которой уменьшается вместе с толщиной. Это одиночные прогоны, без статистического бенчмарка; исторические времена175 м не пересняты. Основная очередь остановлена.</p><img src="tradeoff.png"><img src="profiles.png">"""+''.join(sections)+"""<p><a href="../../../data/dev_1_5_geometry_frequency/pml_thickness_saturn_v1/analysis/metrics.json">Полные метрики</a> · <a href="../pml_homogeneous_v1/index.html">Предыдущий тест175 м</a></p></html>"""
    (THICK_OUT/'index.html').write_text(page)
    write(THICK_OUT/'render.json',{'status':'completed','files':{p.name:file_record(p) for p in THICK_OUT.iterdir() if p.is_file()},'frames_per_movie':21,'fps':5,'movie_count':4,'source_metrics':file_record(THICK_ROOT/'analysis/metrics.json')})
    print('THICKNESS RENDER COMPLETED',flush=True)


if __name__=='__main__':main()
