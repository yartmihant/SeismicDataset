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
from scripts.dev_1_5_geometry_frequency.pml_setup import ROOT,W,H,VP,VS,ALPHA
from scripts.dev_1_5_geometry_frequency.pml_analyze import MODES,traces
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
    argparse.ArgumentParser().parse_args();OUT.mkdir(parents=True,exist_ok=False)
    for order in (4,5):movies(order);figures(order)
    report();print('RENDER COMPLETED',OUT,flush=True)


if __name__=='__main__':main()
