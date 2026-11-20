"""Render completed chapter 1.5 fields with shared fixed signed physical scales."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import hashlib
import html
import json
import shutil
import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import SymLogNorm
from scripts.dev_1_5_geometry_frequency.setup import ROOT,Spec,specs,write
from scripts.dev_1_3_point_source.run import file_record
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json,mapping,number

IMAGES=PROJECT_ROOT/'img/dev_1_5_geometry_frequency/series_v1'
NAMES={'efvb':'EFVB00000','homogeneous':'Однородная','two_layer':'Два слоя'}
FloatArray=NDArray[np.float64]


def scales() -> None:
    IMAGES.mkdir(parents=True,exist_ok=True)
    fields=np.zeros(2);surface=np.zeros(2);records:dict[str,object]={}
    for spec in specs():
        result=read_json(ROOT/'runs'/spec.run_id/'result.json');assert result['status']=='completed'
        fields=np.maximum(fields,np.asarray(result['field_peak_m_s'],dtype=np.float64))
        surface=np.maximum(surface,np.asarray(result['surface_peak_m_s'],dtype=np.float64))
        records[spec.run_id]=result
    write(ROOT/'summary.json',{'runs':records,'count':len(records),'global_field_peaks_m_s':fields.tolist(),
        'global_surface_peaks_m_s':surface.tolist(),'render_note':'fixed global per-component signed SymLog scale; linthresh=1e-4 global peak; bilinear interpolation only for movie raster'})


def verify(path: Path, meta: dict[str,object]) -> None:
    assert file_record(path)['sha256']==meta['sha256'],path


def render(spec: Spec) -> None:
    encoder = shutil.which('ffmpeg')
    if encoder is None:
        binaries = list((PROJECT_ROOT / '.runtime_ch15_ffmpeg/imageio_ffmpeg/binaries').glob('ffmpeg-*'))
        if len(binaries) != 1:
            raise RuntimeError('Install isolated imageio-ffmpeg==0.6.0 into .runtime_ch15_ffmpeg')
        encoder = str(binaries[0])
    matplotlib.rcParams['animation.ffmpeg_path'] = encoder
    base=ROOT/'runs'/spec.run_id
    meta=read_json(base/'result.json');summary=read_json(ROOT/'summary.json')
    target=IMAGES/spec.run_id;target.mkdir(parents=True,exist_ok=False)
    for name,key in (('nodes.npy','nodes'),('velocity.npy','fields'),('surface.npz','surface')):verify(base/name,mapping(meta[key]))
    coords=np.load(base/'nodes.npy',mmap_mode='r');fields=np.load(base/'velocity.npy',mmap_mode='r')
    rounded=np.round(coords,8);xs=np.unique(rounded[:,0]);ys=np.unique(rounded[:,1]);sort=np.lexsort((rounded[:,0],rounded[:,1]))
    assert len(xs)*len(ys)==len(coords)
    ordered=rounded[sort].reshape(len(ys),len(xs),2)
    assert np.all(ordered[:,:,0]==xs[None,:]) and np.all(ordered[:,:,1]==ys[:,None])
    gx=np.linspace(xs[0],xs[-1],min(901,len(xs)));gy=np.linspace(ys[0],ys[-1],min(601,len(ys)))
    ix=np.clip(np.searchsorted(xs,gx,side='right')-1,0,len(xs)-2);iy=np.clip(np.searchsorted(ys,gy,side='right')-1,0,len(ys)-2)
    wx=(gx-xs[ix])/(xs[ix+1]-xs[ix]);wy=(gy-ys[iy])/(ys[iy+1]-ys[iy])
    def raster(frame: FloatArray) -> FloatArray:
        v=frame[sort].reshape(len(ys),len(xs),2)
        low=v[iy[:,None],ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None],ix[None,:]+1]*wx[None,:,None]
        high=v[iy[:,None]+1,ix[None,:]]*(1-wx)[None,:,None]+v[iy[:,None]+1,ix[None,:]+1]*wx[None,:,None]
        return low*(1-wy)[:,None,None]+high*wy[:,None,None]
    peaks=np.asarray(summary['global_field_peaks_m_s'],dtype=np.float64)
    norms=[SymLogNorm(linthresh=p*1e-4,vmin=-p,vmax=p,base=10) for p in peaks]
    fig,axes=plt.subplots(1,2,figsize=(14,5.6),layout='constrained')
    shown=raster(fields[0]);artists=[]
    depths=np.asarray(mapping(meta['geometry'])['depths_m'],dtype=np.float64)
    for c,ax in enumerate(axes):
        artists.append(ax.imshow(shown[:,:,c],extent=(xs[0],xs[-1],ys[0],ys[-1]),origin='lower',cmap='RdBu_r',norm=norms[c],interpolation='nearest',aspect='equal'))
        for x in (0,spec.width):ax.axvline(x,color='k',ls='--',lw=.7,alpha=.6)
        ax.axhline(-spec.height,color='k',ls='--',lw=.7,alpha=.6)
        for d in depths[1:-1]:ax.axhline(-d,color='k',lw=.5,alpha=.4)
        ax.plot(spec.width/2,0,'kv',ms=5,clip_on=False)
        ax.set(xlabel='x, м',ylabel='y, м',title=('Vx' if c==0 else 'Vy')+', м/с')
        fig.colorbar(artists[-1],ax=ax,shrink=.75,label='м/с, знаковая логарифмическая шкала')
    title=fig.suptitle('')
    writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-crf','20','-pix_fmt','yuv420p'])
    with writer.saving(fig,str(target/'velocity.mp4'),dpi=110):
        for i,frame in enumerate(fields):
            values=raster(frame)
            for c,artist in enumerate(artists):artist.set_data(values[:,:,c])
            title.set_text(f'{NAMES[spec.material]} · {spec.width}×{spec.height} м · {spec.frequency} Гц · p5 / ≈5 м\nt={i*.05:.2f} с · слой175 м, k=0,5 + ABC · β=1/27')
            writer.grab_frame()
            if i in (5,10):fig.savefig(target/f'frame_{i*.05:.2f}.png',dpi=130)
    plt.close(fig)
    with np.load(base/'surface.npz') as z:
        times=z['time_s'];rx=z['coordinates_m'][:,0];velocity=z['velocity_m_s']
    surface_peaks=np.asarray(summary['global_surface_peaks_m_s'],dtype=np.float64)
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained')
    for c,ax in enumerate(axes):
        norm=SymLogNorm(linthresh=surface_peaks[c]*1e-4,vmin=-surface_peaks[c],vmax=surface_peaks[c],base=10)
        artist=ax.pcolormesh(rx,times,velocity[:,:,c],shading='nearest',cmap='RdBu_r',norm=norm,rasterized=True)
        ax.invert_yaxis()
        for x in (0,spec.width):ax.axvline(x,color='k',ls='--',lw=.8)
        ax.set(xlabel='x приёмника, м',ylabel='t, с',title=('Vx' if c==0 else 'Vy')+', м/с')
        fig.colorbar(artist,ax=ax,shrink=.8,label='м/с, знаковая логарифмическая шкала')
    fig.suptitle(f'{NAMES[spec.material]} · {spec.width}×{spec.height} м · {spec.frequency} Гц\nВсе {len(rx)} узлов свободного верха, включая слой · 1001 отсчёт через 0,001 с')
    fig.savefig(target/'seismogram.png',dpi=140);plt.close(fig)
    write(target/'render.json',{'spec':asdict(spec),'frames':21,'fps':5,'source':meta['fields'],'all_input_sha_verified':True,
        'video':file_record(target/'velocity.mp4'),'seismogram':file_record(target/'seismogram.png'),'field_raster':[len(gx),len(gy)],
        'global_field_peaks_m_s':peaks.tolist(),'global_surface_peaks_m_s':surface_peaks.tolist(),
        'ffmpeg':file_record(Path(encoder))})
    print('rendered',spec.run_id,flush=True)


def gallery() -> None:
    summary=read_json(ROOT/'summary.json');rows=[]
    for spec in specs():
        target=IMAGES/spec.run_id;meta=read_json(target/'render.json');assert meta['frames']==21
        rows.append(f'<article><h2>{html.escape(NAMES[spec.material])} · {spec.width}×{spec.height} м · {spec.frequency} Гц</h2><video controls preload="none" poster="{spec.run_id}/frame_0.25.png" src="{spec.run_id}/velocity.mp4"></video><p><a href="{spec.run_id}/velocity.mp4">MP4: Vx и Vy</a> · <a href="{spec.run_id}/seismogram.png">Сейсмограмма Vx и Vy</a></p><a href="{spec.run_id}/seismogram.png"><img loading="lazy" src="{spec.run_id}/seismogram.png"></a></article>')
    content='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Глава 1.5 — 30 опытов</title><style>body{font:16px system-ui;background:#f2f4f7;color:#17212b;max-width:1500px;margin:30px auto;padding:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(450px,1fr));gap:24px}article{background:white;padding:18px;border-radius:12px}video,img{width:100%}h2{font-size:19px}a{color:#1263a1}</style><h1>Глава 1.5: геометрия, частота и слоистость</h1><p>30 опытов · p5 · h≈5 м · 1 с · β=1/27 · CFL=0,8. Центральная точечная сила Рикера вниз, 1 Н/м. Слой175 м, квадратичный k=0,5, ABC на внешних боках и дне; верх свободный.</p><p>Анимации: 21 кадр через 0,05 с, воспроизведение 5 кадр/с (замедление 4×). Сейсмограммы: все узлы GLL верха, включая слой, через 0,001 с. Общие фиксированные знаковые SymLog-шкалы для всех опытов, отдельно для Vx/Vy и полей/сейсмограмм. Пунктир — начало слоя, тонкие линии — границы материалов. В поле выполнена только визуальная билинейная интерполяция; исходные поля сохранены во всех узлах.</p><main>'+''.join(rows)+'</main></html>'
    (IMAGES/'index.html').write_text(content)
    print('gallery',summary['count'],IMAGES/'index.html',flush=True)


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('action',choices=('scales','render','gallery'));p.add_argument('--index',type=int);args=p.parse_args()
    if args.action=='scales':scales()
    elif args.action=='gallery':gallery()
    else:
        if args.index is None:raise ValueError('Need --index')
        render(specs()[args.index])


if __name__=='__main__':main()
