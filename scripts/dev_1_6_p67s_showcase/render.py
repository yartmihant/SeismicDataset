"""Фиксированные шкалы, Q2/GLL-растеризация, 42 пары видео/сейсмограмм."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import Normalize, SymLogNorm
from matplotlib.collections import PolyCollection
from scripts.dev_1_3_point_source.mesh import load_frozen
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray, IntArray, SOURCE_POSITIONS
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_6_p67s_showcase.mesh import ROOT, CASES
from scripts.reporting import read_json, mapping, number
IMAGES=Path('img/dev_1_6_p67s_showcase/series_v1')
LABELS={'homogeneous':'Однородная','two_layer':'Двухслойная','EFVB00000':'EFVB 00000','EFVB00001':'EFVB 00001','ECVB12404':'ECVB 12404','ECVB00065':'ECVB 00065'}


def triangles(cells:IntArray)->IntArray:
    a=cells.reshape(-1,7,7);lo=a[:,:-1,:-1].ravel();ri=a[:,:-1,1:].ravel();up=a[:,1:,:-1].ravel();ur=a[:,1:,1:].ravel()
    return np.concatenate((np.column_stack((lo,ri,ur)),np.column_stack((lo,ur,up))))


def raster_map(coords:FloatArray,cells:IntArray)->tuple[IntArray,FloatArray]:
    tri=mtri.Triangulation(coords[:,0],coords[:,1],triangles(cells))
    xx,yy=np.meshgrid(np.linspace(-350,1750,701),np.linspace(-1050,0,351))
    found=np.asarray(tri.get_trifinder()(xx.ravel(),yy.ravel()),dtype=np.int64);assert (found>=0).all()
    vertices=np.asarray(tri.triangles[found],dtype=np.int64);v=coords[vertices]
    a=v[:,1]-v[:,0];b=v[:,2]-v[:,0];q=np.column_stack((xx.ravel(),yy.ravel()))-v[:,0]
    det=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    w1=(q[:,0]*b[:,1]-q[:,1]*b[:,0])/det;w2=(a[:,0]*q[:,1]-a[:,1]*q[:,0])/det
    return vertices,np.column_stack((1-w1-w2,w1,w2))


def bounds(ax:plt.Axes,source:float)->None:
    ax.axvline(0,color='.4',ls='--',lw=.7);ax.axvline(1400,color='.4',ls='--',lw=.7);ax.axhline(-700,color='.4',ls='--',lw=.7)
    ax.plot(source,0,'v',color='black',markersize=5,clip_on=False);ax.set(xlabel='x, м',ylabel='y, м',xlim=(-350,1750),ylim=(-1050,0))


def encoder()->None:
    binary=shutil.which('ffmpeg')
    if binary is None:
        candidates=list(Path('.runtime_ch15_ffmpeg/imageio_ffmpeg/binaries').glob('ffmpeg-*'));assert len(candidates)==1;binary=str(candidates[0].resolve())
    matplotlib.rcParams['animation.ffmpeg_path']=binary


def prepare_scales(root:Path,images:Path)->None:
    images.mkdir(parents=True,exist_ok=False);peak=0.
    for name in CASES:
        record=read_json(root/name/'run/result.json');assert record['status']=='completed'
        peak=max(peak,float(np.max(np.asarray(record['full_peak_m_s'],dtype=float))),float(np.max(np.abs(np.load(root/name/'run/surface.npz')['velocity_m_s']))))
    write(images/'scales.json',{'linear_limit_m_s':1e-10,'symlog_limit_m_s':peak,'symlog_linthresh_m_s':1e-13,'note':'fixed for all cases, components, sources and frames; linear saturation is display only'})


def render_case(root:Path,images:Path,name:str)->None:
    encoder();base=root/name/'run';record=read_json(base/'result.json');target=images/name;target.mkdir(exist_ok=False)
    for filename in ('velocity.npy','nodes.npy','cells.npy','surface.npz'):
        assert file_record(base/filename)['sha256']==mapping(record[filename])['sha256']
    coords=np.load(base/'nodes.npy');cells=np.load(base/'cells.npy');fields=np.load(base/'velocity.npy',mmap_mode='r');s=np.load(base/'surface.npz');surface=s['velocity_m_s'];top=s['coordinates_m'][:,0]
    frozen=load_frozen(root/name/'mesh_q2.npz');polygons=frozen.nodes[frozen.cells[:,[0,2,8,6]]]
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    for ax,values,title in zip(axes,(frozen.vp,frozen.vs),('Vp, м/с','Vs, м/с'),strict=True):
        coll=PolyCollection(list(polygons),array=values,cmap='viridis',edgecolors='none');ax.add_collection(coll);ax.autoscale_view();ax.set_aspect('equal');bounds(ax,700);ax.set_title(title);fig.colorbar(coll,ax=ax,shrink=.7)
    fig.suptitle(LABELS[name]+' · 1400×700 м + PML350 без ABC');fig.savefig(target/'material.png',dpi=140);plt.close(fig)
    vertices,weights=raster_map(coords,cells);scales=read_json(images/'scales.json')
    for scale in ('linear','symlog'):
        norm=Normalize(-1e-10,1e-10) if scale=='linear' else SymLogNorm(linthresh=1e-13,vmin=-number(scales['symlog_limit_m_s']),vmax=number(scales['symlog_limit_m_s']))
        for shot,x in enumerate(SOURCE_POSITIONS):
            dest=target/scale/f'x{int(x):04d}';dest.mkdir(parents=True,exist_ok=False)
            fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
            for c,ax in enumerate(axes):
                im=ax.pcolormesh(top,np.arange(1001)*.001,surface[:,shot,:,c],shading='auto',cmap='RdBu_r',norm=norm,rasterized=True)
                ax.invert_yaxis();ax.axvline(0,color='.3',ls='--',lw=.7);ax.axvline(1400,color='.3',ls='--',lw=.7);ax.set(xlabel='x, м',ylabel='t, с',title=('Vx','Vy')[c]);fig.colorbar(im,ax=ax,label='м/с',extend='both')
            fig.suptitle(f'{LABELS[name]} · источник x={x:g} м · {scale}');fig.savefig(dest/'seismogram.png',dpi=140);plt.close(fig)
            fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained');artists=[]
            for c,ax in enumerate(axes):
                im=ax.imshow(np.zeros((351,701)),extent=(-350,1750,-1050,0),origin='lower',norm=norm,cmap='RdBu_r',interpolation='nearest');artists.append(im);bounds(ax,x);ax.set_title(('Vx','Vy')[c]);fig.colorbar(im,ax=ax,label='м/с',shrink=.7,extend='both')
            suptitle=fig.suptitle('');writer=FFMpegWriter(fps=5,codec='libx264',extra_args=['-crf','23','-preset','fast','-pix_fmt','yuv420p','-threads','2'])
            with writer.saving(fig,str(dest/'velocity.mp4'),dpi=110):
                for k in range(21):
                    field=np.asarray(fields[k,shot]);raster=np.einsum('qvc,qv->qc',field[vertices],weights).reshape(351,701,2)
                    for c,artist in enumerate(artists):artist.set_data(raster[:,:,c])
                    suptitle.set_text(f'{LABELS[name]} · x={x:g} м · t={k*.05:.2f} с · {scale}')
                    writer.grab_frame()
                    if k in (5,10):fig.savefig(dest/f'frame_{k*.05:.2f}.png',dpi=140)
            plt.close(fig);print('RENDERED',name,scale,x,flush=True)
    files=[file_record(p) for p in target.rglob('*') if p.is_file()]
    write(target/'render.json',{'case':name,'files':files,'videos':14,'seismograms':14,'interpolation':'linear on GLL subtriangles, visualization only'})


def gallery(root:Path,images:Path)->None:
    scale_meta=read_json(images/'scales.json');rows=[]
    for name in CASES:
        r=read_json(root/name/'run/result.json');rows.append({'case':name,'dt_s':r['dt_s'],'steps':r['steps'],'run_seconds':r['run_seconds'],'nodes':r['nodes'],'elements':r['elements']})
    style='<style>body{font:17px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#17212b}img,video{max-width:100%;width:100%}article{border-top:1px solid #ccc;padding-top:20px;margin:30px 0}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #ddd}a{color:#1263a1}</style>'
    header='<!doctype html><html lang="ru"><meta charset="utf-8">'+style
    table='<table><tr><th>Модель</th><th>Узлы</th><th>Элементы</th><th>dt, с</th><th>Шаги</th><th>Run, с</th></tr>'+''.join(f'<tr><td>{r["case"]}</td><td>{r["nodes"]}</td><td>{r["elements"]}</td><td>{number(r["dt_s"]):.8g}</td><td>{r["steps"]}</td><td>{number(r["run_seconds"]):.2f}</td></tr>' for r in rows)+'</table>'
    body='<h1>Глава1.6 · p67s · 42 независимых опыта</h1><p>Шесть геометрий, семь источников. 1400×700 м, p6/h≈10 м, Рикер30Гц вниз1Н/м, t0=0.05с, 1с. PML350м по трём сторонам, квадратичный максимум173.08861с⁻¹, без ABC и массового демпфирования. Полная область2100×1050м. β=1/27, γ=0.5, native CFL0.8, tact_dt0.001с.</p><p>Источники x175,350,525,700,875,1050,1225м считаются раздельно на общей геометрии; их поля не складываются. ECVB — сглаженные границы, x растянут вдвое. Материалы плоскопараллельно продолжены в PML. Регистрируются Vx/Vy:21 полный кадр через0.05с и1001 отсчёт во всех1261 узле верха, включая PML. Пунктир отмечает основную область. Видео5fps, замедление4×.</p><p><a href="../../../docs/dev_1_6_p67s_showcase_spec.txt">Полная постановка</a> · <a href="../../../dev_1_6_p67s_showcase_summary.md">Реферат и ограничения</a></p>'+table
    body+='<p>Run включает приёмники и ожидание асинхронных очередей; один запуск на геометрию, это не медианный бенчмарк. Проверены конечность состояния/PML и точное совпадение поверхности с полями. Эта серия не доказывает сеточную сходимость для всех материалов датасета.</p><h2>Модели и галереи</h2>'
    for name in CASES:
        body+=f'<article><h2>{LABELS[name]}</h2><img loading="lazy" src="{name}/material.png"><p><a href="{name}/linear.html">Линейная ±1e−10 м/с</a> · <a href="{name}/symlog.html">Знаковая логарифмическая</a></p></article>'
        for scale in ('linear','symlog'):
            page=f'<h1>{LABELS[name]} · {scale}</h1><a href="../index.html">Все модели и постановка</a><p>Общая фиксированная шкала для обоих компонентов, всех источников и времени. Линейная ±1e−10 м/с насыщается выше предела; symlog имеет порог1e−13 м/с. Данные не обрезаны.</p>'
            for x in SOURCE_POSITIONS:
                prefix=f'{scale}/x{int(x):04d}'
                page+=f'<article><h2>Источник x={x:g} м</h2><video controls preload="none" poster="{prefix}/frame_0.25.png" src="{prefix}/velocity.mp4"></video><p><a href="{prefix}/velocity.mp4">MP4</a> · <a href="{prefix}/seismogram.png">Сейсмограмма</a> · <a href="{prefix}/frame_0.50.png">Поле t=0.5с</a></p><img loading="lazy" src="{prefix}/seismogram.png"></article>'
            (images/name/f'{scale}.html').write_text(header+page+'</html>')
    (images/'index.html').write_text(header+body+'</html>')
    write(images/'summary.json',{'status':'completed','cases':rows,'scales':scale_meta,'videos':84,'seismograms':84,'independent_sources':42})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--images',type=Path,default=IMAGES);p.add_argument('--case',choices=CASES);p.add_argument('--scales',action='store_true');a=p.parse_args()
    if a.scales:prepare_scales(a.root,a.images)
    elif a.case:render_case(a.root,a.images,a.case)
    else:gallery(a.root,a.images)
