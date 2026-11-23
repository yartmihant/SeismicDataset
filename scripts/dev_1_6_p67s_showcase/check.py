"""Проверки численной серии и автономности публикуемой галереи главы1.6."""
from __future__ import annotations
import argparse
from html.parser import HTMLParser
import json
import re
from pathlib import Path
import shutil
import subprocess
from urllib.parse import unquote,urlsplit
import numpy as np
from PIL import Image
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_6_p67s_showcase.mesh import ROOT, CASES
from scripts.dev_1_6_p67s_showcase.render import IMAGES
from scripts.reporting import read_json,mapping,number,sequence


class Links(HTMLParser):
    def __init__(self)->None:
        super().__init__();self.links:list[str]=[]
    def handle_starttag(self,tag:str,attrs:list[tuple[str,str|None]])->None:
        self.links.extend(value for key,value in attrs if key in ('href','src','poster') and value is not None)


def numerical(root:Path)->None:
    records:dict[str,object]={};verified=0
    for name in CASES:
        base=root/name/'run';r=read_json(base/'result.json');assert r['status']=='completed'
        for filename in ('velocity.npy','surface.npz','nodes.npy','cells.npy'):
            assert file_record(base/filename)['sha256']==mapping(r[filename])['sha256'];verified+=1
        fields=np.load(base/'velocity.npy',mmap_mode='r');surface=np.load(base/'surface.npz')['velocity_m_s']
        assert fields.shape[:2]==(21,7) and surface.shape==(1001,7,1261,2)
        for t in range(21):
            assert np.isfinite(fields[t]).all();np.testing.assert_array_equal(fields[t,:,:1261],surface[t*50])
        assert np.isfinite(surface).all() and r['terminal_state_finite'] is True
        check:dict[str,object]={'surface_fields_exact':True,'finite':True,'shape':list(fields.shape)}
        if not name.startswith('ECVB'):
            mirror=surface[:,::-1,::-1].copy();mirror[:,:,:,0]*=-1
            delta=float(np.linalg.norm(surface-mirror)/np.linalg.norm(surface));assert delta<1e-8
            check['mirror_relative_l2']=delta
        if name=='two_layer':
            baseline=Path('data/dev_1_5_geometry_frequency/p67s_receiver_v1/new_sync/repeat0/surface.npz')
            if baseline.exists():
                old=np.load(baseline)['velocity_m_s'].reshape(1001,7,1261,2)
                delta=float(np.linalg.norm(surface-old)/np.linalg.norm(old));assert delta<1e-8
                check['chapter15_repeat_relative_l2']=delta;check['chapter15_surface']=file_record(baseline)
        records[name]=check
    write(root/'numerical_validation.json',{'status':'passed','cases':records,'verified_sha_files':verified,'limitation':'not an independent spatial convergence or long-time PML proof'})


def artifacts(images:Path,output:Path)->None:
    videos=list(images.rglob('*.mp4'));pngs=list(images.rglob('*.png'));pages=list(images.rglob('*.html'));links=0
    assert len(videos)==84 and len(pages)==13
    ffprobe=shutil.which('ffprobe');assert ffprobe is not None
    for p in videos:
        result=json.loads(subprocess.check_output([ffprobe,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate,width,height','-of','json',str(p)],text=True))['streams'][0]
        assert int(result['nb_read_frames'])==21 and result['r_frame_rate']=='5/1'
    for p in pngs:
        with Image.open(p) as im:im.verify()
    for p in pages:
        parser=Links();parser.feed(p.read_text())
        for link in parser.links:
            parsed=urlsplit(link)
            if parsed.scheme:continue
            assert (p.parent/unquote(parsed.path)).exists(),(p,link)
            links+=1
    for p in images.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            parsed=urlsplit(link)
            if parsed.scheme:continue
            assert (p.parent/unquote(parsed.path)).exists(),(p,link)
            links+=1
    verified_render_sha=0
    for name in CASES:
        record=read_json(images/name/'render.json')
        for item in sequence(record['files']):
            original=mapping(item);path=Path(str(original['path']));relative=path.relative_to('/home/antonov/SeismicDataset') if path.is_absolute() else path
            assert file_record(relative)['sha256']==original['sha256'];verified_render_sha+=1
    files=[file_record(p) for p in sorted(images.rglob('*')) if p.is_file()]
    write(output,{'status':'passed','videos':len(videos),'pngs':len(pngs),'html':len(pages),'links':links,'verified_render_sha':verified_render_sha,'files':files,'bytes':sum(p.stat().st_size for p in images.rglob('*') if p.is_file())})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--images',type=Path,default=IMAGES);p.add_argument('--artifacts',action='store_true');a=p.parse_args()
    if a.artifacts:artifacts(a.images,a.root/'artifact_validation.json')
    else:numerical(a.root)
