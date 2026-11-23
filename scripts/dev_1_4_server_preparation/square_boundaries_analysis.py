"""Сравнение трёх завершённых опытов одинакового квадрата1050м."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.dev_1_4_server_preparation.square_boundaries_run import ROOT, VARIANTS
from scripts.dev_1_4_server_preparation.surface_runner import verify_surface
from scripts.dev_1_4_server_preparation.profile_sweep_run import SweepSpec, ALPHA_MAX_S_INV
from scripts.dev_1_3_point_source.run import file_record
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping

def main() -> None:
    output=ROOT/'analysis.json'
    if output.exists():raise FileExistsError(output)
    records:dict[str,dict[str,object]]={}
    arrays:dict[str,np.ndarray[tuple[int,...],np.dtype[np.float64]]]={}
    metadata:dict[str,dict[str,object]]={}
    labels={'reflecting':'Без поглощения','layer_no_abc':'Слой без ABC','layer_abc':'Слой + ABC'}
    fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
    for variant in VARIANTS:
        directory=ROOT/variant
        m=verify_surface(directory,SweepSpec('EFVB',0,350.,96,1.,alpha_max_s_inv=ALPHA_MAX_S_INV));metadata[variant]=m
        with np.load(directory/'surface.npz') as a:
            v=np.asarray(a['velocity_m_s'],dtype=np.float64);t=a['time_s'];arrays[variant]=v
        tail=v[500:1001];peak=np.linalg.norm(tail,axis=2).max(axis=0)
        records[variant]={'l2':float(np.linalg.norm(tail)),'max_speed_m_s':float(peak.max()),'receiver_peaks_m_s':peak.tolist(),'wall_seconds':m['wall_seconds'],'surface':file_record(directory/'surface.npz'),'validation':file_record(directory/'validation.json')}
        ax.semilogy(t,np.sqrt(np.mean(v*v,axis=(1,2)))*1e9,label=labels[variant])
    for m in metadata.values():
        for key in ('source_input_sha256','nodes_sha256','mesh_evidence','run_spec'):
            assert m[key]==metadata['reflecting'][key]
        assert mapping(m['native'])['sha256']==mapping(metadata['reflecting']['native'])['sha256']
    ratios={}
    for numerator,denominator in (('reflecting','layer_no_abc'),('reflecting','layer_abc'),('layer_no_abc','layer_abc')):
        a,b=arrays[numerator][500:],arrays[denominator][500:]
        ratios[f'{numerator}_over_{denominator}']={'l2':float(np.linalg.norm(a)/np.linalg.norm(b)),'peak':float(np.linalg.norm(a,axis=2).max()/np.linalg.norm(b,axis=2).max())}
    ax.axvspan(.5,1.,color='grey',alpha=.12);ax.set(xlabel='Время, с',ylabel='RMS компонент Vx/Vy, нм/с',title='Квадрат 1050×1050 м · квадратичный слой 175 м · αmax=346,177 с⁻¹',xlim=(0,1));ax.legend();ax.grid(alpha=.2)
    image=PROJECT_ROOT/'img/dev_1_4_server_preparation/square_boundaries_1s_v1';image.mkdir(parents=True,exist_ok=False)
    fig.savefig(image/'surface_rms.png',dpi=160);plt.close(fig)
    result={'status':'completed','window_s':[.5,1.],'samples':501,'records':records,'ratios':ratios,'figure':file_record(image/'surface_rms.png'),'code':file_record(Path(__file__)),'interpretation':'Surface velocity sample norms, not physical energy or modal reflection coefficients; same square domain and source in all three runs'}
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(ratios,indent=2))

if __name__=='__main__':main()
