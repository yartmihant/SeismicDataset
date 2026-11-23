"""Пять вариантов поглощения на общей квадратной сетке1050м."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from numpy.typing import NDArray
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.dev_1_4_server_preparation.square_main_controls_run import ROOT, MESH_ROOT, FACTOR, ALPHA_MAX_S_INV
from scripts.dev_1_4_server_preparation.profile_sweep_run import SweepSpec
from scripts.dev_1_4_server_preparation.no_padding_run import CoreSpec
from scripts.dev_1_4_server_preparation.surface_runner import verify_surface
from scripts.dev_1_3_point_source.run import file_record
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, read_json
FloatArray=NDArray[np.float64]

@dataclass(frozen=True)
class Case:
    name:str
    path:Path
    factor:float
    abc:bool
    label:str
    color:str
    style:str

CASES=(
    Case('main',ROOT/'main_abc',FACTOR,True,'Основная: k=0,5684 · ABC','#7823a6','-'),
    Case('one_abc',MESH_ROOT/'layer_abc',1.,True,'1) k=1 · ABC','#167448','--'),
    Case('one_no_abc',MESH_ROOT/'layer_no_abc',1.,False,'2) k=1 · без ABC','#d97706',':'),
    Case('zero_abc',ROOT/'zero_abc',0.,True,'3) k=0 · ABC','#2467b0','-.'),
    Case('zero_no_abc',MESH_ROOT/'reflecting',0.,False,'4) k=0 · без ABC','#444444','--'))

def main() -> None:
    report=ROOT/'comparison.json'
    if report.exists():raise FileExistsError(report)
    arrays:dict[str,FloatArray]={};records:dict[str,dict[str,object]]={};reference:dict[str,object]|None=None
    coords:FloatArray|None=None
    for case in CASES:
        # Historical reflecting run carried the conservative bound in run_spec; actual physics had no damping.
        spec=CoreSpec('EFVB',0,350.,96,1.) if case.name=='zero_abc' else SweepSpec('EFVB',0,350.,96,1.,alpha_max_s_inv=(case.factor if case.factor else 1.)*ALPHA_MAX_S_INV)
        m=verify_surface(case.path,spec)
        validation=read_json(case.path/'validation.json')
        assert validation['status']=='passed' and validation['metadata']==file_record(case.path/'metadata.json')
        expected=[11000]+([32000] if case.abc else [])+([30001] if case.factor else [])+[20200]
        assert m['physics_ids']==expected
        if reference is None:reference=m
        for key in ('source_input_sha256','nodes_sha256','mesh_evidence','physical_model','dt_s','steps_count','initial_state'):
            assert m[key]==reference[key]
        assert mapping(m['native'])['sha256']==mapping(reference['native'])['sha256']
        with np.load(case.path/'surface.npz') as a:
            t=np.asarray(a['time_s'],dtype=np.float64);v=np.asarray(a['velocity_m_s'],dtype=np.float64)
            if coords is None:coords=a['coordinates_m']
            assert np.array_equal(coords,a['coordinates_m'])
        arrays[case.name]=v;tail=v[500:];peaks=np.linalg.norm(tail,axis=2).max(axis=0)
        records[case.name]={'factor':case.factor,'abc':case.abc,'l2':float(np.linalg.norm(tail)),'peak_m_s':float(peaks.max()),'receiver_peaks_m_s':peaks.tolist(),'surface':file_record(case.path/'surface.npz'),'metadata':file_record(case.path/'metadata.json')}
    main_l2=float(np.linalg.norm(arrays['main'][500:]));main_peak=float(np.linalg.norm(arrays['main'][500:],axis=2).max())
    ratios={name:{'l2_over_main':float(np.linalg.norm(v[500:])/main_l2),'peak_over_main':float(np.linalg.norm(v[500:],axis=2).max()/main_peak)} for name,v in arrays.items()}
    fig,(ax,bx)=plt.subplots(1,2,figsize=(14,5.8),width_ratios=(2.2,1),layout='constrained')
    for case in (*CASES[1:],CASES[0]):
        v=arrays[case.name];rms=np.sqrt(np.mean(v*v,axis=(1,2)))*1e9
        ax.semilogy(t,rms,color=case.color,ls=case.style,lw=2.7 if case.name=='main' else 1.6,label=case.label,zorder=5 if case.name=='main' else 2)
    ax.axvspan(.5,1.,color='grey',alpha=.10);ax.set(xlabel='Время, с',ylabel='RMS компонент Vx/Vy, нм/с',xlim=(0,1),ylim=(1e-4,3.));ax.grid(alpha=.2)
    handles,labels=ax.get_legend_handles_labels();ax.legend([handles[-1],*handles[:-1]],[labels[-1],*labels[:-1]],loc='lower left',fontsize=9)
    for index,case in enumerate(CASES):
        ratio=ratios[case.name]['l2_over_main'];bx.barh(index,ratio,color=case.color,alpha=.85);bx.text(ratio*1.06,index,f'{ratio:.3g}×',va='center',fontsize=10)
    bx.set_yticks(range(5),['Основная','1) k=1, ABC','2) k=1, без ABC','3) k=0, ABC','4) k=0, без ABC']);bx.invert_yaxis();bx.set_xscale('log');bx.axvline(1,color='black',lw=.7);bx.set(xlabel='L₂ за 0,5–1 с / L₂ основной',xlim=(.7,max(v['l2_over_main'] for v in ratios.values())*2));bx.grid(axis='x',alpha=.2)
    fig.suptitle('Однородная среда · квадрат 1050×1050 м · квадратичный слой 175 м',fontsize=14)
    image_root=PROJECT_ROOT/'img/dev_1_4_server_preparation/square_main_controls_1s_v1';image_root.mkdir(parents=True,exist_ok=False)
    fig.savefig(image_root/'five_variants.png',dpi=170);fig.savefig(image_root/'five_variants.svg');plt.close(fig)
    with report.open('x') as f:json.dump({'status':'completed','window_s':[.5,1.],'records':records,'ratios':ratios,'figure':file_record(image_root/'five_variants.png'),'code':file_record(Path(__file__)),'scope':'Same square geometry, material, source, p4/K96; k=0 keeps exterior strip as undamped medium. Main factor retained from earlier1050x875 domain, not reoptimized. Norms are not energy.'},f,ensure_ascii=False,indent=2)
    print(json.dumps(ratios,indent=2))

if __name__=='__main__':main()
