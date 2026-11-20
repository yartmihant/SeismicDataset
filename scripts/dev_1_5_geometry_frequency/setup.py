"""Deterministic conforming rectangular meshes; no solver invocation."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from math import ceil, log
from pathlib import Path
import argparse
import json
import numpy as np
from numpy.typing import NDArray
from scripts.dev_1_3_point_source.mesh import FrozenMesh, structured_cells, save_frozen
from scripts.dev_1_4_server_preparation.model_input import read_model
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import number

ROOT = PROJECT_ROOT / 'data/dev_1_5_geometry_frequency/series_v1'
SIZES = ((350,350),(700,350),(700,700),(1400,700),(1000,500))
MATERIALS = ('efvb','homogeneous','two_layer')
ALPHA = .5 * (-3 * 4385 * log(.01) / 175)
FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class Spec:
    width: int
    height: int
    material: str
    frequency: int

    @property
    def mesh_id(self) -> str:
        return f'{self.width}x{self.height}_{self.material}'

    @property
    def run_id(self) -> str:
        return f'{self.mesh_id}_{self.frequency}Hz'


def specs() -> list[Spec]:
    return [Spec(w,h,m,f) for w,h in SIZES for m in MATERIALS for f in (15,30)]


def write(path: Path, record: dict[str,object]) -> None:
    with path.open('x') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write('\n')


def material_layers(material: str, height: int) -> tuple[FloatArray,FloatArray,dict[str,object]]:
    source=read_model('EFVB',0)
    assert np.all(source.vp==source.vp[:,:1]) and np.all(source.vs==source.vs[:,:1])
    column=np.column_stack((source.vp[:,0],source.vs[:,0]))
    changes=np.r_[0,np.flatnonzero(np.any(np.diff(column,axis=0)!=0,axis=1))+1,70]
    props=column[changes[:-1]]
    if material=='efvb':
        depths=changes/70*height
    elif material=='homogeneous':
        depths=np.array([0.,height]);props=props[:1]
    elif material=='two_layer':
        depths=np.array([0.,height/2,height]);props=props[:2]
    else:
        raise ValueError(material)
    return np.asarray(depths,dtype=np.float64),props,source.provenance


def build(spec: Spec) -> tuple[FrozenMesh,dict[str,object]]:
    assert (spec.width,spec.height) in SIZES and spec.frequency in (15,30)
    depths,props,provenance=material_layers(spec.material,spec.height)
    core=np.concatenate([np.linspace(a,b,ceil((b-a)/5)+1)[:-1] for a,b in zip(depths[:-1],depths[1:])]+[depths[-1:]])
    depth_edges=np.r_[core,np.linspace(spec.height+5,spec.height+175,35)]
    y_edges=-depth_edges[::-1]
    x_edges=np.arange(-175,spec.width+175+1,5,dtype=np.float64)
    def q2(edges: FloatArray) -> FloatArray:
        return np.r_[np.column_stack((edges[:-1],(edges[:-1]+edges[1:])/2)).ravel(),edges[-1]]
    xq=np.asarray(q2(x_edges));yq=np.asarray(q2(y_edges))
    xx,yy=np.meshgrid(xq,yq)
    nx,ny=len(x_edges)-1,len(y_edges)-1
    rowdepth=-(y_edges[:-1]+y_edges[1:])/2
    regions=np.clip(np.searchsorted(depths,rowdepth,side='right')-1,0,len(props)-1)
    regions=np.repeat(regions,nx).astype(np.int64)
    frozen=FrozenMesh(np.column_stack((xx.ravel(),yy.ravel())),structured_cells(nx,ny),props[regions,0],props[regions,1],regions,0,'curved')
    expected_nodes=(5*nx+1)*(5*ny+1)
    report:dict[str,object]={'mesh_id':spec.mesh_id,'width_m':spec.width,'height_m':spec.height,'padding_m':175,'order':5,
        'nx':nx,'ny':ny,'sem_nodes':expected_nodes,'surface_nodes':5*nx+1,'elements':nx*ny,
        'depths_m':depths.tolist(),'materials_vp_vs':props.tolist(),'density_kg_m3':1000.,
        'dx_m':5.,'dy_min_m':float(np.diff(y_edges).min()),'dy_max_m':float(np.diff(y_edges).max()),
        'jacobian_min_m2':float(5*np.diff(y_edges).min()/4),'area_m2':float((spec.width+350)*(spec.height+175)),
        'mapping':'x=W*x_original/700; depth=H*depth_original/700; homogeneous/two_layer override; remeshed to h<=5m',
        'source':provenance,'output_bytes_estimate':expected_nodes*(21*2*8+2*8)+(5*nx+1)*1001*2*8}
    assert number(report['jacobian_min_m2'])>0
    return frozen,report


def damping(coords: FloatArray,width: int,height: int) -> FloatArray:
    x,y=coords.T
    sx=np.maximum.reduce((-x,x-width,np.zeros_like(x)))/175
    sy=np.maximum(-y-height,0)/175
    return ALPHA*(sx*sx+sy*sy)


def prepare() -> None:
    directory=ROOT/'meshes';directory.mkdir(parents=True,exist_ok=False)
    records:dict[str,object]={};budget=0
    for w,h in SIZES:
        for m in MATERIALS:
            spec=Spec(w,h,m,15);frozen,report=build(spec)
            target=directory/spec.mesh_id;target.mkdir()
            save_frozen(frozen,target/'mesh_q2.npz')
            write(target/'geometry.json',report)
            records[spec.mesh_id]=report
            budget+=2*int(number(report['output_bytes_estimate']))
    write(ROOT/'plan.json',{'runs':[asdict(s) for s in specs()],'geometry':records,'estimated_raw_bytes':budget,
        'alpha_max_s_inv':ALPHA,'damping':'C=alpha M; quadratic; sum side/bottom at corners; same alpha in all cases',
        'source':'nodal Fy=-Ricker; peak magnitude 1 N/m, t0=1.5/f; dense input table 1 microsecond',
        'duration_s':1.,'frame_interval_s':.05,'surface_interval_s':.001,'kim_beta':1/27,'kim_gamma':.5,'cfl_safety':.8,
        'production_queue':'not accessed','receiver_policy':'all GLL nodes on entire free top, including padding'})
    print('prepared',len(records),'meshes; raw bytes',budget,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();prepare()
