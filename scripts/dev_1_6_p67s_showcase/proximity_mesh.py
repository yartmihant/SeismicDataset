"""Плоская EFVB00001: перенос источника к левому PML при неизменных p6/h10."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from semgpu2.model import SGPETALineModel, SGPhysicsModel
from semgpu2.model.mesh import SGMeshModel, SGMeshStratumData
from scripts.dev_1_4_server_preparation.model_input import read_model
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import FloatArray, IntArray
from scripts.dev_1_5_geometry_frequency.setup import ALPHA, write
ROOT=Path('data/dev_1_6_p67s_showcase/proximity_v1')
DISTANCES=(0.,5.,10.,20.,35.,50.,75.,100.,150.,175.,250.,350.,700.)

@dataclass(frozen=True)
class Case:
    name:str
    domain:str
    source:float
    homogeneous:bool=False
    half:bool=False

@dataclass
class Grid:
    mesh:SGMeshModel
    cells:IntArray
    centers:FloatArray
    top:IntArray
    top_weights:FloatArray
    source_node:int
    vp:FloatArray
    vs:FloatArray
    pml_ids:IntArray
    pml_values:FloatArray

BOUNDS={'production':(-350.,1750.,-1050.),'reference':(-2500.,4500.,-2800.),'larger':(-3000.,5000.,-3300.),'left_only':(-350.,4500.,-2800.)}


def cases()->list[Case]:
    result=[Case(f'P{d:g}','production',d) for d in DISTANCES]
    for phase in (0,5):
        result.extend((Case(f'R{phase}','reference',700.+phase),Case(f'H{phase}','reference',700.+phase,True),Case(f'B{phase}','larger',700.+phase),Case(f'R{phase}_half','reference',700.+phase,False,True)))
    result.extend(Case(f'L{d}','left_only',float(d)) for d in (0,35,175,700))
    result.append(Case('direct5','reference',5.))
    result.extend(Case(f'P{d}_half','production',float(d),False,True) for d in (0,5,175))
    return result


def grid(case:Case)->Grid:
    xmin,xmax,ymin=BOUNDS[case.domain];h=10;p=6;nx=round((xmax-xmin)/h);ny=round(-ymin/h)
    line=SGPETALineModel('GLL',order=p);ref=np.asarray(line.p_nodes.coords[:,0],dtype=np.float64)
    def axis(n:int,begin:float)->FloatArray:return np.r_[begin+h*(np.arange(n)[:,None]+(ref[:-1]+1)/2).ravel(),begin+n*h]
    x,y=axis(nx,xmin),axis(ny,ymin);xx,yy=np.meshgrid(x,y);coords=np.column_stack((xx.ravel(),yy.ravel()));del xx,yy
    ex,ey=np.meshgrid(np.arange(nx),np.arange(ny));local=np.arange(p+1)
    cells=((ey.ravel()[:,None,None]*p+local[None,:,None])*len(x)+ex.ravel()[:,None,None]*p+local[None,None,:]).reshape(-1,49).astype(np.int64)
    centers=np.column_stack((xmin+(ex.ravel()+.5)*h,ymin+(ey.ravel()+.5)*h));del ex,ey
    n=len(coords);e=len(cells);ix=np.tile(np.arange(len(x)),len(y));iy=np.repeat(np.arange(len(y)),len(x))
    tags=np.zeros((n,3),dtype=np.int64);tags[:,0]=(ix%p!=0).astype(np.int64)+(iy%p!=0)
    tags[:,1]=2-((ix==0)|(ix==len(x)-1)).astype(np.int64)-((iy==0)|(iy==len(y)-1)).astype(np.int64)
    mesh=SGMeshModel(coords,tags,[line],{1:SGMeshStratumData((2,2,2),cells.ravel(),np.arange(e+1,dtype=np.int64)*49,np.zeros(2*e,dtype=np.int32),np.arange(e+1,dtype=np.int64)*2,np.ones(e,dtype=np.int64))})
    source=read_model('EFVB',1);assert np.all(source.vp==source.vp[:,:1]) and np.all(source.vs==source.vs[:,:1])
    row=np.clip((-centers[:,1]/10).astype(np.int64),0,69)
    vp=np.full(e,source.vp[0,0]) if case.homogeneous else source.vp[row,0]
    vs=np.full(e,source.vs[0,0]) if case.homogeneous else source.vs[row,0]
    select=(x>=-1e-9)&(x<=((1400 if case.domain in ('production','left_only') else 2100)+1e-9))
    top=(np.arange(len(x),dtype=np.int64)+n-len(x))[select]
    weights=np.asarray(line.build_data(use_symmetry=False).weights,dtype=np.float64)
    full_weights=np.bincount((np.arange(nx)[:,None]*p+np.arange(7)).ravel(),weights=np.tile(5*weights,nx),minlength=len(x))
    found=np.flatnonzero(np.abs(x-case.source)<1e-8);assert len(found)==1,(case.source,x[np.argmin(abs(x-case.source))])
    if case.domain=='production':ids=np.flatnonzero((centers[:,0]<0)|(centers[:,0]>1400)|(centers[:,1]<-700))
    elif case.domain=='left_only':ids=np.flatnonzero(centers[:,0]<0)
    else:ids=np.empty(0,dtype=np.int64)
    xy=coords[cells[ids].ravel()]
    sx=np.maximum(-xy[:,0],0)/350 if case.domain=='left_only' else np.maximum(np.maximum(-xy[:,0],xy[:,0]-1400),0)/350
    sy=np.zeros(len(xy)) if case.domain=='left_only' else np.maximum(-xy[:,1]-700,0)/350
    return Grid(mesh,cells,centers,top,full_weights[select].astype(np.float64),int(n-len(x)+found[0]),vp,vs,ids,ALPHA*np.column_stack((sx*sx,sy*sy)))


def prepare(root:Path)->None:
    root.mkdir(parents=True,exist_ok=False);source=read_model('EFVB',1)
    values=np.column_stack((source.vp[:,0],source.vs[:,0]));change=np.r_[0,np.flatnonzero(np.any(np.diff(values,axis=0)!=0,axis=1))+1,70]
    g=grid(Case('cpu','production',5.));xy=g.mesh.nodes_coords
    assert len(xy)==795691 and len(g.cells)==22050 and len(g.top)==841 and set(g.mesh.strata)=={1}
    for d in DISTANCES:assert np.any(np.abs(xy[g.top,0]-d)<1e-9)
    assert np.all(g.pml_values>=0) and np.isclose(g.pml_values.max(),ALPHA)
    # One-sided weighted receiver norm excludes x0/1400 and every PML receiver.
    np.testing.assert_allclose(xy[g.source_node],[5,0],atol=1e-10)
    metadata={name:{'bounds':list(bounds),'elements':round((bounds[1]-bounds[0])/10)*round(-bounds[2]/10),'nodes':(round((bounds[1]-bounds[0])/10)*6+1)*(round(-bounds[2]/10)*6+1)} for name,bounds in BOUNDS.items()}
    write(root/'plan.json',{'status':'cpu_checked','distances_m':list(DISTANCES),'cases':[c.__dict__ for c in cases()],'domains':metadata,
        'model':'EFVB00001, horizontal layers; stretched x2, velocity values unchanged','source':source.provenance,
        'layer_depths_m':(change*10).tolist(),'materials_vp_vs':values[change[:-1]].tolist(),'rho':1000,
        'p':6,'h_m':10,'PML_m':350,'PML_dmax_s_inv':ALPHA,'ABC':False,'beta':1/27,'gamma':.5,'duration_s':1,'tact_dt_s':.001,
        'source_wavelet':'Ricker30Hz t0=.05s, point Fy=-1N/m; cubic table at1us as prior p6 tests',
        'dt':'common min(native CFL0.8 probes production/reference/larger/left_only); explicit same dt for comparisons, plus dt/2',
        'reference':'large unabsorbed half-space approximation; two source phases700/705; shift by integer multiples10m only',
        'reference_return_lower_bound_s':min(5700,6195,5600)/float(source.vp.max()),
        'reference_check':'two larger domains plus direct reference source5; all same dt',
        'normalization':'first-horizon signal = layered reference - homogeneous reference; first PP window ±.025s, offsets20..200m to right, no late multiples',
        'criteria':'report peak/error-to-first-horizon and weighted energy proxy; tiers0.1/1/5percent peak and1e-6/1e-4/1e-2 energy; all interior receivers and margins10/25/50/100/175m',
        'storage':'surface-only files; production fields observed at21 times for diagnostics but not stored; <1GiB outputs',
        'production':'unchanged, stopped'})
