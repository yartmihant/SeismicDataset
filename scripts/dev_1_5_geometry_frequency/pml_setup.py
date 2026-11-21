"""Homogeneous PML test: common padded mesh and remote-boundary reference."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from numpy.typing import NDArray
from semgpu2.model import SGPETALineModel, SGPhysicsModel
from semgpu2.model.mesh import SGMeshModel
from scripts.dev_1_3_point_source.mesh import FrozenMesh,build_sem,load_frozen,save_frozen,structured_cells
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.setup import ROOT as SERIES, ALPHA,build_damping,damping,write

ROOT=SERIES.parent/'pml_homogeneous_saturn_v1'
W=1400;H=700;L=175;VP=4087.;VS=2431.62036;RHO=1000.
REFERENCE_PADDING=2500
FloatArray=NDArray[np.float64]
IntArray=NDArray[np.int64]
PADDED=SERIES/'meshes/1400x700_homogeneous/mesh_q2.npz'


def profile(coords: FloatArray) -> FloatArray:
    sx=np.maximum(np.maximum(-coords[:,0],coords[:,0]-W),0)/L
    sy=np.maximum(-coords[:,1]-H,0)/L
    return ALPHA*np.column_stack((sx*sx,sy*sy))


def pml_field(frozen: FrozenMesh,mesh: SGMeshModel,scale: float=1.) -> SGPhysicsModel:
    center=frozen.nodes[frozen.cells[:,4]]
    entities=np.flatnonzero((center[:,0]<0)|(center[:,0]>W)|(center[:,1]<-H))
    cells=mesh.strata[1].p_nodes.reshape(len(frozen.cells),-1)
    values=scale*profile(mesh.nodes_coords[cells[entities].ravel()])
    return SGPhysicsModel('cartesian_pml',31100,mesh,1,entities,values)


def observation(mesh: SGMeshModel) -> tuple[IntArray,IntArray]:
    x,y=mesh.nodes_coords.T
    field=np.flatnonzero((x>=-L-1e-8)&(x<=W+L+1e-8)&(y>=-H-L-1e-8))
    xy=np.round(mesh.nodes_coords[field],8)
    field=field[np.lexsort((xy[:,0],xy[:,1]))]
    surface=field[np.abs(mesh.nodes_coords[field,1])<1e-8]
    return field,surface


def core_mass(frozen: FrozenMesh,mesh: SGMeshModel,order: int,selected: IntArray) -> FloatArray:
    center=frozen.nodes[frozen.cells[:,4]]
    core=(center[:,0]>0)&(center[:,0]<W)&(center[:,1]>-H)
    cells=mesh.strata[1].p_nodes.reshape(len(frozen.cells),-1)
    weights=SGPETALineModel('GLL',order=order).build_data(use_symmetry=False).weights
    local=(RHO*6.25*np.outer(weights,weights)).ravel()
    mass=np.bincount(cells[core].ravel(),weights=np.tile(local,np.count_nonzero(core)),minlength=len(mesh.nodes_coords))
    assert np.isclose(mass.sum(),RHO*W*H,rtol=1e-12)
    return np.asarray(mass[selected],dtype=np.float64)


def reference_mesh() -> FrozenMesh:
    x=np.arange(-REFERENCE_PADDING,W+REFERENCE_PADDING+2.5,2.5)
    y=np.arange(-H-REFERENCE_PADDING,2.5,2.5)
    xx,yy=np.meshgrid(x,y);nx=(len(x)-1)//2;ny=(len(y)-1)//2
    cells=structured_cells(nx,ny);count=len(cells)
    return FrozenMesh(np.column_stack((xx.ravel(),yy.ravel())),cells,np.full(count,VP),np.full(count,VS),np.zeros(count,dtype=np.int64),0,'homogeneous_reference')


def check(out: Path) -> None:
    out.mkdir(parents=True,exist_ok=False)
    frozen=load_frozen(PADDED);assert np.all(frozen.vp==VP) and np.all(frozen.vs==VS)
    records=[]
    for order in (4,5):
        mesh=build_sem(frozen,order).mesh;field,surface=observation(mesh)
        pml=pml_field(frozen,mesh);dto=pml.build_data()
        assert dto.values.shape==(2,len(pml.entities)*(order+1)**2)
        assert np.all(dto.values>=0) and np.isclose(dto.values.max(),ALPHA)
        assert len(pml.entities)==61250-39200
        np.testing.assert_allclose(profile(mesh.nodes_coords).sum(axis=1),damping(mesh.nodes_coords,W,H),atol=1e-12,rtol=1e-14)
        # PML touches the full free surface in both side strips; no vertical stretch there.
        values=profile(mesh.nodes_coords[surface]);assert np.all(values[:,1]==0) and values[0,0]>0 and values[-1,0]>0
        assert np.all(profile(np.array([[0.,0.],[W,0.],[700.,-H]]))==0)
        mass=core_mass(frozen,mesh,order,field)
        old=build_damping(frozen,mesh,W,H)
        records.append({'order':order,'nodes':len(mesh.nodes_coords),'surface_nodes':len(surface),'pml_elements':len(pml.entities),'old_damping_nodes':len(old.entities),'core_mass_kg_per_m':float(mass.sum()),'observation_nodes':len(field)})
    # Nearest image-source distance to any point in the common observed rectangle.
    nearest=2*REFERENCE_PADDING+W/2-L
    assert nearest==2*(H+REFERENCE_PADDING)-(H+L)
    write(out/'validation.json',{'status':'passed','orders':records,'material':[VP,VS,RHO],'pml_dmax_s_inv':ALPHA,'mesh':file_record(PADDED),'reference_domain':[-REFERENCE_PADDING,W+REFERENCE_PADDING,-H-REFERENCE_PADDING,0],'minimum_geometric_boundary_return_s':nearest/VP,'duration_s':1.,'reference_note':'geometric continuum travel-time bound; common local SEM mesh, verify early direct arrivals numerically'})


def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--check',type=Path);p.add_argument('--reference',action='store_true');a=p.parse_args()
    if a.check:check(a.check)
    if a.reference:
        path=ROOT/'meshes/reference_q2.npz';path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():raise FileExistsError(path)
        frozen=reference_mesh();save_frozen(frozen,path)
        write(path.with_suffix('.json'),{'file':file_record(path),'elements':len(frozen.cells),'bounds':[-2500,3900,-3200,0],'h_m':5,'material':[VP,VS,RHO]})


if __name__=='__main__':main()
