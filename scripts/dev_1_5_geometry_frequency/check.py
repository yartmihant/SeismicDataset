"""CPU checks of geometry, materials, damping and source/receiver support."""
from __future__ import annotations
import argparse
import numpy as np
from semgpu2.model.mesh import SGMeshModel
from scripts.dev_1_5_geometry_frequency.setup import ROOT,ALPHA,Spec,specs,build_damping,damping,write
from scripts.dev_1_3_point_source.mesh import FrozenMesh,load_frozen,build_sem,jacobians
from scripts.reporting import read_json,number,mapping


def results() -> None:
    records:dict[str,object]={}
    native_hashes:set[str]=set()
    for spec in specs():
        base=ROOT/'runs'/spec.run_id
        meta=read_json(base/'result.json')
        assert meta['status']=='completed' and meta['frames']==21 and meta['surface_samples']==1001
        assert meta['surface_field_exact'] is True
        assert meta['kim_beta']==1/27 and meta['cfl_safety']==.8 and meta['alpha_max_s_inv']==ALPHA
        native_hashes.add(str(mapping(meta['native'])['sha256']))
        nodes=np.load(base/'nodes.npy',mmap_mode='r')
        fields=np.load(base/'velocity.npy',mmap_mode='r')
        assert fields.shape==(21,len(nodes),2)
        xy=np.round(nodes,8);nx=len(np.unique(xy[:,0]));ny=len(np.unique(xy[:,1]))
        assert nx*ny==len(nodes)
        order=np.lexsort((xy[:,0],xy[:,1]))
        grid=xy[order].reshape(ny,nx,2)
        np.testing.assert_allclose(grid[:,:,0]+grid[:,::-1,0],spec.width,atol=2e-8,rtol=0)
        difference=0.;reference=0.
        for frame in fields[1:]:
            v=frame[order].reshape(ny,nx,2)
            residual=v-v[:,::-1,:]*np.array([-1.,1.])
            difference+=float(np.sum(residual**2));reference+=float(np.sum(v**2))
        symmetry=float(np.sqrt(difference/reference))
        assert symmetry<1e-6, (spec.run_id,symmetry)
        with np.load(base/'surface.npz') as z:
            assert z['velocity_m_s'].shape==(1001,nx,2)
            np.testing.assert_allclose(z['time_s'],np.arange(1001)*.001,atol=1e-12,rtol=0)
            assert np.all(np.diff(z['coordinates_m'][:,0])>0)
        if spec.frequency==30:
            partner=read_json(ROOT/'runs'/Spec(spec.width,spec.height,spec.material,15).run_id/'result.json')
            assert partner['dt_s']==meta['dt_s'] and partner['mesh']==meta['mesh']
        records[spec.run_id]={'velocity_reflection_relative_l2':symmetry,'frames':21,'surface_nodes':nx,
            'dt_s':meta['dt_s'],'total_seconds':meta['total_seconds']}
    assert len(native_hashes)==1
    write(ROOT/'results_validation.json',{'status':'passed','runs':records,'count':30,'native_sha256':next(iter(native_hashes)),
        'checks':'full frame/surface shapes, times, common mesh and dt within frequency pairs, Vx odd and Vy even about central source, common native and settings'})
    print('Results validation passed',flush=True)


def check_damping_domain(frozen: FrozenMesh, mesh: SGMeshModel, width: int, height: int) -> int:
    dto = build_damping(frozen, mesh, width, height).build_data()
    coords = mesh.nodes_coords
    x, y = coords.T
    # Include the shared interface despite roundoff in the mapped GLL coordinates.
    tol = 1e-8
    expected = (x <= tol) | (x >= width - tol) | (y <= -height + tol)
    np.testing.assert_array_equal(dto.entity_ids, np.flatnonzero(expected))
    dense = np.zeros(len(coords), dtype=np.float64)
    dense[dto.entity_ids] = dto.values[0]
    np.testing.assert_array_equal(dense, damping(coords, width, height))
    return len(dto.entity_ids)


def main() -> None:
    plan=read_json(ROOT/'plan.json');assert len(specs())==30
    checks=[]
    for spec in specs()[::2]:
        base=ROOT/'meshes'/spec.mesh_id
        mesh=load_frozen(base/'mesh_q2.npz');meta=read_json(base/'geometry.json')
        assert np.all(jacobians(mesh,np.array([-1.,0.,1.]))>0)
        np.testing.assert_allclose(mesh.nodes.min(axis=0),[-175,-spec.height-175])
        np.testing.assert_allclose(mesh.nodes.max(axis=0),[spec.width+175,0])
        assert len(mesh.cells)==meta['elements']
        xy=np.array([[0.,0.],[spec.width,-spec.height],[-175,0],[spec.width+175,-spec.height-175],[0.,-spec.height-87.5]])
        np.testing.assert_allclose(damping(xy,spec.width,spec.height),[0.,0.,ALPHA,2*ALPHA,ALPHA/4])
        if spec.material=='homogeneous':assert np.all(mesh.vp==4087) and np.all(mesh.vs==2431.62036)
        if spec.material=='two_layer':
            cy=mesh.nodes[mesh.cells[:,4],1];upper=cy>-spec.height/2
            assert np.all(mesh.vp[upper]==4087) and np.all(mesh.vp[~upper]==3963)
        checks.append(spec.mesh_id)
    sem_checks=[]
    for mesh_id,width,height in (('350x350_efvb',350,350),('1400x700_homogeneous',1400,700)):
        base=ROOT/'meshes'/mesh_id;frozen=load_frozen(base/'mesh_q2.npz');case=build_sem(frozen,5)
        coords=case.mesh.nodes_coords;meta=read_json(base/'geometry.json')
        assert len(coords)==meta['sem_nodes']
        top=np.flatnonzero(np.abs(coords[:,1])<1e-9)
        assert len(top)==meta['surface_nodes']
        assert np.count_nonzero(np.all(np.abs(coords-[width/2,0])<1e-9,axis=1))==1
        damping_count=check_damping_domain(frozen,case.mesh,width,height)
        sem_checks.append({'mesh':mesh_id,'nodes':len(coords),'top_nodes':len(top),'damping_nodes':damping_count})
    assert number(plan['estimated_raw_bytes'])<20*2**30
    write(ROOT/'cpu_validation.json',{'status':'passed','meshes':checks,'sem_checks':sem_checks,'estimated_raw_bytes':plan['estimated_raw_bytes'],
        'solver_run':False,'checks':'positive rectangular Q2 Jacobians, dimensions, interface materials, corner damping, full top and exact central source'})
    print('CPU validation passed',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--results',action='store_true');args=parser.parse_args()
    if args.results:results()
    else:main()
