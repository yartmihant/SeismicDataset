"""CPU checks of geometry, materials, damping and source/receiver support."""
from __future__ import annotations
import numpy as np
from scripts.dev_1_5_geometry_frequency.setup import ROOT,ALPHA,Spec,specs,damping,write
from scripts.dev_1_3_point_source.mesh import load_frozen,build_sem,jacobians
from scripts.reporting import read_json,number


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
    for mesh_id,width in (('350x350_efvb',350),('1400x700_homogeneous',1400)):
        base=ROOT/'meshes'/mesh_id;case=build_sem(load_frozen(base/'mesh_q2.npz'),5)
        coords=case.mesh.nodes_coords;meta=read_json(base/'geometry.json')
        assert len(coords)==meta['sem_nodes']
        top=np.flatnonzero(np.abs(coords[:,1])<1e-9)
        assert len(top)==meta['surface_nodes']
        assert np.count_nonzero(np.all(np.abs(coords-[width/2,0])<1e-9,axis=1))==1
        sem_checks.append({'mesh':mesh_id,'nodes':len(coords),'top_nodes':len(top)})
    assert number(plan['estimated_raw_bytes'])<20*2**30
    write(ROOT/'cpu_validation.json',{'status':'passed','meshes':checks,'sem_checks':sem_checks,'estimated_raw_bytes':plan['estimated_raw_bytes'],
        'solver_run':False,'checks':'positive rectangular Q2 Jacobians, dimensions, interface materials, corner damping, full top and exact central source'})
    print('CPU validation passed',flush=True)


if __name__=='__main__':main()
