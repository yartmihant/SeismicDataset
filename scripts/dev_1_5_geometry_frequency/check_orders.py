"""CPU audit of fixed ALine tables, common mesh and corrected damping support."""
from __future__ import annotations
import argparse
from pathlib import Path
import re
import numpy as np
from semgpu2.model import SGPETALineModel
from scripts.dev_1_3_point_source.mesh import build_sem,load_frozen
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_5_geometry_frequency.setup import ROOT,build_damping,damping,write


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    meshpath=ROOT/'meshes/1400x700_two_layer/mesh_q2.npz';frozen=load_frozen(meshpath)
    records=[]
    for order in (4,5):
        mesh=build_sem(frozen,order).mesh;coords=mesh.nodes_coords;n=order+1
        local=coords[mesh.strata[1].p_nodes].reshape(-1,n,n,2)
        assert np.all(np.diff(local[:,:,:,0],axis=2)>0) and np.all(np.diff(local[:,:,:,1],axis=1)>0)
        assert len(coords)==(350*order+1)*(175*order+1)
        xy=coords.T;dto=build_damping(frozen,mesh,1400,700).build_data()
        expected=np.flatnonzero((xy[0]<=1e-8)|(xy[0]>=1400-1e-8)|(xy[1]<=-700+1e-8))
        np.testing.assert_array_equal(dto.entity_ids,expected)
        assert len(np.unique(dto.entity_ids))==len(dto.entity_ids)
        dense=np.zeros(len(coords));dense[dto.entity_ids]=dto.values[0]
        np.testing.assert_array_equal(dense,damping(coords,1400,700))
        table=Path(f'docs/runtime_sources/seismic_p45_damping_v1/src/semgpu2/solvers/lin_elast_2d_seismic_newmark_kim_p{order}_cu/backend/factor_tables.cuh')
        text=table.read_text().split('static __constant__ const FactorTables factor = {',1)[1].split('};',1)[0]
        values=np.array([float.fromhex(v) for v in re.findall(r'-?0x[0-9a-f.]+p[+-][0-9]+',text)])
        dto_line=SGPETALineModel('GLL',order=order).build_data(use_symmetry=False)
        standard=np.r_[dto_line.weights.ravel(),dto_line.values.ravel(),dto_line.grads[:,:,0].ravel()]
        np.testing.assert_array_equal(values.view(np.uint64),standard.view(np.uint64))
        records.append({'order':order,'nodes':len(coords),'elements':len(local),'surface_nodes':int(np.count_nonzero(abs(xy[1])<1e-9)),'damping_nodes':len(expected),'zero_alpha_interface_nodes':int(np.count_nonzero(dto.values[0]==0)),'positive_alpha_nodes':int(np.count_nonzero(dto.values[0]>0)),'dense_profile_bitwise_equal':True,'fixed_tables_bitwise_equal':True,'fixed_coefficient_count':len(values)})
    write(args.out/'validation.json',{'status':'passed','mesh':file_record(meshpath),'orders':records})
    print(records)


if __name__=='__main__':main()
