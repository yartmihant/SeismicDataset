"""CPU mesh/acquisition checks, first-chunk flatness audit and disk-guard tests."""
from __future__ import annotations
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import hashlib
import numpy as np
from scripts.dev_1_1_preprocessing.prepare_data import read_field
from scripts.project_paths import SOURCE
from scripts.dev_1_4_server_preparation.production_plan import DiskGuard,BudgetUnavailable
from scripts.dev_1_5_geometry_frequency.p6_7s_compare import leading_grid
from scripts.dev_1_5_geometry_frequency.pml_fixed_orders import pml
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_6_p64s_production.queue import CONFIG
from scripts.dev_1_6_p64s_production.run import SOURCES,RECEIVERS


def check(out:Path)->None:
    out.mkdir(parents=True,exist_ok=False);g=leading_grid();layer=pml(g)
    assert len(g.mesh.nodes_coords)==795691 and len(g.cells)==22050 and len(layer.entities)==12250
    assert set(g.mesh.strata)=={1}
    np.testing.assert_allclose(g.mesh.nodes_coords[[270,390,510,630],0],SOURCES,atol=1e-9)
    ids=np.array([np.argmin(abs(g.mesh.nodes_coords[g.top,0]-x)) for x in RECEIVERS]);np.testing.assert_allclose(g.mesh.nodes_coords[ids,0],RECEIVERS,atol=1e-9)
    np.testing.assert_allclose(RECEIVERS,1400-RECEIVERS[::-1]);assert len(set(ids))==140
    with TemporaryDirectory(prefix='tmp_p64s_guard_',dir=out) as tmp:
        root=Path(tmp);guard=DiskGuard(root/'test.sqlite',root,CONFIG,lambda:10**12)
        a=guard.reserve_next(0);b=guard.reserve_next(1);assert a and b and (a.model_id,b.model_id)==(0,1)
        guard.check_before_write(a.job_id,9_100_000);guard.complete(b.job_id,9_100_000);guard.fail(a.job_id,'injected test')
        try:guard.reserve_next(1)
        except BudgetUnavailable:pass
        else:raise AssertionError('Failed job did not block other GPU')
    archive=SOURCE/'data_EFVB_out.zip';records=[]
    with ZipFile(archive) as stream:
        names={Path(name).name:name for name in stream.namelist() if not name.endswith('/')}
        for model in range(3000):
            arrays=[];hashes=[]
            for component in ('vp','vs'):
                name=f'{component}_{model:05d}.txt';raw=stream.read(names[name]);array=read_field(raw,name)
                assert np.all(array==array[:,:1]),(model,component,'not flat')
                arrays.append(array);hashes.append(hashlib.sha256(raw).hexdigest())
            vp,vs=arrays;assert np.all(vp*vp>4/3*vs*vs)
            records.append({'id':model,'vp_max':float(vp.max()),'vs_min':float(vs.min()),'raw_sha256':hashes})
    write(out/'check.json',{'status':'passed','nodes':795691,'elements':22050,'PML_elements':12250,'boundary_elements':0,'receiver_count':140,'sources':list(SOURCES),'guard_failure_blocks_both_gpus':True,'flat_models':3000,'models':records,'raw_velocity_bytes_chunk':3000*1001*4*140*2*8})
    print('CPU checks passed,3000 flat EFVB models',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('data/dev_1_6_p64s_production/cpu_v1'));a=p.parse_args();check(a.out)
