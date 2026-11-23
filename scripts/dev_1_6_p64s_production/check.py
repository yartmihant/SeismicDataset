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
from scripts.dev_1_6_p64s_production.queue import CONFIG,CONT_CONFIG,FlatContinuationGuard
from scripts.dev_1_6_p64s_production.run import SOURCES,RECEIVERS


def check_continuation_guard(out:Path)->None:
    with TemporaryDirectory(prefix='tmp_cont_guard_',dir=out) as tmp:
        root=Path(tmp);g=FlatContinuationGuard(root/'guard.sqlite',root,CONT_CONFIG,lambda:10**12)
        a=g.reserve_next(0);b=g.reserve_next(1);assert a and b and (a.model_id,b.model_id)==(3000,3001)
        try:g.reserve_next(0)
        except BudgetUnavailable:pass
        else:raise AssertionError('Duplicate GPU reservation')
        g.complete(a.job_id,1);g.complete(b.job_id,1)
        with g._transaction() as c:
            c.executemany("INSERT INTO jobs VALUES (?,2,'EFVB',?,0,?,'completed',1,NULL)",[(f'EFVB/{i:05d}',i,CONT_CONFIG.model_budget_bytes) for i in range(3002,5999)])
        last=g.reserve_next(0);assert last and last.model_id==5999 and last.chunk_number==2
        assert g.reserve_next(1) is None,'Chunk barrier crossed while previous model active'
        g.complete(last.job_id,1);next_job=g.reserve_next(1);assert next_job and next_job.model_id==6000 and next_job.chunk_number==3
        g.fail(next_job.job_id,'test injected')
        try:g.reserve_next(0)
        except BudgetUnavailable:pass
        else:raise AssertionError('Failure did not stop both GPUs')
        final=FlatContinuationGuard(root/'final.sqlite',root,CONT_CONFIG,lambda:10**12)
        with final._transaction() as c:
            c.executemany("INSERT INTO jobs VALUES (?,?,'EFVB',?,0,?,'completed',1,NULL)",[(f'EFVB/{i:05d}',i//3000+1,i,CONT_CONFIG.model_budget_bytes) for i in range(3000,12000)])
        assert final.reserve_next(0) is None and final.reserve_next(1) is None,'Fourth continuation chunk was enabled'
        poor=FlatContinuationGuard(root/'poor.sqlite',root,CONT_CONFIG,lambda:CONT_CONFIG.reserve_bytes+CONT_CONFIG.working_floor_bytes+CONT_CONFIG.model_budget_bytes-1)
        try:poor.reserve_next(0)
        except BudgetUnavailable:pass
        else:raise AssertionError('Disk reserve violated')
    write(out/'continuation_guard.json',{'status':'passed','first_id':3000,'last_id':11999,'checks':['unique GPU reservations','strict3000-model chunk barrier','failure blocks both GPUs','stop after11999','40GiB combined reserve maintained']})


def check(out:Path,first:int=0,last:int=2999)->None:
    assert 0<=first<=last<30000
    out.mkdir(parents=True,exist_ok=False);check_continuation_guard(out);g=leading_grid();layer=pml(g)
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
        for model in range(first,last+1):
            arrays=[];hashes=[]
            for component in ('vp','vs'):
                name=f'{component}_{model:05d}.txt';raw=stream.read(names[name]);array=read_field(raw,name)
                assert np.all(array==array[:,:1]),(model,component,'not flat')
                arrays.append(array);hashes.append(hashlib.sha256(raw).hexdigest())
            vp,vs=arrays;assert np.all(vp*vp>4/3*vs*vs)
            records.append({'id':model,'vp_max':float(vp.max()),'vs_min':float(vs.min()),'raw_sha256':hashes})
    write(out/'check.json',{'status':'passed','nodes':795691,'elements':22050,'PML_elements':12250,'boundary_elements':0,'receiver_count':140,'sources':list(SOURCES),'guard_failure_blocks_both_gpus':True,'first_id':first,'last_id':last,'flat_models':last-first+1,'models':records,'raw_velocity_bytes_chunk':(last-first+1)*1001*4*140*2*8})
    print('CPU checks passed, flat EFVB models',first,last,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('data/dev_1_6_p64s_production/cpu_v1'));p.add_argument('--first',type=int,default=0);p.add_argument('--last',type=int,default=2999);a=p.parse_args();check(a.out,a.first,a.last)
