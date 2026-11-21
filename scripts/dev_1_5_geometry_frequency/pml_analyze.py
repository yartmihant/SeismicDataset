"""Reference-subtracted reflection and physical core kinetic energy, saved data only."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from numpy.typing import NDArray
from scripts.dev_1_5_geometry_frequency.pml_setup import ROOT,W,H,VP
from scripts.dev_1_5_geometry_frequency.setup import write
from scripts.dev_1_3_point_source.run import file_record
from scripts.reporting import read_json,mapping,number

FloatArray=NDArray[np.float64]
MODES=('old','pml','pml_abc','reference')


def surface_weights(order: int) -> FloatArray:
    from semgpu2.model import SGPETALineModel
    w=SGPETALineModel('GLL',order=order).build_data(use_symmetry=False).weights
    ids=np.arange(35,315)[:,None]*order+np.arange(order+1)[None,:]
    result=np.bincount(ids.ravel(),weights=np.tile(2.5*w,280),minlength=350*order+1)
    assert np.isclose(result.sum(),W)
    return np.asarray(result,dtype=np.float64)


def traces(base: Path) -> tuple[FloatArray,FloatArray,FloatArray]:
    with np.load(base/'surface.npz') as z:
        return z['time_s'],z['coordinates_m'],z['velocity_m_s']


def weighted_surface(error: FloatArray,reference: FloatArray,weights: FloatArray,mask: NDArray[np.bool_]) -> dict[str,object]:
    e=error[mask];ref=reference
    squared=np.sum(e*e*weights[None,:,None],axis=(0,1))
    direct_squared=np.sum(ref*ref*weights[None,:,None],axis=(0,1))
    active=weights>0
    peak=np.max(np.abs(e[:,active]),axis=(0,1));direct_peak=np.max(np.abs(ref[:,active]),axis=(0,1))
    return {'error_weighted_l2':float(np.sqrt(squared.sum())),'relative_l2_to_full_reference':float(np.sqrt(squared.sum()/direct_squared.sum())),'relative_l2_components_to_full_reference':np.sqrt(squared/direct_squared).tolist(),'max_abs_error_m_s':peak.tolist(),'max_error_over_global_reference_peak':(peak/direct_peak).tolist()}


def analyze() -> None:
    out=ROOT/'analysis';out.mkdir(exist_ok=False)
    records:dict[str,object]={};verified=[]
    for order in (4,5):
        base=ROOT/f'p{order}';refbase=base/'reference';refmeta=read_json(refbase/'result.json')
        nodes=np.load(refbase/'nodes.npy');mass=np.load(refbase/'core_mass.npy');refv=np.load(refbase/'velocity.npy',mmap_mode='r')
        times,rx,refs=traces(refbase);weights=surface_weights(order);far=weights*(np.abs(rx[:,0]-700)>=20)
        late=(times>=.5)&(times<=1.);early=times<=.1;fieldtimes=np.arange(21,dtype=np.float64)*.05
        reference_energy=np.array([.5*np.sum(frame*frame*mass[:,None]) for frame in refv]);reference_peak=np.max(np.abs(refs),axis=(0,1))
        order_records:dict[str,object]={};series:dict[str,FloatArray]={'time_s':fieldtimes,'reference_kinetic_j_per_m':reference_energy,'surface_time_s':times}
        for mode in MODES:
            path=base/mode;meta=read_json(path/'result.json');assert meta['status']=='completed' and meta['dt_s']==refmeta['dt_s']
            np.testing.assert_allclose(np.load(path/'nodes.npy'),nodes,atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.load(path/'core_mass.npy'),mass,atol=1e-10,rtol=1e-13)
            t,x,v=traces(path);np.testing.assert_array_equal(t,times);np.testing.assert_allclose(x,rx,atol=1e-8,rtol=0)
            error=v-refs
            metrics=weighted_surface(error,refs,weights,late);far_metrics=weighted_surface(error,refs,far,late)
            early_metrics=weighted_surface(error,refs,weights,early)
            fields=np.load(path/'velocity.npy',mmap_mode='r');energy=[];errenergy=[];field_error_peak=[]
            core=mass>0
            for frame,ref in zip(fields,refv):
                assert np.isfinite(frame).all()
                delta=frame-ref
                energy.append(float(.5*np.sum(frame*frame*mass[:,None])))
                errenergy.append(float(.5*np.sum(delta*delta*mass[:,None])))
                field_error_peak.append(float(np.max(np.abs(delta[core]))))
            energy_a=np.array(energy);error_a=np.array(errenergy)
            series[mode+'_kinetic_j_per_m']=energy_a;series[mode+'_error_kinetic_norm_j_per_m']=error_a
            series[mode+'_surface_error_l2_m_s_sqrt_m']=np.sqrt(np.sum(error*error*weights[None,:,None],axis=(1,2)))
            station_ids=np.array([int(np.argmin(abs(rx[:,0]-q))) for q in (0,350,700,1050,1400)])
            series[mode+'_stations_vx_vy']=v[:,station_ids,:]
            order_records[mode]={'late_surface':metrics,'late_surface_outside_20m':far_metrics,'early_surface':early_metrics,'kinetic_final_j_per_m':float(energy_a[-1]),'kinetic_final_over_reference_peak':float(energy_a[-1]/reference_energy.max()),'error_kinetic_final_j_per_m':float(error_a[-1]),'error_kinetic_mean_late_j_per_m':float(error_a[fieldtimes>=.5].mean()),'core_field_max_error_by_frame_m_s':field_error_peak,'run_seconds':meta['run_seconds'],'prepare_seconds':meta['prepare_seconds'],'steps':meta['steps'],'dt_s':meta['dt_s'],'pml_elements':meta['pml_elements'],'ABC':meta['ABC'],'global_nodes':meta['global_nodes']}
        old_l2=number(mapping(mapping(order_records['old'])['late_surface'])['error_weighted_l2'])
        for mode in ('pml','pml_abc'):
            row=mapping(order_records[mode]);value=number(mapping(row['late_surface'])['error_weighted_l2'])
            row['late_surface_error_ratio_to_old']=value/old_l2
            row['late_surface_error_reduction_factor_vs_old']=old_l2/value
        temporal:dict[str,object]={}
        if order==5:
            for coarse,fine in (('old','old_half'),('pml','pml_half')):
                coarse_meta=read_json(base/coarse/'result.json');fine_meta=read_json(base/fine/'result.json')
                assert np.isclose(number(fine_meta['dt_s']),number(coarse_meta['dt_s'])/2,rtol=1e-13,atol=0)
                _,x,a=traces(base/fine);_,y,b=traces(base/coarse);np.testing.assert_array_equal(x,y)
                temporal[coarse]={'late_coarse_fine':weighted_surface(a-b,refs,weights,late),'full_relative_l2_components':np.sqrt(np.sum((a-b)**2,axis=(0,1))/np.sum(a*a,axis=(0,1))).tolist(),'dt_coarse_s':coarse_meta['dt_s'],'dt_fine_s':fine_meta['dt_s']}
                series[coarse+'_half_surface_difference_l2']=np.sqrt(np.sum((a-b)**2*weights[None,:,None],axis=(1,2)))
        # Early comparison is an arrival check, not a declaration of analytic exactness.
        assert all(number(mapping(mapping(order_records[m])['early_surface'])['relative_l2_to_full_reference'])<1e-5 for m in ('old','pml','pml_abc'))
        for result in sorted(base.glob('*/result.json')):
            meta=read_json(result)
            for key,name in [('fields','velocity.npy'),('nodes','nodes.npy'),('mass','core_mass.npy'),('surface','surface.npz')]:
                assert file_record(result.parent/name)['sha256']==mapping(meta[key])['sha256']
            verified.append(str(result.relative_to(ROOT)))
        np.savez(out/f'p{order}_curves.npz',allow_pickle=False,**series)
        records[f'p{order}']={'modes':order_records,'time_step_check':temporal,'reference_surface_peak_m_s':reference_peak.tolist(),'station_x_m':[0,350,700,1050,1400],'reference_min_geometric_boundary_return_s':5525/VP}
    write(out/'metrics.json',{'status':'completed','orders':records,'verified_runs':verified,'primary_window_s':[.5,1.],'surface_metric':'sqrt(sum_t,i,c(dx_GLL_i * (v-v_reference)^2)); common dt=0.001, reported L2 omits common time factor; x in [0,1400]','kinetic_metric':'0.5 sum_i m_core_i |v_i|^2, physical core only, J per metre out of plane; not total elastic energy','reference_note':'expanded homogeneous domain, same local h/order/dt; finite-box discretized reference, not an analytical exact solution; field comparison uses 21 saved snapshots','half_step_note':'p5 coarse/fine compared to each other; reference computed at common coarse dt only'})
    print('ANALYSIS COMPLETED',flush=True)


if __name__=='__main__':
    argparse.ArgumentParser().parse_args();analyze()
