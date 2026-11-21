"""Reference-subtracted reflection and physical core kinetic energy, saved data only."""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
from numpy.typing import NDArray
from scripts.dev_1_5_geometry_frequency.pml_setup import ROOT,THICK_ROOT,THICKNESSES,W,H,VP
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


def core_indices(nodes: FloatArray) -> NDArray[np.int64]:
    return np.flatnonzero((nodes[:,0]>=-1e-8)&(nodes[:,0]<=W+1e-8)&(nodes[:,1]>=-H-1e-8))


def core_traces(path: Path) -> tuple[FloatArray,FloatArray,FloatArray]:
    t,x,v=traces(path);keep=(x[:,0]>=-1e-8)&(x[:,0]<=W+1e-8)
    return t,x[keep],v[:,keep]


def thickness_paths(order: int) -> list[tuple[str,Path]]:
    base=ROOT/f'p{order}'
    return [('old',base/'old'),('L175',base/'pml_abc')]+[(f'L{L}',THICK_ROOT/f'p{order}'/f'L{L}') for L in THICKNESSES]+([(f'F{L}',THICK_ROOT/f'p{order}'/f'L{L}_fixed') for L in THICKNESSES] if order==5 else [])+[('reference',base/'reference')]


def analyze_thickness() -> None:
    out=THICK_ROOT/'analysis';out.mkdir(exist_ok=False);orders:dict[str,object]={};verified=[]
    for order in (4,5):
        refpath=ROOT/f'p{order}'/'reference';refmeta=read_json(refpath/'result.json')
        t,x,ref=core_traces(refpath);weights=surface_weights(order)[35*order:315*order+1]
        assert len(weights)==len(x) and np.isclose(weights.sum(),W)
        refnodes=np.load(refpath/'nodes.npy');ri=core_indices(refnodes);refnodes=refnodes[ri]
        mass=np.load(refpath/'core_mass.npy')[ri];ref_fields=np.load(refpath/'velocity.npy',mmap_mode='r')
        reference_peak=max(float(abs(frame).max()) for frame in ref_fields)
        windows={'full':t>=0,'wide':t>=.2,'late':t>=.5,'early':t<=.1}
        curves:dict[str,FloatArray]={'surface_time_s':t,'field_time_s':np.arange(21,dtype=np.float64)*.05}
        rows:dict[str,object]={}
        for label,path in thickness_paths(order):
            meta=read_json(path/'result.json');assert meta['status']=='completed' and meta['dt_s']==refmeta['dt_s']
            assert mapping(meta['native'])['sha256']==mapping(refmeta['native'])['sha256']
            tt,xx,v=core_traces(path);np.testing.assert_array_equal(tt,t);np.testing.assert_allclose(xx,x,atol=1e-8,rtol=0)
            delta=v-ref;metrics={name:weighted_surface(delta,ref,weights,mask) for name,mask in windows.items()}
            assert number(metrics['early']['relative_l2_to_full_reference'])<1e-5
            nodes=np.load(path/'nodes.npy');idx=core_indices(nodes);np.testing.assert_allclose(nodes[idx],refnodes,atol=1e-8,rtol=0)
            np.testing.assert_allclose(np.load(path/'core_mass.npy')[idx],mass,rtol=1e-12,atol=1e-10)
            fields=np.load(path/'velocity.npy',mmap_mode='r');energy=[];error_energy=[];peaks=[];full_peaks=[]
            for i,frame in enumerate(fields):
                assert np.isfinite(frame).all()
                full_peaks.append(float(abs(frame).max()))
                core=frame[idx];difference=core-ref_fields[i,ri]
                energy.append(float(.5*np.sum(core**2*mass[:,None])))
                error_energy.append(float(.5*np.sum(difference**2*mass[:,None])))
                peaks.append(float(abs(difference).max()))
            curves[label+'_surface_error']=np.sqrt(np.sum(delta**2*weights[None,:,None],axis=(1,2)))
            curves[label+'_full_peak']=np.asarray(full_peaks,dtype=np.float64)
            curves[label+'_kinetic']=np.asarray(energy,dtype=np.float64)
            curves[label+'_error_kinetic']=np.asarray(error_energy,dtype=np.float64)
            rows[label]={'surface':metrics,'thickness_m':175 if label in ('old','L175') else (0 if label=='reference' else int(label[1:])),'rows':35 if label in ('old','L175') else (0 if label=='reference' else int(label[1:])//5),'dmax_s_inv':meta['dmax_or_alpha_s_inv'],'run_seconds':meta['run_seconds'],'prepare_seconds':meta['prepare_seconds'],'steps':meta['steps'],'dt_s':meta['dt_s'],'elements':meta['elements'],'global_nodes':meta['global_nodes'],'pml_elements':meta['pml_elements'],'surface_nodes':meta['surface_nodes'],'kinetic_final_j_per_m':energy[-1],'error_kinetic_final_j_per_m':error_energy[-1],'core_error_peak_by_frame_m_s':peaks,'source_result':file_record(path/'result.json'),'late_full_peak_m_s':max(full_peaks[10:]),'late_full_peak_over_reference_peak':max(full_peaks[10:])/reference_peak,'strong_growth':max(full_peaks[10:])>10*reference_peak,'fixed_strength':label.startswith('F')}
            for key,name in [('fields','velocity.npy'),('nodes','nodes.npy'),('mass','core_mass.npy'),('surface','surface.npz')]:
                assert file_record(path/name)['sha256']==mapping(meta[key])['sha256']
            verified.append(str(path))
        for label,row_object in rows.items():
            row=mapping(row_object)
            if label=='reference':continue
            for window in ('full','wide','late'):
                val=number(mapping(mapping(row['surface'])[window])['error_weighted_l2'])
                baseline=number(mapping(mapping(mapping(rows['L175'])['surface'])[window])['error_weighted_l2'])
                old=number(mapping(mapping(mapping(rows['old'])['surface'])[window])['error_weighted_l2'])
                row[window+'_error_ratio_to_L175']=val/baseline
                row[window+'_error_reduction_vs_old']=old/val
        time_check:dict[str,object]={}
        if order==5:
            fine=THICK_ROOT/'p5/L10_half';meta=read_json(fine/'result.json');coarse=read_json(THICK_ROOT/'p5/L10/result.json')
            assert np.isclose(number(meta['dt_s']),number(coarse['dt_s'])/2,rtol=1e-13,atol=0)
            tt,xx,a=core_traces(fine);_,_,b=core_traces(THICK_ROOT/'p5/L10');np.testing.assert_array_equal(tt,t);np.testing.assert_allclose(xx,x,atol=1e-8,rtol=0)
            time_check={name:weighted_surface(a-b,ref,weights,mask) for name,mask in windows.items()}
            time_check['full_difference_over_L10_error']=number(mapping(time_check['full'])['error_weighted_l2'])/number(mapping(mapping(mapping(rows['L10'])['surface'])['full'])['error_weighted_l2'])
            for key,name in [('fields','velocity.npy'),('nodes','nodes.npy'),('mass','core_mass.npy'),('surface','surface.npz')]:
                assert file_record(fine/name)['sha256']==mapping(meta[key])['sha256']
            fine_fields=np.load(fine/'velocity.npy',mmap_mode='r')
            time_check['late_full_peak_fine_m_s']=max(float(abs(frame).max()) for frame in fine_fields[10:])
            time_check['late_full_peak_coarse_m_s']=mapping(rows['L10'])['late_full_peak_m_s']
            time_check['strong_growth_fine']=number(time_check['late_full_peak_fine_m_s'])>10*reference_peak
            verified.append(str(fine))
        np.savez(out/f'p{order}_curves.npz',allow_pickle=False,**curves)
        orders[f'p{order}']={'cases':rows,'time_step_check_L10':time_check}
    write(out/'metrics.json',{'status':'completed','orders':orders,'verified_runs':verified,'new_runs':16,'reused_runs':6,'primary_window_s':[0.,1.],'secondary_windows_s':[[.2,1.],[.5,1.]],'metric':'GLL weighted L2 of Vx/Vy minus expanded reference, physical top x=0..1400; common sample dt omitted','profile':'L labels: dmax*L constant; F labels p5: fixed dmax173.0886108476381; quadratic, ABC at three outer edges; no mass damping','growth_diagnostic':'strong_growth means late full-domain peak exceeds 10 times the full-time reference peak; includes PML and is a conservative gross-growth indicator, not a proof of long-time stability','reference':'reused 6400x3200 homogeneous domain, same local h/p/dt','timing_note':'single runs including full-domain output; output size decreases with thickness; historical L175 and old timings reused, not statistical kernel benchmarks'})
    print('THICKNESS ANALYSIS COMPLETED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--thickness',action='store_true');args=parser.parse_args()
    if args.thickness:analyze_thickness()
    else:analyze()
