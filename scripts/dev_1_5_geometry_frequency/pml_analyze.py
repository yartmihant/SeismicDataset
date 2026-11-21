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


def analyze_layered() -> None:
    from scripts.dev_1_5_geometry_frequency.pml_layered import ROOT as root,study_cases
    from semgpu2.model import SGPETALineModel
    out=root/'analysis';out.mkdir(exist_ok=False)
    t,x,ref=core_traces(root/'reference');th,xh,hom=core_traces(root/'homogeneous_reference')
    np.testing.assert_array_equal(t,th);np.testing.assert_allclose(x,xh,rtol=0,atol=1e-8)
    physical=ref-hom
    gll=SGPETALineModel('GLL',order=5).build_data(use_symmetry=False).weights
    ids=np.arange(280)[:,None]*5+np.arange(6)[None,:]
    weights=np.bincount(ids.ravel(),weights=np.tile(2.5*gll,280),minlength=1401)
    assert np.isclose(weights.sum(),1400.)
    refmeta=read_json(root/'reference/result.json');native=mapping(refmeta['native'])['sha256']
    refnodes=np.load(root/'reference/nodes.npy');reffields=np.load(root/'reference/velocity.npy',mmap_mode='r')
    refpeak=max(float(np.abs(f).max()) for f in reffields)
    early=t<=.1
    assert np.linalg.norm(physical[early])/np.linalg.norm(ref[early])<1e-9
    curves:dict[str,FloatArray]={'time_s':t,'x_m':x[:,0],'reference':ref,'homogeneous':hom,'physical':physical,'field_time_s':np.arange(21,dtype=np.float64)*.05}
    offset=np.abs(x[:,0]-700)
    bands={'all':np.ones(len(x),dtype=bool),'inner':offset<=400,'outer':offset>400,'central':offset<=200}
    windows={'full':(0.,1.),'echo_1':(.15,.25),'echo_2':(.25,.40),'echo_3':(.40,.60),'late':(.60,1.)}
    def norms(values: FloatArray,band: NDArray[np.bool_],mask: NDArray[np.bool_]) -> FloatArray:
        v=values[mask][:,band];w=weights[band]
        return np.r_[np.sqrt(np.sum(v*v*w[None,:,None])),np.max(np.abs(v)),np.max(np.abs(v),axis=(0,1))]
    physical_metrics:dict[str,object]={}
    for band,active in bands.items():
        physical_metrics[band]={win:norms(physical,active,(t>=a)&(t<=b)).tolist() for win,(a,b) in windows.items()}
    pp_windows:dict[str,object]={};pp_masks=[]
    central=offset<=200
    for bounce in range(1,6):
        arrivals=.05+np.sqrt((700*bounce)**2+offset[central]**2)/4087
        mask_pp=np.abs(t[:,None]-arrivals[None,:])<=.02
        pp_masks.append(mask_pp)
        pp_windows[str(bounce)]={'central_arrival_s':.05+700*bounce/4087,'half_width_s':.02,'physical_peak_vy_m_s':float(np.max(np.abs(physical[:,central,1][mask_pp])))}
    records:dict[str,object]={};verified=[]
    for label,spec in study_cases().items():
        base=root/label;meta=read_json(base/'result.json')
        assert meta['status']=='completed' and meta['dt_s']==refmeta['dt_s'] and mapping(meta['native'])['sha256']==native
        for key,name in [('fields','velocity.npy'),('nodes','nodes.npy'),('mass','core_mass.npy'),('surface','surface.npz')]:
            assert file_record(base/name)['sha256']==mapping(meta[key])['sha256']
        verified.append(label)
        if spec.mode=='reference':continue
        tt,xx,v=core_traces(base);np.testing.assert_array_equal(tt,t);np.testing.assert_allclose(xx,x,rtol=0,atol=1e-8)
        error=v-ref;ratios:dict[str,object]={};errors:dict[str,object]={}
        for band,active in bands.items():
            ratios[band]={};errors[band]={}
            br:dict[str,object]={};be:dict[str,object]={}
            for win,(a,b) in windows.items():
                mask=(t>=a)&(t<=b);denom=norms(physical,active,mask);num=norms(error,active,mask)
                br[win]=(num/denom).tolist();be[win]=num.tolist()
            ratios[band]=br;errors[band]=be
        nodes=np.load(base/'nodes.npy');fields=np.load(base/'velocity.npy',mmap_mode='r');core=core_indices(nodes)
        np.testing.assert_allclose(nodes[core],refnodes,rtol=0,atol=1e-8)
        peaks=[];pmlpeaks=[];core_error=[]
        outside=(nodes[:,0]<-1e-8)|(nodes[:,0]>1400+1e-8)|(nodes[:,1]<-700-1e-8)
        for f,r in zip(fields,reffields):
            assert np.isfinite(f).all();peaks.append(float(abs(f).max()));pmlpeaks.append(float(abs(f[outside]).max()));core_error.append(float(abs(f[core]-r).max()))
        pa=np.array(peaks);pmla=np.array(pmlpeaks)
        # Flags describe measured growth, not a proof of infinite-time stability.
        strong=bool(pa[10:].max()>10*refpeak)
        late_growth=bool(np.all(np.diff(pmla[12:])>0) and pmla[-1]>2*pmla[12])
        allratio=np.asarray(mapping(ratios['all'])['full'],dtype=float)
        pp_ratios:dict[str,object]={}
        for bounce,mask_pp in enumerate(pp_masks,1):
            pp_denom=number(mapping(pp_windows[str(bounce)])['physical_peak_vy_m_s'])
            pp_ratios[str(bounce)]={'matched_peak_ratio':float(np.max(np.abs(error[:,central,1][mask_pp]))/pp_denom),'full_time_peak_ratio':float(np.max(np.abs(error[:,central,1]))/pp_denom),'full_surface_both_peak_ratio':float(np.max(np.abs(error))/pp_denom)}
        curves[label+'_error']=error;curves[label+'_full_peak']=pa;curves[label+'_pml_peak']=pmla;curves[label+'_core_error_peak']=np.array(core_error)
        curves[label+'_surface_error_envelope']=np.max(abs(error),axis=1)
        records[label]={'thickness_m':spec.thickness,'rows':spec.thickness//5,'run_seconds':meta['run_seconds'],'dmax_s_inv':meta['dmax_or_alpha_s_inv'],'ratios_error_to_physical':ratios,'central_pp_window_ratios':pp_ratios,'error_norms':errors,'strong_growth':strong,'monotone_late_pml_growth':late_growth,'pml_peak_060_m_s':float(pmla[12]),'pml_peak_100_m_s':float(pmla[-1]),'late_full_peak_m_s':float(pa[10:].max()),'passes_100x_all_five_pp_global_peak':bool(all(number(mapping(value)['full_surface_both_peak_ratio'])<=.01 for value in pp_ratios.values()) and not strong and not late_growth),'passes_100x_global_l2_and_peak':bool(np.all(allratio[:2]<=.01) and not strong and not late_growth),'early_relative_l2':float(np.linalg.norm(error[early])/np.linalg.norm(ref[early]))}
    np.savez_compressed(out/'curves.npz',allow_pickle=False,**curves)
    write(out/'metrics.json',{'status':'completed','cases':records,'physical_reflection_norms':physical_metrics,'central_pp_windows':pp_windows,'central_pp_note':'Vy, offset<=200m, +/-20ms around geometric n*PP arrivals; windows can include other phases, not a pure-mode decomposition. Matched ratio, worst full-time central Vy error, and conservative full-surface/all-time/both-component peak relative to each window peak are reported. The last criterion asks for all five windows to exceed the global boundary error100x.','norm_order':['weighted_L2','peak_both','peak_Vx','peak_Vy'],'bands':'all x0..1400; inner offset<=400; outer offset>400; central offset<=200','windows':windows,'reference_interface_y_m':-350,'control_return_lower_bound_s':5700/4087,'physical_definition':'extended layered minus extended homogeneous; all interface-induced response including mode conversion and multiples, not individual pure phases','error_definition':'candidate minus extended layered; includes PML entrance, PML interior, outer boundary and numerical growth, not solely outer boundary','primary_criterion':'full-time/all-physical-top error/physical reflected response <=0.01 for weighted L2 and peak; reject strong or monotonic late PML growth. Echo windows and components additionally reported.','verified_runs':verified,'native_sha256':native,'dt_s':refmeta['dt_s'],'limitations':'1 s, central source30Hz, p5/h5; finite-domain numerical control not analytical exact solution; no long-time stability proof'})
    print('LAYERED ANALYSIS COMPLETED',flush=True)


def layered_tail_spectrum() -> None:
    from scripts.dev_1_5_geometry_frequency.pml_layered import ROOT as root
    from semgpu2.model import SGPETALineModel
    out=root/'analysis'
    gll=SGPETALineModel('GLL',order=5).build_data(use_symmetry=False).weights
    ids=np.arange(280)[:,None]*5+np.arange(6)[None,:]
    weights=np.bincount(ids.ravel(),weights=np.tile(2.5*gll,280),minlength=1401)
    cases:dict[str,object]={}
    with np.load(out/'curves.npz') as z:
        mask=z['time_s']>=.5
        for label in ('physical','L175_error','F550_error','F1200_error'):
            values=z[label][mask];freq=np.fft.rfftfreq(len(values),.001)
            spectrum=np.fft.rfft(values*np.hanning(len(values))[:,None,None],axis=0)
            power=np.sum(abs(spectrum)**2*weights[None,:,None],axis=(1,2));power[1:]*=2
            cases[label]={'peak_frequency_hz':float(freq[np.argmax(power)]),'fraction_power_above100hz':float(power[freq>100].sum()/power.sum())}
    write(out/'tail_spectrum.json',{'status':'completed','window_s':[.5,1.],'method':'Hann window, rFFT of sampled surface Vx/Vy, GLL spatial weights, summed components; one-sided power ratios; not an identification of the cause or a no-aliasing proof','sample_dt_s':.001,'cases':cases})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--thickness',action='store_true');parser.add_argument('--layered',action='store_true');parser.add_argument('--layered-spectrum',action='store_true');args=parser.parse_args()
    if args.layered_spectrum:layered_tail_spectrum()
    elif args.layered:analyze_layered()
    elif args.thickness:analyze_thickness()
    else:analyze()
