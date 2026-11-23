"""Независимая постобработка золотого сечения силы демпфирования слоя 175 м.

Читает завершённые поверхностные записи; никогда не запускает решатель.
Основная цель — L2 всех Vx/Vy за 0,5–1 с. Пики служат диагностикой.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import hashlib
from math import isfinite, sqrt
from pathlib import Path
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scripts.dev_1_4_server_preparation.profile_sweep_analysis import (
    DATA, IMAGES, Run, SOURCE_CENTER_S, amplitude, key_for_window,
    load_run, plt, rayleigh_speed,
)
from scripts.dev_1_4_server_preparation.compare_smoke import Surface, file_evidence
from scripts.dev_1_4_server_preparation.homogeneous_visualize import MATERIAL
from scripts.dev_1_3_point_source.physics import ricker
from scripts.dev_1_4_server_preparation.layer_analysis import beam, path_length
from scripts.dev_1_4_server_preparation.velocity_seismograms import save_figure
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json, sequence
from matplotlib.colors import SymLogNorm

FloatArray = NDArray[np.float64]
ROOT = DATA / "profile_golden_1s_3060_v1"
ALPHA0 = 346.1772216952762
COLORS = ("#444b54", "#2466a4", "#bf5b20", "#8f3184")
TOLERANCE = .02
PHI = (sqrt(5.) - 1.) / 2.
WINDOWS = ((.5,1.),(.5,.8),(.8,1.))


@dataclass(frozen=True)
class ProfileRun:
    factor: float
    run: Run
    evaluation_index: int


@dataclass(frozen=True)
class Interval:
    iteration: int
    lower: float
    upper: float
    left: float
    right: float
    left_objective: float
    right_objective: float

    @property
    def width(self) -> float:
        return self.upper - self.lower


def finite(value: object, name: str, *, positive: bool = False) -> float:
    result = number(value)
    if not isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Nonfinite or invalid {name}")
    return result


def objective(run: Run) -> float:
    mask = (run.surface.times >= .5) & (run.surface.times <= 1.)
    return float(np.linalg.norm(run.surface.velocity[mask]))


def source_prefix_hashes() -> dict[str, str]:
    full_times=np.linspace(0.,2.,192001)
    short_times=np.linspace(0.,1.,96001)
    if not np.array_equal(full_times[:96001],short_times):
        raise ValueError("Source sampling prefix differs")
    forces=np.zeros((len(full_times),1,2),dtype=np.float64)
    forces[:,0,1]=-ricker(full_times)
    short_forces=np.zeros((len(short_times),1,2),dtype=np.float64)
    short_forces[:,0,1]=-ricker(short_times)
    if not np.array_equal(forces[:96001],short_forces):
        raise ValueError("Source force prefix differs")
    return {"full2s":hashlib.sha256(forces.tobytes()).hexdigest(),
        "prefix1s":hashlib.sha256(short_forces.tobytes()).hexdigest()}


def load_profile(factor: float, directory: Path, index: int) -> ProfileRun:
    """Новые записи 1 с и неизменяемые префиксы прежних записей 2 с."""
    meta=read_json(directory/"metadata.json")
    horizon=finite(mapping(meta["run_spec"])["t_end_s"],"record duration",positive=True)
    name=f"k_{factor:.12f}";label=f"175 м · k={factor:.6f}"
    if horizon==2.:
        original=load_run(name,label,directory)
        source=original.surface
        surface=Surface(source.times[:1001],source.coordinates,source.node_ids,source.velocity[:1001])
        return ProfileRun(factor,Run(name,label,directory,meta,surface),index)
    if horizon!=1. or meta.get("status")!="completed" or meta.get("physical_model")!=MATERIAL or meta.get("is_original_dataset_model") is not False:
        raise ValueError("Need completed homogeneous1s result or historical2s control")
    validation=read_json(directory/"validation.json")
    if (directory/"metadata.failed.json").exists() or validation.get("status")!="passed":
        raise ValueError("Missing successful validation or failed run exists")
    if mapping(validation["metadata"])["sha256"]!=file_evidence(directory/"metadata.json")["sha256"]:
        raise ValueError("Validation belongs to different metadata")
    spec=mapping(meta["run_spec"])
    if any(spec.get(k)!=v for k,v in {"order":4,"substeps":96,"source_x_m":350.}.items()):
        raise ValueError("Wrong golden-run setup")
    runtime=mapping(meta["runtime"])
    if runtime["result_code"]!=runtime["time_end_code"] or runtime["steps_count"]!=96000 or not np.isclose(number(runtime["current_time_s"]),1.,rtol=0,atol=2e-10):
        raise ValueError("Incomplete1s run")
    if mapping(meta["recorded_counts"])["surface"]!=1001 or mapping(meta["recorded_counts"]).get("fields",0)!=0:
        raise ValueError("Expected1001 surface records without full fields")
    file=directory/"surface.npz"
    if file_evidence(file)["sha256"]!=mapping(meta["surface_file"])["sha256"]:
        raise ValueError("Surface SHA mismatch")
    with np.load(file,allow_pickle=False) as archive:
        if set(archive.files)!={"time_s","coordinates_m","node_ids","velocity_m_s"}:
            raise ValueError("Unexpected surface schema")
        times=cast(FloatArray,np.asarray(archive["time_s"]))
        coords=cast(FloatArray,np.asarray(archive["coordinates_m"]))
        velocity=cast(FloatArray,np.asarray(archive["velocity_m_s"]))
        ids=np.asarray(archive["node_ids"])
    if times.shape!=(1001,) or coords.shape!=(70,2) or velocity.shape!=(1001,70,2) or ids.shape!=(70,):
        raise ValueError("Wrong1s observation shape")
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in (times,coords,velocity)) or ids.dtype!=np.int64:
        raise ValueError("Invalid surface dtype or nonfinite values")
    if not np.array_equal(times,np.linspace(0.,1.,1001)) or not np.allclose(coords[:,0],np.arange(5.,700.,10.),rtol=0,atol=1e-9) or not np.allclose(coords[:,1],0.,rtol=0,atol=1e-9):
        raise ValueError("Unexpected time/receiver grid")
    if np.any(velocity[0]!=0) or np.any(ids<0) or len(np.unique(ids))!=70:
        raise ValueError("Invalid initial velocity or receiver IDs")
    return ProfileRun(factor,Run(name,label,directory,meta,Surface(times,coords,ids,velocity)),index)


def validate_profile(candidate: ProfileRun, baseline: Run) -> None:
    """Проверить произвольный k без изменения исторического sweep-валидатора."""
    run, factor = candidate.run, candidate.factor
    if not isfinite(factor) or not .25 <= factor <= 1.:
        raise ValueError("Damping factor is outside the approved search interval")
    for key in ("initial_state","physical_model","dt_s","source_node_id"):
        if run.metadata[key]!=baseline.metadata[key]:
            raise ValueError(f"Time-prefix physical setup differs: {key}")
    if mapping(run.metadata["native"])["sha256"]!=mapping(baseline.metadata["native"])["sha256"]:
        raise ValueError("Native solver binary changed")
    hashes=source_prefix_hashes()
    horizon=number(mapping(run.metadata["run_spec"])["t_end_s"])
    expected=hashes["full2s" if horizon==2. else "prefix1s"]
    if run.metadata["source_input_sha256"]!=expected or baseline.metadata["source_input_sha256"]!=hashes["full2s"]:
        raise ValueError("Historical and short forcing are not identical prefixes")
    if run.metadata["steps_count"]!=int(horizon*96000):
        raise ValueError("Wrong solver step count")
    if not np.array_equal(run.surface.times, baseline.surface.times) or not np.array_equal(run.surface.coordinates, baseline.surface.coordinates) or not np.array_equal(run.surface.node_ids,baseline.surface.node_ids):
        raise ValueError("Surface observation grid changed")
    for key in ("nodes_sha256", "abc_edge_count"):
        if run.metadata[key] != baseline.metadata[key]:
            raise ValueError(f"Fixed mesh/ABC changed: {key}")
    if mapping(run.metadata["mesh_evidence"])["sha256"] != mapping(baseline.metadata["mesh_evidence"])["sha256"]:
        raise ValueError("Fixed175m mesh changed")
    if factor != 1.:
        profile = mapping(run.metadata["profile"])
        if profile["shape"] != "power" or profile["exponent"] != 2. or not np.isclose(number(profile["amplitude_factor"]), factor, rtol=0, atol=2e-14):
            raise ValueError("Run does not implement the declared quadratic factor")
    if not np.isclose(number(mapping(run.metadata["run_spec"])["alpha_max_s_inv"]), ALPHA0*factor, rtol=2e-14, atol=1e-13):
        raise ValueError("Incorrect coefficient in solver metadata")
    if not isfinite(objective(run)) or objective(run) <= 0:
        raise ValueError("Invalid objective")


def validate_history(history: list[Interval], final_interval: tuple[float,float], tolerance: float = TOLERANCE) -> dict[str, object]:
    """Проверить геометрию интервалов и решение ветвления золотого сечения."""
    if not history or tolerance != TOLERANCE:
        raise ValueError("Need the approved golden-section history and tolerance")
    if not np.allclose([history[0].lower, history[0].upper], [.25, 1.], rtol=0, atol=2e-13):
        raise ValueError("Wrong starting interval")
    for index, step in enumerate(history):
        values = (step.lower, step.upper, step.left, step.right, step.left_objective, step.right_objective)
        if step.iteration != index or not all(isfinite(v) for v in values):
            raise ValueError("Invalid history index or nonfinite value")
        if not .25 <= step.lower < step.left < step.right < step.upper <= 1.:
            raise ValueError("Invalid golden-section point ordering")
        if min(step.left_objective, step.right_objective) <= 0:
            raise ValueError("Invalid history objective")
        expected = (step.upper-PHI*step.width, step.lower+PHI*step.width)
        if not np.allclose([step.left, step.right], expected, rtol=0, atol=2e-12):
            raise ValueError("Interior points do not satisfy the golden ratio")
        if index:
            old = history[index-1]
            expected_interval = (old.lower, old.right) if old.left_objective <= old.right_objective else (old.left, old.upper)
            if not np.allclose([step.lower, step.upper], expected_interval, rtol=0, atol=2e-12):
                raise ValueError("Interval shrink disagrees with evaluated objective values")
            if not np.isclose(step.width/old.width, PHI, rtol=0, atol=2e-11):
                raise ValueError("Unexpected interval contraction")
    last=history[-1]
    expected_final=(last.lower,last.right) if last.left_objective<=last.right_objective else (last.left,last.upper)
    if not np.allclose(final_interval,expected_final,rtol=0,atol=2e-12):
        raise ValueError("Final interval does not follow the last comparison")
    final_width=final_interval[1]-final_interval[0]
    if final_width > tolerance or last.width <= tolerance:
        raise ValueError("Search did not stop at the requested first interval width")
    return {"status": "passed", "initial_interval": [.25, 1.],
        "final_interval": list(final_interval),
        "final_width": final_width, "tolerance": tolerance,
        "contractions": len(history), "expected_contraction_factor": PHI,
        "unimodality": "Assumed by golden-section search; a finite sampled curve and interval contraction do not prove unimodality or a global optimum"}


def peak_rows(run: Run, baseline: Run, mode: str) -> list[dict[str, object]]:
    times, xs = run.surface.times, run.surface.coordinates[:, 0]
    rows: list[dict[str, object]] = []
    speed = rayleigh_speed()
    for index, x in enumerate(xs):
        start = SOURCE_CENTER_S+abs(float(x)-350.)/speed+.08 if mode == "after_direct" else .5
        mask = (times >= start) & (times <= 1.)
        values = np.linalg.norm(run.surface.velocity[mask, index], axis=1)
        reference = np.linalg.norm(baseline.surface.velocity[mask, index], axis=1)
        peak, base_peak = float(np.max(values)), float(np.max(reference))
        if base_peak <= 0:
            raise ValueError("Zero baseline receiver peak")
        rows.append({"x_m": float(x), "gate_s": [start, 1.], "peak_vector_m_s": peak,
            "peak_time_s": float(times[mask][int(np.argmax(values))]),
            "baseline_peak_vector_m_s": base_peak, "peak_ratio_baseline": peak/base_peak,
            "l2_ratio_baseline": float(np.linalg.norm(values)/np.linalg.norm(reference))})
    return rows


def run_metrics(profile: ProfileRun, baseline: Run) -> dict[str, object]:
    run, factor = profile.run, profile.factor
    times, xs = run.surface.times, run.surface.coordinates[:, 0]
    windows: dict[str, object] = {}
    for start, end in WINDOWS:
        mask = (times >= start) & (times <= end)
        row = amplitude(run.surface.velocity[mask])
        ratio = row["l2_samples_m_s"]/float(np.linalg.norm(baseline.surface.velocity[mask]))
        row.update(l2_relative_baseline=ratio, amplitude_db_relative_baseline=float(20*np.log10(ratio)))
        windows[key_for_window(start,end)] = row
    peaks: dict[str, object] = {}
    for mode in ("after_direct", "late_0.5_1"):
        rows = peak_rows(run, baseline, mode)
        ratios = np.array([number(row["peak_ratio_baseline"]) for row in rows])
        worst = rows[int(np.argmax(ratios))]
        largest=max(rows,key=lambda row:number(row["peak_vector_m_s"]))
        peaks[mode] = {"all_70_receivers": rows, "maximum_peak_ratio": float(np.max(ratios)),
            "worst_ratio_receiver": worst, "receivers_above_baseline": int(np.count_nonzero(ratios>1.+1e-8)),
            "maximum_vector_peak_m_s":largest["peak_vector_m_s"],"maximum_vector_peak_receiver":largest,
            "no_receiver_worse_than_baseline": bool(np.all(ratios<=1.+1e-8)),
            "selected_receivers": [row for row in rows if row["x_m"] in (5.,175.,345.)]}
    beams: list[dict[str, object]] = []
    for xmax in (175.,315.):
        tau, values = beam(times,xs,run.surface.velocity,rayleigh_speed(),xmax)
        _, ref = beam(times,xs,baseline.surface.velocity,rayleigh_speed(),xmax)
        by_window: dict[str, object] = {}
        for name,start,end in (("early",.18,.30),("outer",.30,.46),("all",.18,.70)):
            mask = (tau>=start)&(tau<=end)
            row = amplitude(values[mask])
            row.update(l2_relative_baseline=float(np.linalg.norm(values[mask])/np.linalg.norm(ref[mask])),
                peak_tau_s=float(tau[mask][int(np.argmax(np.linalg.norm(values[mask],axis=1)))]))
            by_window[name] = {"tau_interval_s": [start,end], **row}
        beams.append({"receiver_range_m": [65.,xmax], "windows": by_window})
    delta = run.surface.velocity-baseline.surface.velocity
    pre_layer = times[:,None]<path_length(xs,0.)[None,:]/4087.-.003
    return {"factor": factor, "alpha_peak_per_direction_s_inv": ALPHA0*factor,
        "label": run.label, "evaluation_index": profile.evaluation_index,
        "source_record_horizon_s":number(mapping(run.metadata["run_spec"])["t_end_s"]),
        "analysis_horizon_s":1.,
        "early_0_0.5_exclusive_diagnostic":amplitude(run.surface.velocity[times<.5]),
        "surface_windows": windows, "peak_guards": peaks, "incoming_beams": beams,
        "pre_layer_max_abs_delta_over_baseline_peak": float(np.max(np.abs(delta[pre_layer]))/np.max(np.abs(baseline.surface.velocity))),
        "wall_seconds": number(run.metadata["wall_seconds"]),
        "recording_mode":"all-node fields + surface" if mapping(run.metadata["recorded_counts"]).get("fields",0) else "surface only",
        "inputs": {"metadata": file_evidence(run.directory/"metadata.json"), "surface": file_evidence(run.directory/"surface.npz")}}


def summarize(profiles: list[ProfileRun], history: list[Interval], final_interval: tuple[float,float], selected_factor: float) -> dict[str, object]:
    convergence = validate_history(history,final_interval)
    baseline = next(p.run for p in profiles if p.factor == 1.)
    if len({p.factor for p in profiles}) != len(profiles):
        raise ValueError("Duplicate factor evaluations in final manifest")
    for profile in profiles:
        validate_profile(profile,baseline)
    selected = next(p for p in profiles if np.isclose(p.factor,selected_factor,rtol=0,atol=2e-14))
    best = min(profiles,key=lambda p:objective(p.run))
    if objective(selected.run) != objective(best.run):
        raise ValueError("Selected factor is not the minimum observed objective")
    for step in history:
        for factor,value in ((step.left,step.left_objective),(step.right,step.right_objective)):
            match = next((p for p in profiles if np.isclose(p.factor,factor,rtol=0,atol=2e-14)),None)
            if match is None or not np.isclose(objective(match.run),value,rtol=2e-12,atol=0):
                raise ValueError("History objective does not match its saved surface")
    sorted_profiles = sorted(profiles,key=lambda p:p.factor)
    objectives = np.array([objective(p.run) for p in sorted_profiles])
    signs = np.sign(np.diff(objectives))
    decreasing_after_increase = bool(np.any(np.maximum.accumulate(signs)>0) and any(signs[i]<0 and np.any(signs[:i]>0) for i in range(len(signs))))
    rows = [run_metrics(p,baseline) for p in sorted_profiles]
    selected_metrics = next(row for row in rows if row["factor"] == selected.factor)
    guards = {key: {"l2_ratio_baseline": mapping(value)["l2_relative_baseline"],
        "no_worse_than_baseline": number(mapping(value)["l2_relative_baseline"])<=1.+1e-8}
        for key,value in mapping(selected_metrics["surface_windows"]).items()}
    return {"status": "completed", "selected_factor": selected.factor,
        "selected_objective_m_s": objective(selected.run), "selected_l2_ratio_baseline": objective(selected.run)/objective(baseline),
        "all_evaluations": rows, "convergence": convergence,
        "history": [{"iteration":h.iteration,"lower":h.lower,"upper":h.upper,"width":h.width,"left":h.left,"right":h.right,"left_objective":h.left_objective,"right_objective":h.right_objective} for h in history],
        "selection": {"objective": "Minimum observed unweighted L2 of all Vx/Vy surface samples over inclusive0.5–1s; peak checks do not alter selection",
            "selected_inside_final_interval": final_interval[0]<=selected.factor<=final_interval[1],
            "sampled_curve_decreases_after_an_increase": decreasing_after_increase,
            "window_guards": guards, "decision_scope": "Best evaluated coefficient in this golden-section search on one homogeneous central-source model; no independent proof of global optimum or solver convergence"},
        "method": {"amplitudes": "L2/RMS and receiver peaks are velocity-sample statistics, not energy or reflection coefficients",
            "peak_gates": "All70 receivers: (a) t0+abs(x-350)/cR+80ms to1s, (b)0.5–1s. Edge gates can omit early returns. Maxima at different times need not be the same arrival",
            "prefix_compatibility":{"recording_times_identical":True,"solver_dt_and_native_identical":True,"reconstructed_source_force_hashes":source_prefix_hashes(),"meaning":"Historical2s surfaces are read-only prefixes. Source samples at1s exactly match the prefix of2s forcing; same initial state, material, mesh, ABC, dt and native verified. This is provenance compatibility, not an independently rerun control."},
            "beam": "Average after t=tau+x/cR; directional diagnostic, not pure modal decomposition",
            "timing":"Historical wall times describe the complete2s runs, some with all-node fields; new wall times describe1s surface-only runs and are not a direct speed comparison",
            "unimodality": convergence["unimodality"], "field_output": "Surface-only candidate runs; no new all-node velocity fields"}}


def display_profiles(profiles: list[ProfileRun], factor: float) -> list[ProfileRun]:
    return [next(p for p in profiles if np.isclose(p.factor,k,rtol=0,atol=2e-14)) for k in (1.,.5,.25,factor)]


def interval_pair(value: object) -> tuple[float,float]:
    values=sequence(value)
    if len(values)!=2:
        raise ValueError("Expected interval endpoints")
    return finite(values[0],"lower"),finite(values[1],"upper")


def history_from_records(records: list[dict[str,object]], evaluations: dict[int,ProfileRun]) -> tuple[list[Interval],tuple[float,float]]:
    history: list[Interval]=[]
    final_interval=(.25,1.)
    for index,row in enumerate(records):
        if number(row["iteration"])!=index+1:
            raise ValueError("Noncontiguous runner history")
        lower,upper=interval_pair(row["before_interval"])
        step=Interval(index,lower,upper,finite(row["left"],"left"),finite(row["right"],"right"),
            finite(row["left_objective"],"left objective",positive=True),finite(row["right_objective"],"right objective",positive=True))
        ids=sequence(row["compared_ids"])
        if len(ids)!=2:
            raise ValueError("Expected two compared evaluation IDs")
        for identifier,factor in zip(ids,(step.left,step.right),strict=True):
            value=number(identifier)
            if not value.is_integer() or int(value) not in evaluations or not np.isclose(evaluations[int(value)].factor,factor,rtol=0,atol=2e-14):
                raise ValueError("History references a different or missing evaluation")
        final_interval=(finite(row["lower"],"after lower"),finite(row["upper"],"after upper"))
        expected=(step.lower,step.right) if step.left_objective<=step.right_objective else (step.left,step.upper)
        expected_branch="keep_left" if step.left_objective<=step.right_objective else "keep_right"
        if not np.allclose(final_interval,expected,rtol=0,atol=2e-12) or row["branch"]!=expected_branch:
            raise ValueError("Recorded branch and interval disagree")
        if not np.isclose(finite(row["width"],"width",positive=True),final_interval[1]-final_interval[0],rtol=0,atol=2e-13):
            raise ValueError("Recorded interval width disagrees")
        history.append(step)
    return history,final_interval


def read_search(root: Path) -> tuple[list[ProfileRun],list[Interval],tuple[float,float],float,dict[str,object]]:
    root.resolve().relative_to(DATA)
    result_path=root/"result.json";manifest_path=root/"manifest.json"
    result=read_json(result_path);manifest=read_json(manifest_path)
    if result.get("status")!="completed" or (root/"failure.json").exists():
        raise ValueError("Search is incomplete or has failure evidence")
    if result["manifest"]!=file_evidence(manifest_path):
        # Runner file_record also includes size; the SHA is the common identity.
        if mapping(result["manifest"])["sha256"]!=file_evidence(manifest_path)["sha256"]:
            raise ValueError("Search manifest hash changed")
    approved={"lower":.25,"upper":1.,"tolerance":.02,"max_evaluations":9}
    for document in (result,manifest):
        if document["config"]!=approved or document["t_end_s"]!=1. or document["time_window_s"]!=[.5,1.]:
            raise ValueError("Search configuration or objective window changed")
    source=mapping(manifest["source_prefix"]);hashes=source_prefix_hashes()
    if source["historical_2s_sha256"]!=hashes["full2s"] or source["new_1s_sha256"]!=hashes["prefix1s"] or source["prefix_array_equal"] is not True or source["target_samples"]!=96001:
        raise ValueError("Source-prefix evidence disagrees with independent reconstruction")
    profiles: list[ProfileRun]=[]
    controls=[mapping(row) for row in sequence(manifest["controls"])]
    if [row["factor"] for row in controls]!=[1.,.5,.25]:
        raise ValueError("Unexpected historical controls")
    evaluation_records=[mapping(row) for row in sequence(result["evaluations"])]
    if len(evaluation_records)!=9 or result["new_evaluations"]!=9:
        raise ValueError("Expected nine golden evaluations")
    evaluated: dict[int,ProfileRun]={}
    for position,row in enumerate([*controls,*evaluation_records]):
        factor=finite(row["factor"],"factor",positive=True)
        directory=Path(str(row["directory"])).resolve()
        directory.relative_to(DATA)
        index=position-3 if position<3 else position-2
        if position>=3:
            if number(row["index"])!=index or directory!=(root/"runs"/f"eval_{index:02d}").resolve():
                raise ValueError("Wrong evaluation index or run directory")
            ledger=root/"evaluations"/f"eval_{index:02d}.json"
            if read_json(ledger)!=row:
                raise ValueError("Result and immutable evaluation ledger disagree")
            if mapping(row["validation"])["sha256"]!=file_evidence(directory/"validation.json")["sha256"]:
                raise ValueError("Validation file changed after evaluation")
        for key,filename in (("metadata","metadata.json"),("surface","surface.npz")):
            if mapping(row[key])["sha256"]!=file_evidence(directory/filename)["sha256"]:
                raise ValueError("Input identity changed since evaluation")
        profile=load_profile(factor,directory,index)
        if not np.isclose(finite(row["objective"],"recorded objective",positive=True),objective(profile.run),rtol=2e-12,atol=0):
            raise ValueError("Recorded objective disagrees with surface samples")
        if position>=3:
            if profile.run.metadata.get("source_prefix")!=manifest["source_prefix"]:
                raise ValueError("Run carries different source-prefix evidence")
            request=mapping(profile.run.metadata["golden_request"])
            if request["index"]!=index or request["factor"]!=factor:
                raise ValueError("Run factor differs from exact request")
            baseline=finite(controls[0]["objective"],"baseline objective",positive=True)
            if not np.isclose(finite(row["relative_baseline"],"relative objective",positive=True),objective(profile.run)/baseline,rtol=2e-12,atol=0):
                raise ValueError("Recorded objective ratio disagrees")
            evaluated[index]=profile
        profiles.append(profile)
    history,final_interval=history_from_records([mapping(row) for row in sequence(result["history"])],evaluated)
    if len(history)!=8 or not np.allclose(interval_pair(result["final_interval"]),final_interval,rtol=0,atol=2e-13):
        raise ValueError("Final interval does not agree with eight contractions")
    if not np.isclose(finite(result["final_width"],"final width",positive=True),final_interval[1]-final_interval[0],rtol=0,atol=2e-13) or result["tolerance"]!=TOLERANCE:
        raise ValueError("Final convergence parameters differ")
    selected=finite(result["selected_factor"],"selected factor",positive=True)
    if mapping(result["best_observed"])["factor"]!=selected:
        raise ValueError("Selected factor and best-observed record disagree")
    best_new=min(evaluated.values(),key=lambda p:objective(p.run))
    if not np.isclose(finite(mapping(result["best_new"])["objective"],"best new objective",positive=True),objective(best_new.run),rtol=2e-12,atol=0):
        raise ValueError("Best-new record is not the minimum new observation")
    evidence={"result":file_evidence(result_path),"manifest":file_evidence(manifest_path),
        "source_prefix":source,"new_runs":len(evaluated),"historical_controls":len(controls)}
    return profiles,history,final_interval,selected,evidence


def plot_results(profiles: list[ProfileRun], history: list[Interval], report: dict[str,object], output: Path) -> list[dict[str,object]]:
    figures: list[dict[str,object]] = []
    selected_factor = number(report["selected_factor"])
    shown = display_profiles(profiles,selected_factor)
    labels = ("Исходный · k=1", "Половина · k=0,5", "Четверть · k=0,25", f"Выбранный · k={selected_factor:.5f}")
    times,xs = shown[0].run.surface.times,shown[0].run.surface.coordinates[:,0]
    baseline_objective = objective(shown[0].run)
    ordered = sorted(profiles,key=lambda p:p.factor)
    convergence=mapping(report["convergence"])
    final_bounds=[number(v) for v in sequence(convergence["final_interval"])]
    fig,axes = plt.subplots(1,2,figsize=(14.5,5.3),layout="constrained")
    axes[0].plot([p.factor for p in ordered],[objective(p.run)/baseline_objective for p in ordered],"o-",color="#50565d",lw=1.2)
    axes[0].axvspan(final_bounds[0],final_bounds[1],color=COLORS[3],alpha=.13,label="Финальный интервал")
    for profile,label,color in zip(shown,labels,COLORS,strict=True):
        axes[0].scatter(profile.factor,objective(profile.run)/baseline_objective,s=70,color=color,zorder=3,label=label)
    axes[0].set(xlabel="Множитель k в α=kα₀(s/175)²",ylabel="L2 поверхности за 0,5–1 с / исходная L2",title="Фактически вычисленные значения цели")
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    axes[1].semilogy(list(range(len(history)+1)),[*[h.width for h in history],number(convergence["final_width"])],"o-",color=COLORS[3])
    axes[1].axhline(TOLERANCE,color="#555555",ls="--",label="Требуемая ширина 0,02")
    axes[1].set(xlabel="Число сужений интервала",ylabel="Ширина интервала k",title="Остановка по ширине интервала")
    axes[1].legend(fontsize=9);axes[1].grid(alpha=.2,which="both")
    fig.suptitle("Золотое сечение · минимизируется фактическая поздняя поверхностная запись",fontsize=14)
    fig.supxlabel("Линии соединяют вычисленные точки; унимодальность функции предполагается, но этим опытом не доказана.",fontsize=9)
    figures.extend(save_figure(fig,output,"objective_and_convergence"))
    fig,axes = plt.subplots(1,2,figsize=(14.5,5.3),layout="constrained")
    s=np.linspace(0.,175.,701);x=np.linspace(-175.,875.,2101)
    distance=np.maximum.reduce((-x,x-700.,np.zeros_like(x)))
    axes[1].axvspan(0,700,color="#eceff1")
    for p,label,color in zip(shown,labels,COLORS,strict=True):
        axes[0].plot(s,ALPHA0*p.factor*(s/175)**2,color=color,lw=2,label=label)
        axes[1].plot(x,ALPHA0*p.factor*(distance/175)**2,color=color,lw=2,label=label)
    axes[0].set(xlabel="Глубина в слой s, м",ylabel="Коэффициент α, с⁻¹",title="Поперёк слоя толщиной 175 м",xlim=(0,175))
    axes[1].set(xlabel="Координата верхней поверхности x, м",ylabel="Коэффициент α(x,0), с⁻¹",title="Свободный верх y=0 и боковые продолжения",xlim=(-175,875),xticks=[-175,0,175,350,525,700,875])
    axes[0].legend(fontsize=9)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle("Выбранный коэффициент массового демпфирования · C=αM",fontsize=14)
    fig.supxlabel("В основной области α=0; в нижних углах складываются боковой и нижний вклады. Это коэффициент, не амплитуда волны.",fontsize=9)
    figures.extend(save_figure(fig,output,"selected_profiles"))
    fig,axes=plt.subplots(2,4,figsize=(18,10),sharex=True,sharey=True,layout="constrained")
    mask=(times>=.2)&(times<=1.)
    maximum=max(float(np.max(np.abs(p.run.surface.velocity[mask]))) for p in shown)*1e9
    norm=SymLogNorm(linthresh=.001,vmin=-maximum,vmax=maximum,base=10)
    mesh=None
    for component in range(2):
        for column,(p,label) in enumerate(zip(shown,labels,strict=True)):
            ax=axes[component,column]
            mesh=ax.pcolormesh(xs,times,p.run.surface.velocity[:,:,component]*1e9,cmap="RdBu_r",norm=norm,shading="nearest",rasterized=True)
            ax.set(xlim=(0,700),ylim=(1.,.2),title=f"{'Vx' if component==0 else 'Vy'} · {label}")
            ax.axhline(.5,color="#444444",ls=":",lw=1)
            if column==0:ax.set_ylabel("Время, с")
            if component==1:ax.set_xlabel("x, м")
    if mesh is None:raise AssertionError("Missing seismograms")
    fig.colorbar(mesh,ax=axes.ravel().tolist(),label="Скорость, нм/с · общая SymLog-шкала; линейно до 0,001",shrink=.8)
    fig.suptitle("Верхняя поверхность · общая физическая шкала\nЦель оценивается за 0,5–1 с; более ранняя часть показана для диагностики",fontsize=15)
    figures.extend(save_figure(fig,output,"surface_seismograms"))
    fig,ax=plt.subplots(figsize=(13.5,6),layout="constrained")
    for p,label,color in zip(shown,labels,COLORS,strict=True):
        rms=np.sqrt(np.mean(p.run.surface.velocity**2,axis=(1,2)))
        ax.semilogy(times,np.maximum(rms*1e9,1e-15),color=color,lw=1.8,label=label)
    ax.axvline(.5,color="#777777",ls=":")
    ax.set(xlim=(.2,1),ylim=(1e-6,1),xlabel="Время, с",ylabel="RMS компонент поверхности, нм/с",title="Остаточная поверхностная скорость · цель за 0,5–1 с")
    ax.legend();ax.grid(alpha=.2,which="both")
    figures.extend(save_figure(fig,output,"surface_tail_rms"))
    fig,axes=plt.subplots(3,2,figsize=(15,12),sharex=True,layout="constrained")
    trace_mask=(times>=.18)&(times<=1.)
    for row,xval in enumerate((5.,175.,345.)):
        index=int(np.flatnonzero(xs==xval)[0])
        for component in range(2):
            ax=axes[row,component]
            for p,label,color in zip(shown,labels,COLORS,strict=True):
                ax.plot(times[trace_mask],p.run.surface.velocity[trace_mask,index,component]*1e9,color=color,lw=1.3,label=label)
            ax.set_yscale("symlog",linthresh=1e-4)
            ax.axvline(.5,color="#777777",ls=":",lw=1)
            ax.set(xlim=(.18,1.),ylabel="Скорость, нм/с",title=f"{'Vx' if component==0 else 'Vy'} · x={xval:g} м")
            ax.grid(alpha=.2,which="both")
            if row==2:ax.set_xlabel("Время, с")
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Фактические поверхностные трассы · без индивидуальной нормировки",fontsize=14)
    figures.extend(save_figure(fig,output,"return_traces"))
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,layout="constrained")
    for column,mode in enumerate(("after_direct","late_0.5_1")):
        for p,label,color in zip(shown,labels,COLORS,strict=True):
            rows=peak_rows(p.run,shown[0].run,mode)
            peaks=[number(row["peak_vector_m_s"])*1e9 for row in rows]
            ratios=[number(row["peak_ratio_baseline"]) for row in rows]
            axes[0,column].plot(xs,peaks,color=color,lw=1.6,label=label)
            if p.factor!=1.:axes[1,column].plot(xs,ratios,color=color,lw=1.6,label=label)
        axes[0,column].set(ylabel="Максимум |v|, нм/с",title="После прямой волны + 80 мс … 1 с" if column==0 else "Общее окно 0,5–1 с")
        axes[1,column].axhline(1,color="#555555",ls=":")
        axes[1,column].set(xlabel="Координата приёмника x, м",ylabel="Локальный максимум / исходный максимум")
        for row in range(2):axes[row,column].grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Пики всех 70 приёмников · дополнительная проверка компромиссов",fontsize=14)
    fig.supxlabel("Эти показатели не заменяют выбранную цель L2. Пики разных вариантов могут относиться к разным приходам.",fontsize=9)
    figures.extend(save_figure(fig,output,"all_receiver_peak_guards"))
    return figures


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path,default=DATA/"profile_golden_1s_analysis_v1")
    parser.add_argument("--images",type=Path,default=IMAGES/"profile_golden_1s_comparison_v1")
    args=parser.parse_args()
    output:Path=args.out;images:Path=args.images
    output.resolve().relative_to(DATA);images.resolve().relative_to(IMAGES)
    if output.exists() or images.exists():raise FileExistsError("Choose fresh analysis output directories")
    profiles,history,final_interval,selected,evidence=read_search(args.root)
    report=summarize(profiles,history,final_interval,selected)
    output.mkdir(parents=True,exist_ok=False);images.mkdir(parents=True,exist_ok=False)
    report.update(schema="chapter14.profile_golden_1s_analysis.v1",utc=datetime.now(timezone.utc).isoformat(),
        code=file_evidence(Path(__file__)),search=evidence,figures=plot_results(profiles,history,report,images))
    with (output/"analysis.json").open("x",encoding="utf-8") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"status":"completed","selected_factor":selected,
        "relative_baseline":report["selected_l2_ratio_baseline"],"final_interval":final_interval},ensure_ascii=False))


if __name__ == "__main__":
    main()
