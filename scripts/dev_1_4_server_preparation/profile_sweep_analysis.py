"""Сравнение трёх сил квадратичного демпфирования в неизменном слое 175 м."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scripts.project_paths import PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

from scripts.dev_1_4_server_preparation.compare_smoke import Surface, file_evidence
from scripts.dev_1_4_server_preparation.homogeneous_visualize import MATERIAL, load_surface, validate_pair
from scripts.dev_1_4_server_preparation.layer_analysis import beam, path_length, rayleigh_speed
from scripts.dev_1_4_server_preparation.velocity_seismograms import save_figure
from scripts.reporting import mapping, number, read_json

FloatArray = NDArray[np.float64]
DATA = PROJECT_ROOT / "data/dev_1_4_server_preparation"
IMAGES = PROJECT_ROOT / "img/dev_1_4_server_preparation"
NAMES = ("baseline", "q2_half", "q2_quarter", "wide350")
LABELS = ("175 м · α₀", "175 м · 0,5α₀", "175 м · 0,25α₀", "350 м · прежний плавный профиль")
COLORS = ("#444b54", "#2466a4", "#bf5b20", "#397c57")
FACTORS = (1., .5, .25, 1.)
WINDOWS = ((.3, 2.), (.3, .8), (.8, 2.), (1., 2.))
SOURCE_CENTER_S = .05
VP_M_S = 4087.


@dataclass(frozen=True)
class Run:
    name: str
    label: str
    directory: Path
    metadata: dict[str, object]
    surface: Surface


def load_run(name: str, label: str, directory: Path) -> Run:
    meta = read_json(directory/"metadata.json")
    if meta.get("status") != "completed" or meta.get("physical_model") != MATERIAL or meta.get("is_original_dataset_model") is not False:
        raise ValueError(f"Need completed homogeneous run: {directory}")
    if read_json(directory/"validation.json").get("status") != "passed":
        raise ValueError("Run validation is missing")
    if (directory/"metadata.failed.json").exists():
        raise ValueError("Failed metadata coexists with result")
    spec = mapping(meta["run_spec"])
    for key, value in {"t_end_s": 2., "substeps": 96, "order": 4, "source_x_m": 350.}.items():
        if spec.get(key) != value:
            raise ValueError(f"Unexpected run parameter {key}")
    runtime = mapping(meta["runtime"])
    if runtime["result_code"] != runtime["time_end_code"] or runtime["steps_count"] != 192000 or not np.isclose(number(runtime["current_time_s"]), 2., rtol=0, atol=2e-10):
        raise ValueError("Incomplete 2-second record")
    if mapping(meta["recorded_counts"])["surface"] != 2001:
        raise ValueError("Wrong surface sample count")
    return Run(name, label, directory, meta, load_surface(directory, meta))


def validate_runs(runs: list[Run]) -> None:
    baseline = runs[0]
    for run in runs[1:]:
        validate_pair(baseline.metadata, run.metadata)
        if not np.array_equal(run.surface.times, baseline.surface.times) or not np.allclose(run.surface.coordinates, baseline.surface.coordinates, rtol=0, atol=1e-9):
            raise ValueError("Observation axes differ")
    for run, factor in zip(runs[1:3], FACTORS[1:3], strict=True):
        if mapping(run.metadata["mesh_evidence"])["sha256"] != mapping(baseline.metadata["mesh_evidence"])["sha256"]:
            raise ValueError("Candidate changed the fixed 175m mesh")
        if run.metadata.get("nodes_sha256") != baseline.metadata["nodes_sha256"]:
            raise ValueError("Candidate changed SEM node coordinates/order")
        if run.metadata.get("abc_edge_count") != baseline.metadata["abc_edge_count"]:
            raise ValueError("Candidate changed ABC edge count")
        profile = mapping(run.metadata["profile"])
        if profile.get("name") != run.name or profile.get("shape") != "power" or profile.get("exponent") != 2. or profile.get("amplitude_factor") != factor:
            raise ValueError("Unexpected candidate damping profile")


def amplitude(values: FloatArray) -> dict[str, float]:
    """Нормы записей без пространственной квадратуры и физической энергии."""
    if values.size == 0 or values.shape[-1] != 2 or not np.isfinite(values).all():
        raise ValueError("Empty or invalid velocity window")
    magnitude = np.linalg.norm(values, axis=-1)
    return {"l2_samples_m_s": float(np.linalg.norm(values)),
        "rms_component_m_s": float(np.sqrt(np.mean(values**2))),
        "max_vector_m_s": float(np.max(magnitude))}


def key_for_window(start: float, end: float) -> str:
    return f"{start:g}_{end:g}"


def analyze(runs: list[Run]) -> dict[str, object]:
    times, xs = runs[0].surface.times, runs[0].surface.coordinates[:, 0]
    speed = rayleigh_speed()
    records: dict[str, object] = {}
    for run in runs:
        windows: dict[str, object] = {}
        for start, end in WINDOWS:
            mask = (times >= start) & (times <= end)
            row = amplitude(run.surface.velocity[mask])
            baseline_norm = float(np.linalg.norm(runs[0].surface.velocity[mask]))
            if baseline_norm <= 0:
                raise ValueError("The baseline window contains no velocity signal")
            ratio = row["l2_samples_m_s"]/baseline_norm
            row.update(l2_relative_baseline=ratio, amplitude_db_relative_baseline=float(20*np.log10(ratio)))
            windows[key_for_window(start, end)] = row
        receiver_records: list[dict[str, object]] = []
        for x in (5., 175., 345.):
            index = int(np.flatnonzero(xs == x)[0])
            direct_time = SOURCE_CENTER_S+abs(x-350.)/speed
            start = direct_time+.08
            mask = times >= start
            values = run.surface.velocity[mask, index]
            row = amplitude(values)
            ids = np.flatnonzero(mask)
            row["peak_time_s"] = float(times[ids[np.argmax(np.linalg.norm(values, axis=1))]])
            receiver_records.append({"x_m": x, "direct_rayleigh_peak_estimate_s": direct_time,
                "gate_s": [start, 2.], "after_direct_plus80ms": row})
        difference = run.surface.velocity-runs[0].surface.velocity
        outer_limit = path_length(xs, 175.)/VP_M_S-.003
        pre_outer = times[:, None] < outer_limit[None, :]
        layer_limit = path_length(xs, 0.)/VP_M_S-.003
        pre_layer = times[:, None] < layer_limit[None, :]
        baseline_peak = float(np.max(np.abs(runs[0].surface.velocity)))
        changes = {"before_old175_outer_P_path": amplitude(difference[pre_outer]),
            "before_layer_interface_P_path": amplitude(difference[pre_layer]),
            "pre_layer_max_abs_relative_baseline_global_peak": float(np.max(np.abs(difference[pre_layer])))/baseline_peak,
            "margin_s": .003,
            "meaning": "Changes from the baseline before paths from the layer interface or outer ABC; not an objective error norm"}
        beams: list[dict[str, object]] = []
        for xmax in (175., 315.):
            tau, mean = beam(times, xs, run.surface.velocity, speed, xmax)
            _, mean_delta = beam(times, xs, difference, speed, xmax)
            windows_beam: dict[str, object] = {}
            for label, start, end in (("early", .18, .30), ("outer", .30, .46), ("all_return", .18, .70)):
                mask = (tau >= start) & (tau <= end)
                actual = amplitude(mean[mask]); delta = amplitude(mean_delta[mask])
                ids = np.flatnonzero(mask)
                actual["peak_tau_s"] = float(tau[ids[np.argmax(np.linalg.norm(mean[mask], axis=1))]])
                windows_beam[label] = {"tau_interval_s": [start, end], "actual_velocity": actual, "difference_from_baseline": delta}
            beams.append({"receiver_range_m": [65., xmax], "receiver_count": int(np.count_nonzero((xs>=65.)&(xs<=xmax))), "windows": windows_beam})
        records[run.name] = {"label": run.label, "surface_windows": windows, "receiver_gates": receiver_records,
            "causal_changes": changes, "incoming_beams": beams,
            "wall_seconds": number(run.metadata["wall_seconds"]),
            "recording_mode": "all-node fields + surface" if mapping(run.metadata["recorded_counts"]).get("fields", 0) else "surface only",
            "inputs": {"metadata": file_evidence(run.directory/"metadata.json"), "surface": file_evidence(run.directory/"surface.npz")}}
    ranked = sorted(runs[:3], key=lambda run: number(mapping(mapping(mapping(records[run.name])["surface_windows"])["0.3_2"])["l2_samples_m_s"]))
    ranking = [{"rank": rank+1, "name": run.name, "label": run.label,
        "l2_relative_baseline": mapping(mapping(mapping(records[run.name])["surface_windows"])["0.3_2"])["l2_relative_baseline"]} for rank, run in enumerate(ranked)]
    winner = ranked[0]
    guards: dict[str, object] = {}
    for start, end in WINDOWS:
        key = key_for_window(start, end)
        ratio = number(mapping(mapping(mapping(records[winner.name])["surface_windows"])[key])["l2_relative_baseline"])
        guards[key] = {"l2_ratio": ratio, "no_worse_than_baseline": ratio<=1.+1e-8}
    wide = runs[3]
    quarter = runs[2]
    common_profile_delta = quarter.surface.velocity-wide.surface.velocity
    pre_outer = times[:, None] < path_length(xs,175.)[None,:]/VP_M_S-.003
    quarter_match = {"profile_identity": "alpha0*(s/350)^2 == (alpha0/4)*(s/175)^2 on the old175m domain",
        "pre_old175_outer_P_path_difference": amplitude(common_profile_delta[pre_outer]),
        "full_difference": amplitude(common_profile_delta), "margin_s": .003,
        "meaning": "Quarter-strength175 and smooth350 have identical damping coefficients within175m; their difference diagnoses the truncation/continuation beyond that boundary, not exact solution error"}
    return {"runs": records, "ranking_175m": ranking, "selected_name": winner.name,
        "selection": {"primary_objective": "minimum unweighted L2 of Vx/Vy surface samples, inclusive0.3–2.0s, among baseline/q2_half/q2_quarter", "wide350_eligible": False,
            "guards": guards, "decision_scope": "Best of these three coefficients for this one homogeneous central-source test; no global optimization or heterogeneous-model validation"},
        "quarter_vs_wide350": quarter_match, "rayleigh_speed_m_s": speed,
        "method": {"amplitude_metrics": "L2/RMS/peaks of recorded velocity samples; not elastic energy, reflection coefficients or accuracy relative to an infinite domain",
            "receiver_gate": "Times after t0+abs(x-350)/cR+80ms; near the edge this excludes some early returns along with the direct wave",
            "beam": "Mean after t=tau+x/cR over65..175 and65..315; not modal decomposition, direct-wave tails and other paths can remain",
            "pre_boundary_changes": "Difference from baseline is used only to locate coefficient effects in time; its minimization would incorrectly favor the baseline",
            "reference350": "Historical wider-domain result used only as context and a matched-inner-profile control for q2_quarter; not an exact reference",
            "timing": "Candidate surface-only times are not directly comparable to historical runs with all-node output"}}


def style(run: Run) -> tuple[str, str]:
    index = NAMES.index(run.name)
    return COLORS[index], "--" if run.name=="wide350" else "-"


def plot_results(runs: list[Run], result: dict[str, object], output: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    times, xs = runs[0].surface.times, runs[0].surface.coordinates[:,0]
    speed = rayleigh_speed()
    metrics = mapping(result["runs"])
    fig, axes = plt.subplots(1,2,figsize=(15,5.8),layout="constrained",gridspec_kw={"width_ratios":[1.5,1]})
    for run in runs:
        color, ls=style(run)
        rms=np.sqrt(np.mean(run.surface.velocity**2,axis=(1,2)))
        axes[0].semilogy(times,np.maximum(rms*1e9,1e-15),color=color,ls=ls,lw=1.8,label=run.label)
    axes[0].axvline(.3,color="#777777",ls=":",lw=1)
    axes[0].set(xlim=(.2,2),ylim=(1e-6,1),xlabel="Физическое время, с",ylabel="RMS компонент поверхности, нм/с",title="Сохранённая скорость на 70 приёмниках")
    axes[0].legend(fontsize=9);axes[0].grid(alpha=.2,which="both")
    ratios=[number(mapping(mapping(mapping(metrics[run.name])["surface_windows"])["0.3_2"])["l2_relative_baseline"]) for run in runs]
    bars=axes[1].barh(np.arange(4),ratios,color=COLORS)
    bars[3].set_hatch("//")
    for index, ratio in enumerate(ratios):
        axes[1].text(ratio+max(ratios)*.015,index,f"{ratio:.4f}",va="center",fontsize=10)
    axes[1].set(yticks=np.arange(4),yticklabels=["175 м · α₀","175 м · 0,5α₀","175 м · 0,25α₀","350 м · контекст"],xlim=(0,max(ratios)*1.2),xlabel="L2 записи за 0,3–2 с / L2 исходного 175 м",title="Основной критерий: меньше — ниже остаток")
    axes[1].invert_yaxis();axes[1].axvline(1,color="#777777",ls=":",lw=1)
    fig.suptitle("Однородная среда · ослабление квадратичного демпфирования при толщине 175 м",fontsize=14)
    fig.supxlabel("Сравниваются амплитудные показатели поверхностной записи, не энергия или истинная ошибка решения.\nВариант 350 м показан для контекста и не участвует в выборе профиля 175 м.",fontsize=9)
    records.extend(save_figure(fig,output,"surface_tail_rms"))
    fig,axes=plt.subplots(2,3,figsize=(15,10),sharex=True,sharey=True,layout="constrained")
    shown=(times>=.2)&(times<=.8)
    maximum=max(float(np.max(np.abs(run.surface.velocity[shown]))) for run in runs[:3])*1e9
    norm=SymLogNorm(linthresh=.001,vmin=-maximum,vmax=maximum,base=10)
    image=None
    for component in range(2):
        for column,run in enumerate(runs[:3]):
            ax=axes[component,column]
            image=ax.pcolormesh(xs,times,run.surface.velocity[:,:,component]*1e9,cmap="RdBu_r",norm=norm,shading="nearest",rasterized=True)
            ax.set(xlim=(0,700),ylim=(.8,.2),title=f"{'Vx' if component==0 else 'Vy'} · {run.label}")
            if column==0:ax.set_ylabel("Время, с")
            if component==1:ax.set_xlabel("x, м")
    if image is None:raise AssertionError("Missing seismogram image")
    fig.colorbar(image,ax=axes.ravel().tolist(),label="Скорость, нм/с · общая SymLog-шкала; линейно до 0,001",shrink=.8)
    fig.suptitle("Верхняя поверхность · абсолютные Vx/Vy на интервале 0,2–0,8 с\nНеизменная область и источник; одна физическая шкала для всех шести панелей",fontsize=15)
    records.extend(save_figure(fig,output,"surface_seismograms"))
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,layout="constrained")
    trace_mask=(times>=.18)&(times<=.8)
    for row,x in enumerate((175.,345.)):
        index=int(np.flatnonzero(xs==x)[0])
        for component in range(2):
            ax=axes[row,component]
            for run in runs:
                color,ls=style(run)
                ax.plot(times[trace_mask],run.surface.velocity[trace_mask,index,component]*1e9,color=color,ls=ls,lw=1.4,label=run.label)
            ax.set_yscale("symlog",linthresh=1e-4)
            ax.set(xlim=(.18,.8),title=f"{'Vx' if component==0 else 'Vy'} · x={x:g} м",ylabel="Скорость, нм/с")
            ax.grid(alpha=.2,which="both")
            if row==1:ax.set_xlabel("Время, с")
    axes[0,0].legend(fontsize=9)
    fig.suptitle("Фактические поверхностные трассы · без индивидуальной нормировки",fontsize=14)
    records.extend(save_figure(fig,output,"return_traces"))
    fig,axes=plt.subplots(2,2,figsize=(15,9),sharex=True,layout="constrained")
    for column,xmax in enumerate((175.,315.)):
        for run in runs:
            color,ls=style(run)
            tau,mean=beam(times,xs,run.surface.velocity,speed,xmax)
            _,delta=beam(times,xs,run.surface.velocity-runs[0].surface.velocity,speed,xmax)
            axes[0,column].semilogy(tau,np.maximum(np.linalg.norm(mean,axis=1)*1e9,1e-15),color=color,ls=ls,lw=1.5,label=run.label)
            if run.name!="baseline":axes[1,column].semilogy(tau,np.maximum(np.linalg.norm(delta,axis=1)*1e9,1e-15),color=color,ls=ls,lw=1.5,label=run.label+" − исходный175м")
        for row in range(2):
            ax=axes[row,column]
            for padding in (0.,175.):
                prediction=SOURCE_CENTER_S+(350.+2*padding)/speed
                ax.axvline(prediction,color="#777777",ls=":",lw=1)
                ax.text(prediction+.003,.95,f"R{padding:g}",transform=ax.get_xaxis_transform(),va="top",fontsize=9)
            ax.set(xlim=(.18,.7),ylim=(1e-6,1),ylabel="Модуль среднего, нм/с",title=f"x=65…{xmax:g} м · "+("фактические скорости" if row==0 else "изменение относительно исходного 175 м"))
            ax.grid(alpha=.2,which="both")
            if row==1:ax.set_xlabel("τ=t−x/cR, с")
    axes[0,0].legend(fontsize=9);axes[1,0].legend(fontsize=8)
    fig.suptitle("Среднее после выравнивания по входящему поверхностному пути\nДиагностика направления прихода; хвосты прямой волны и другие пути могут сохраняться",fontsize=14)
    records.extend(save_figure(fig,output,"incoming_returns"))
    return records


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=DATA/"profile_sweep_3060_v1/runs")
    parser.add_argument("--out",type=Path,default=DATA/"profile_sweep_analysis_v1")
    parser.add_argument("--images",type=Path,default=IMAGES/"profile_sweep_comparison_v1")
    args=parser.parse_args()
    output:Path=args.out; images:Path=args.images
    output.resolve().relative_to(DATA);images.resolve().relative_to(IMAGES)
    if output.exists() or images.exists():raise FileExistsError("Select fresh analysis outputs")
    paths=(DATA/"homogeneous_3060_v1/padded",args.runs/"q2_half",args.runs/"q2_quarter",DATA/"double_layer_3060_v1/padded350")
    runs=[load_run(name,label,path) for name,label,path in zip(NAMES,LABELS,paths,strict=True)]
    validate_runs(runs)
    result=analyze(runs)
    output.mkdir(parents=True,exist_ok=False);images.mkdir(parents=True,exist_ok=False)
    result.update(status="completed",schema="chapter14.profile_sweep_analysis.v1",utc=datetime.now(timezone.utc).isoformat(),
        code=file_evidence(Path(__file__)),figures=plot_results(runs,result,images),
        profile_plot=file_evidence(DATA/"profile_sweep_plan_v1/profile_plot.json"))
    with (output/"analysis.json").open("x",encoding="utf-8") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write("\n")
    print(json.dumps({"status":"completed","selected_name":result["selected_name"],"ranking":result["ranking_175m"]},ensure_ascii=False))


if __name__=="__main__":
    main()
