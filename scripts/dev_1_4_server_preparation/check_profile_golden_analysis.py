"""CPU-проверки чтения и графиков поиска k; численный решатель не запускается."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import numpy as np

from scripts.dev_1_4_server_preparation import profile_golden_analysis as analysis
from scripts.dev_1_4_server_preparation.profile_golden_run import CONFIG, Request, golden_search
from scripts.dev_1_4_server_preparation.compare_smoke import file_evidence
from scripts.reporting import mapping, number, read_json, sequence


def write_json(path: Path, record: dict[str,object]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2,allow_nan=False)
        stream.write("\n")


def fixture(root: Path) -> tuple[float,float]:
    control_specs=((1.,analysis.DATA/"homogeneous_3060_v1/padded"),
        (.5,analysis.DATA/"profile_sweep_3060_v1/runs/q2_half"),
        (.25,analysis.DATA/"profile_sweep_3060_v1/runs/q2_quarter"))
    controls: list[dict[str,object]]=[]
    for index,(factor,directory) in enumerate(control_specs):
        profile=analysis.load_profile(factor,directory,index-3)
        controls.append({"factor":factor,"directory":str(directory),"objective":analysis.objective(profile.run),
            "metadata":file_evidence(directory/"metadata.json"),"surface":file_evidence(directory/"surface.npz")})
    baseline=analysis.load_profile(1.,control_specs[0][1],-3)
    analysis.validate_profile(baseline,baseline.run)
    hashes=analysis.source_prefix_hashes()
    source={"historical_2s_sha256":hashes["full2s"],"new_1s_sha256":hashes["prefix1s"],
        "target_samples":96001,"prefix_array_equal":True,"full_hashes_equal":False}
    manifest={"config":asdict(CONFIG),"t_end_s":1.,"time_window_s":[.5,1.],"source_prefix":source,"controls":controls}
    write_json(root/"manifest.json",manifest)
    evaluated: list[dict[str,object]]=[]

    def evaluate(request: Request) -> float:
        directory=root/"runs"/request.name
        directory.mkdir(parents=True)
        scale=.1+(request.factor-.6)**2
        surf=baseline.run.surface
        np.savez(directory/"surface.npz",time_s=surf.times,coordinates_m=surf.coordinates,
            node_ids=surf.node_ids,velocity_m_s=surf.velocity*scale)
        metadata=read_json(control_specs[1][1]/"metadata.json")
        spec=dict(mapping(metadata["run_spec"]))
        spec.update(t_end_s=1.,alpha_max_s_inv=analysis.ALPHA0*request.factor)
        runtime=dict(mapping(metadata["runtime"]))
        runtime.update(steps_count=96000,current_time_s=1.)
        profile=dict(mapping(metadata["profile"]))
        profile.update(name=request.name,amplitude_factor=request.factor,
            alpha_peak_per_direction_s_inv=analysis.ALPHA0*request.factor)
        metadata.update(run_spec=spec,runtime=runtime,profile=profile,steps_count=96000,
            recorded_counts={"surface":1001,"fields":0,"source_history":0},
            source_input_sha256=hashes["prefix1s"],source_prefix=source,golden_request=asdict(request),
            surface_file=file_evidence(directory/"surface.npz"))
        write_json(directory/"metadata.json",metadata)
        write_json(directory/"validation.json",{"status":"passed","metadata":file_evidence(directory/"metadata.json")})
        result=analysis.load_profile(request.factor,directory,request.index)
        value=analysis.objective(result.run)
        record={**asdict(request),"directory":str(directory),"objective":value,
            "relative_baseline":value/analysis.objective(baseline.run),
            "metadata":file_evidence(directory/"metadata.json"),"surface":file_evidence(directory/"surface.npz"),
            "validation":file_evidence(directory/"validation.json")}
        write_json(root/"evaluations"/f"{request.name}.json",record)
        evaluated.append(record)
        return value

    search=golden_search(evaluate)
    best=min(evaluated,key=lambda row:number(row["objective"]))
    result={"status":"completed","config":asdict(CONFIG),"t_end_s":1.,"time_window_s":[.5,1.],
        "manifest":file_evidence(root/"manifest.json"),"evaluations":evaluated,"history":search.history,
        "final_interval":list(search.final_interval),"final_width":search.final_interval[1]-search.final_interval[0],
        "tolerance":CONFIG.tolerance,"new_evaluations":len(evaluated),"best_new":best,"best_observed":best,
        "selected_factor":best["factor"]}
    write_json(root/"result.json",result)
    return number(best["factor"]),number(best["relative_baseline"])


def check(output: Path) -> dict[str,object]:
    output.resolve().relative_to(analysis.DATA)
    if output.exists():
        raise FileExistsError("Choose fresh CPU-check output")
    with tempfile.TemporaryDirectory(prefix="tmp_profile_golden_analysis_",dir=analysis.DATA) as temporary:
        root=Path(temporary)
        expected_factor,expected_ratio=fixture(root)
        profiles,history,interval,factor,_=analysis.read_search(root)
        report=analysis.summarize(profiles,history,interval,factor)
        if factor!=expected_factor or not np.isclose(number(report["selected_l2_ratio_baseline"]),expected_ratio,rtol=1e-13,atol=0):
            raise AssertionError("Synthetic minimum was not recovered")
        if not interval[0]<.6<interval[1] or interval[1]-interval[0]>.02:
            raise AssertionError("Synthetic golden interval does not contain its known minimum")
        selected=next(mapping(row) for row in sequence(report["all_evaluations"]) if mapping(row)["factor"]==factor)
        peaks=mapping(mapping(selected["peak_guards"])["late_0.5_1"])
        if len(sequence(peaks["all_70_receivers"]))!=70 or len(sequence(peaks["selected_receivers"]))!=3:
            raise AssertionError("Incomplete receiver diagnostics")
        altered=[replace(history[0],left=float("nan")),*history[1:]]
        try:
            analysis.validate_history(altered,interval)
        except ValueError:
            pass
        else:
            raise AssertionError("Nonfinite history was accepted")
        try:
            analysis.validate_history(history,(.25,1.))
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid final interval was accepted")
        plots=root/"figures";plots.mkdir()
        figures=analysis.plot_results(profiles,history,report,plots)
        if len(figures)!=12 or len(list(plots.glob("*.png")))!=6:
            raise AssertionError("Expected six PNG/SVG pairs")
    result:dict[str,object]={"status":"passed","utc":datetime.now(timezone.utc).isoformat(),
        "scope":"Synthetic adapter/history/plot checks on CPU; no wave solver execution",
        "checks":["Historical2s controls read as exact1001-sample prefixes; reconstructed force hashes match",
            "Complete nine-evaluation synthetic result, ledgers and before/after contraction history reloaded and verified",
            "Known quadratic minimum0.6 remains in final interval; minimum observed sample and objective recovered",
            "Both all70-receiver peak gates and x5/175/345 summaries present",
            "Nonfinite history and wrong final interval rejected","Six PNG/SVG pairs constructed in a temporary directory and removed"],
        "code":file_evidence(Path(analysis.__file__)),"checker":file_evidence(Path(__file__))}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/"validation.json",result)
    return result


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,default=analysis.DATA/"profile_golden_1s_analysis_cpu_v1")
    args=parser.parse_args()
    result=check(args.out)
    print(json.dumps({"status":result["status"],"output":str(args.out)}))


if __name__=="__main__":
    main()
