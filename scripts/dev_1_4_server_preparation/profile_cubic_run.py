"""Первые две точки кубического слоя175м, 1с, перед поиском золотым сечением."""
from __future__ import annotations
from dataclasses import asdict
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from time import perf_counter
import traceback
import numpy as np
from scripts.dev_1_3_point_source.physics import ExperimentSpec
from scripts.dev_1_3_point_source.run import file_record
from scripts.dev_1_4_server_preparation import profile_golden_run as golden
from scripts.dev_1_4_server_preparation import profile_sweep_run as sweep
from scripts.dev_1_4_server_preparation.surface_runner import _check_write, _write_metadata
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, read_json
ROOT = PROJECT_ROOT / 'data/dev_1_4_server_preparation/profile_cubic_initial_1s_v1'
T_END = 1.
INITIAL_STATE = golden.INITIAL_STATE
Request = golden.Request
exclusive_json = golden.exclusive_json

def code_records() -> list[dict[str, object]]:
    return [file_record(Path(__file__)), *golden.code_records()]

def verify_run(output: Path, request: Request, manifest: dict[str, object]) -> dict[str, object]:
    metadata = sweep.verify(output, sweep.Profile(request.name, 'power', 3., request.factor), 1.)
    target = read_json(golden.ROOT / 'runs/eval_09/metadata.json')
    for key in ('nodes_sha256', 'source_input_sha256', 'initial_state', 'dt_s', 'steps_count', 'physical_model', 'abc_edge_count', 'physics_ids', 'experiment_spec'):
        if metadata[key] != target[key]:
            raise ValueError(f'Changed identity: {key}')
    for key in ('native', 'mesh_evidence', 'geometry_evidence'):
        if mapping(metadata[key])['sha256'] != mapping(target[key])['sha256']:
            raise ValueError(f'Changed evidence: {key}')
    if metadata['recorded_counts'] != {'surface':1001,'fields':0,'source_history':0}:
        raise ValueError('Wrong recording count')
    return metadata

def execute_run(request: Request, manifest: dict[str, object], output: Path) -> dict[str, object]:
    """New output wrapper; frozen sweep functions and its ROOT remain unchanged."""
    output.resolve().relative_to(ROOT / "runs")
    if output.exists():
        raise FileExistsError(output)
    profile = sweep.Profile(request.name, "power", 3., request.factor)
    spec = sweep.spec_for(profile, T_END)
    case, edges = sweep.homogeneous.load_case("padded", spec)
    mesh, geometry, certificate = sweep.homogeneous.mesh_inputs("padded")
    stability = sweep.profile_stability(case, profile)
    _check_write(output.parent, 10_000_000, None)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {"status": "started", "variant": "padded", "padding_m": 175.,
        "profile": sweep.profile_record(profile), "material_model": sweep.homogeneous.MATERIAL_MODEL,
        "physical_model": sweep.homogeneous.PHYSICAL_MODEL, "is_original_dataset_model": False,
        "run_spec": asdict(spec), "experiment_spec": asdict(ExperimentSpec(end_s=T_END)), "golden_request": asdict(request),
        "source_prefix": manifest["source_prefix"], "boundary_mode": "outer_abc", "mass_damping_disabled": False,
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata, "mesh_evidence": file_record(mesh),
        "geometry_evidence": file_record(geometry), "reference_stability_evidence": file_record(certificate),
        "profile_stability": stability, "short_test": False, "components": ["Vx", "Vy"], "receiver_count": 70,
        "save_fields": False, "field_components": [], "field_cadence_s": None, "field_shape": None,
        "field_node_order": None, "initial_state": INITIAL_STATE,
        "timing_note": "Golden search:1s surface-only; no all-node output",
        "provenance": {"python": sys.executable, "numpy": np.__version__, "semgpu2": version("semgpu2"), "files": code_records()}}
    try:
        metadata.update(sweep.execute(case, spec, output, metadata, edges, profile, False))
        metadata.update(status="completed", wall_seconds=perf_counter()-started)
        _write_metadata(output / "metadata.json", metadata, None)
        verified = verify_run(output, request, manifest)
        exclusive_json(output / "validation.json", {"status": "passed", "metadata": file_record(output / "metadata.json"),
            "source_prefix_verified": True, "recorded_counts": metadata["recorded_counts"],
            "scope": "Complete finite1s output and exact physical/source-prefix identity; not independent solver validation"})
        return verified
    except BaseException as error:
        metadata.update(status="failed", wall_seconds=perf_counter()-started, error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        _write_metadata(output / "metadata.failed.json", metadata, None)
        raise

def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=False)
    manifest = golden.verify_manifest()
    exclusive_json(ROOT/'plan.json', {'factors':[.5,1.], 'exponent':3, 'duration_s':1., 'objective_window_s':[.5,1.], 'alpha_reference_s_inv':sweep.BASELINE.peak, 'code':code_records(), 'golden_search_status':'pending_initial_comparison'})
    baseline=golden.surface_objective(golden.ROOT/'runs/eval_09',1001)
    records:list[dict[str,object]]=[]
    for index,factor in enumerate((.5,1.),1):
        request=Request(index,factor,.5,1.,'cubic_initial')
        directory=ROOT/'runs'/request.name
        metadata=execute_run(request,manifest,directory)
        value=golden.surface_objective(directory,1001)
        record={'factor':factor,'alpha_max_s_inv':factor*sweep.BASELINE.peak,'objective':value,'ratio_best_quadratic':value/baseline,'change_percent':100*(value/baseline-1), 'wall_seconds':metadata['wall_seconds'], 'surface':file_record(directory/'surface.npz')}
        records.append(record)
        exclusive_json(ROOT/f'evaluation_{index}.json',record)
        print(json.dumps(record),flush=True)
    exclusive_json(ROOT/'comparison.json',{'status':'completed','window_s':[.5,1.], 'quadratic_reference':file_record(golden.ROOT/'result.json'), 'quadratic_objective':baseline,'evaluations':records,'search_status':'initial_two_points_only'})

if __name__=='__main__':
    main()
