"""Compare core-ABC and padded source timings; waveform changes are diagnostics."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from scripts.dev_1_4_server_preparation.compare_benchmark import load_run
from scripts.dev_1_4_server_preparation.compare_smoke import file_evidence, metrics
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number

DEFAULT_REFERENCE = PROJECT_ROOT / "data/dev_1_4_server_preparation/benchmark_3060_v1/EFVB_00000/x350"


def compare(core_directory: Path, reference_directory: Path = DEFAULT_REFERENCE) -> dict[str, object]:
    core, reference = load_run(core_directory), load_run(reference_directory)
    core_spec, reference_spec = (mapping(run.metadata["run_spec"]) for run in (core, reference))
    fixed: dict[str, object] = {"dataset": "EFVB", "model_id": 0, "source_x_m": 350., "substeps": 96,
                                "t_end_s": 2., "order": 4, "kind": "curved", "backend": "cu"}
    if any(spec.get(key) != value for spec in (core_spec, reference_spec) for key, value in fixed.items()):
        raise ValueError("Both runs must be full EFVB0/x350/p4/K96 CU sources")
    if core_spec.get("alpha_max_s_inv") != 0. or core_spec.get("padding_m") != 0. or core_spec.get("boundary_mode") != "core_abc" or core_spec.get("mass_damping_enabled") is not False:
        raise ValueError("Core specification does not explicitly disable padding and mass damping")
    if core.metadata.get("boundary_mode") != "core_abc" or core.metadata.get("mass_damping_disabled") is not True or core.metadata.get("physics_ids") != [11000, 32000, 20200] or core.metadata.get("abc_edge_count") != 420:
        raise ValueError("Core metadata does not record the intended three-physics/420-edge configuration")
    core_experiment, reference_experiment = (mapping(run.metadata["experiment_spec"]) for run in (core, reference))
    if core_experiment.get("padding_m") != 0. or reference_experiment.get("padding_m") != 175. or number(reference_spec["alpha_max_s_inv"]) <= 0:
        raise ValueError("Expected core padding=0 versus reference padding=175 m with positive mass damping")
    if {key: value for key, value in core_experiment.items() if key != "padding_m"} != {key: value for key, value in reference_experiment.items() if key != "padding_m"}:
        raise ValueError("Unintended physical-parameter changes beyond padding")
    if core.metadata["source_input_sha256"] != reference.metadata["source_input_sha256"] or core.metadata["initial_state"] != reference.metadata["initial_state"]:
        raise ValueError("Source-force input or initial-state convention differs")
    if mapping(core.metadata["native"])["sha256"] != mapping(reference.metadata["native"])["sha256"]:
        raise ValueError("Timing comparison requires the identical native binary")
    core_code, reference_code = (mapping(run.evidence["code_sha256"]) for run in (core, reference))
    common_code = sorted(core_code.keys() & reference_code.keys())
    if any(core_code[key] != reference_code[key] for key in common_code):
        raise ValueError("Common source/recorder/physics code differs")
    if not np.array_equal(core.surface.times, reference.surface.times) or not np.array_equal(core.surface.coordinates, reference.surface.coordinates):
        raise ValueError("Receiver coordinates and time axes must be exactly equal")
    core_mesh, reference_mesh = (mapping(run.metadata["mesh"]) for run in (core, reference))
    core_nodes, reference_nodes = (int(number(mesh["nodes"])) for mesh in (core_mesh, reference_mesh))
    core_elements, reference_elements = (int(number(mesh["elements"])) for mesh in (core_mesh, reference_mesh))
    if (core_nodes, core_elements, reference_nodes, reference_elements) != (314721, 19600, 589541, 36750):
        raise ValueError("Unexpected physical core/padded mesh sizes")
    core_seconds, reference_seconds = (number(run.metadata["wall_seconds"]) for run in (core, reference))
    difference = metrics(core.surface.velocity, reference.surface.velocity)
    return {"schema": "chapter14.no_padding_comparison.v1", "status": "completed",
            "utc": datetime.now(timezone.utc).isoformat(), "recording_and_parameter_checks_passed": True,
            "comparison_kind": "changed_boundary_physics_diagnostic", "waveform_agreement_required": False,
            "waveform_match_threshold": None, "no_abc_accuracy_or_convergence_claim": True,
            "waveform_interpretation": "Differences are expected because the sponge and exterior material are removed and ABC moves to the physical core boundary. These norms do not measure solver error, ABC accuracy or mesh convergence.",
            "fixed_parameters": fixed, "shape": [2001, 70, 2], "finite": True,
            "checks": {"source_force_sha256_equal": True, "native_sha256_equal": True,
                "common_python_code_sha256_equal": True, "receiver_coordinates_times_exact": True,
                "receiver_node_ids_equal": bool(np.array_equal(core.surface.node_ids, reference.surface.node_ids)),
                "receiver_node_id_note": "Node IDs are local numbering and need not match after cropping"},
            "physics_change": {"core": {"padding_m": 0., "alpha_max_s_inv": 0., "mass_damping": False,
                "physics_ids": [11000, 32000, 20200], "abc_edge_count": 420,
                "boundary": "ABC on x=0, x=700, y=-700; free top y=0"},
                "reference": {"padding_m": 175., "alpha_max_s_inv": reference_spec["alpha_max_s_inv"],
                    "mass_damping": True, "boundary": "ABC on x=-175, x=875, y=-875; free top y=0"}},
            "mesh_counts": {"core_nodes": core_nodes, "reference_nodes": reference_nodes,
                "core_elements": core_elements, "reference_elements": reference_elements,
                "nodes_removed": reference_nodes - core_nodes, "elements_removed": reference_elements - core_elements,
                "core_to_reference_node_ratio": core_nodes / reference_nodes,
                "core_to_reference_element_ratio": core_elements / reference_elements},
            "timing": {"core_source_wall_seconds": core_seconds, "reference_source_wall_seconds": reference_seconds,
                "core_speedup_over_reference": reference_seconds / core_seconds,
                "core_to_reference_wall_ratio": core_seconds / reference_seconds,
                "source_steps_each": 192000, "dt_s_each": .001 / 96,
                "scope": "Source pipeline with native construction/prepare/run, recording and NPZ write/hash; geometry preparation excluded",
                "caveat": "Single runs on the local GPU. Mesh size and mass-damping workload change together; this is not isolated kernel performance, a timing distribution or a pure mesh-scaling law."},
            "metrics": difference,
            "components": {name: metrics(core.surface.velocity[:, :, axis], reference.surface.velocity[:, :, axis]) for axis, name in enumerate(("Vx", "Vy"))},
            "common_python_code": {key: core_code[key] for key in common_code},
            "core_only_python_code": {key: value for key, value in core_code.items() if key not in reference_code},
            "core": core.evidence, "reference": reference.evidence,
            "comparison_code": [file_evidence(Path(__file__)), file_evidence(Path(__file__).with_name("compare_benchmark.py")),
                                file_evidence(Path(__file__).with_name("compare_smoke.py"))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core", type=Path, required=True, help="Completed core x350 directory")
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.out.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if output.exists():
        raise FileExistsError(output)
    result = compare(args.core, args.reference)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "timing": result["timing"], "wave_difference": result["metrics"], "out": str(output)}))


if __name__ == "__main__":
    main()
