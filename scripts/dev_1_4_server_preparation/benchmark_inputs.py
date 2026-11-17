"""CPU-only reconstruction of the exact source input for cross-host auditing."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from typing import Literal, cast

import numpy as np

from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
from scripts.dev_1_3_point_source.physics import ExperimentSpec, nodal_point_force, ricker
from scripts.dev_1_4_server_preparation.compare_smoke import file_evidence
from scripts.project_paths import PROJECT_ROOT


@dataclass(frozen=True)
class InputSpec:
    dataset: Literal["EFVB", "ECVB"]
    model_id: int
    substeps: int
    source_x_m: float = 350.0
    t_end_s: float = 2.0
    order: int = 4

    def __post_init__(self) -> None:
        if self.dataset not in ("EFVB", "ECVB") or not 0 <= self.model_id < 30000 or self.substeps < 1:
            raise ValueError("Invalid dataset/model/substeps")
        if self.source_x_m not in ExperimentSpec().source_x_m or self.t_end_s != 2.0 or self.order != 4:
            raise ValueError("Expected one of five sources, p4 and two seconds")


def reconstruct(mesh_path: Path, spec: InputSpec, output: Path) -> dict[str, object]:
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if output.suffix != ".npz" or output.exists() or output.with_suffix(".json").exists():
        raise ValueError("Use a new .npz path and a new companion .json")
    frozen = load_frozen(mesh_path)
    with np.load(mesh_path, allow_pickle=False) as archive:
        if str(archive["dataset"]) != spec.dataset:
            raise ValueError("Frozen mesh dataset differs")
    if frozen.model_id != spec.model_id or frozen.kind != "curved":
        raise ValueError("Frozen mesh model/kind differs")
    case = build_sem(frozen, spec.order)
    source = nodal_point_force(case.mesh.nodes_coords, spec.source_x_m, ExperimentSpec().force_amplitude_n_per_m)
    times = np.linspace(0., spec.t_end_s, 2000 * spec.substeps + 1)
    values = np.zeros((times.size, 1, 2), dtype=np.float64)
    values[:, 0, 1] = source.force_y * ricker(times)
    solver_native = [name for name in sys.modules if name.startswith("semgpu2.solvers.") and name.endswith("._native")]
    if solver_native:
        raise RuntimeError("CPU input audit unexpectedly imported solver native code")
    probe = output.parent
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < 20 * 1024**3 + times.nbytes + values.nbytes + 65536:
        raise OSError("Input audit must preserve 20 GiB of free space")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        np.savez_compressed(stream, source_times_s=times, source_values=values)
        stream.flush()
        os.fsync(stream.fileno())
    record: dict[str, object] = {"schema": "chapter14.benchmark_inputs.v1", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU reconstruction only; no solver/native execution", "solver_native_imported": False,
        "spec": asdict(spec), "experiment_spec": asdict(ExperimentSpec()), "mesh": file_evidence(mesh_path),
        "source_node_id": source.node_id, "nodes_sha256": hashlib.sha256(case.mesh.nodes_coords.tobytes()).hexdigest(),
        "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": hashlib.sha256(array.tobytes()).hexdigest()}
                   for name, array in (("source_times_s", times), ("source_values", values))},
        "output": file_evidence(output), "provenance": {"hostname": platform.node(), "platform": platform.platform(),
            "python": sys.version, "packages": {name: version(name) for name in ("numpy", "scipy", "semgpu2")},
            "environment": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "files": [file_evidence(Path(__file__)), file_evidence(PROJECT_ROOT / "scripts/dev_1_3_point_source/physics.py"),
                      file_evidence(PROJECT_ROOT / "scripts/dev_1_3_point_source/mesh.py")]}}
    with output.with_suffix(".json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--dataset", choices=("EFVB", "ECVB"), required=True)
    parser.add_argument("--model", type=int, required=True)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--source", type=float, default=350.)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = reconstruct(args.mesh, InputSpec(cast(Literal["EFVB", "ECVB"], args.dataset), args.model, args.substeps, args.source), args.out)
    print(json.dumps({"out": str(args.out), "arrays": record["arrays"], "nodes_sha256": record["nodes_sha256"]}))


if __name__ == "__main__":
    main()
