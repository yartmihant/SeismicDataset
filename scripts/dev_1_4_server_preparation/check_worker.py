"""CPU integration of the queue worker; mocked solver, real guards/files/verifiers."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

from semgpu2.model.model import SGModel
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.check_run import synthetic_case
from scripts.dev_1_3_point_source.mesh import FrozenMesh
from scripts.dev_1_4_server_preparation import surface_runner as runner
from scripts.dev_1_4_server_preparation import worker
from scripts.dev_1_4_server_preparation.check_surface_runner import MockModel, expect_error
from scripts.dev_1_4_server_preparation.chunk_plan import BudgetUnavailable, DiskGuard, GIB, GuardConfig, build_chunks
from scripts.dev_1_4_server_preparation.curved_mesh import GeometryResult
from scripts.dev_1_4_server_preparation.model_input import ModelInput
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, read_json, sequence

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


def lifecycle(root: Path, fail_source: bool, fail_work_budget: bool = False) -> dict[str, object]:
    root.mkdir()
    config = GuardConfig(working_floor_bytes=8192) if fail_work_budget else GuardConfig()
    guard = DiskGuard(root / "queue.sqlite", root, config, available_bytes=lambda: 100 * GIB)
    job = guard.reserve_next(0)
    assert job is not None
    case = synthetic_case(4)
    case.metadata.update({"model_id": 0, "dataset": "EFVB", "kind": "curved"})
    model = ModelInput("EFVB", 0, np.full((70, 70), 2000.), np.full((70, 70), 1000.), {"fixture": "CPU mock only"})
    tiny = FrozenMesh(np.zeros((9, 2)), np.arange(9, dtype=np.int64).reshape((1, 9)),
                      np.array([2000.]), np.array([1000.]), np.array([0], dtype=np.int64), 0, "curved")
    report: dict[str, object] = {"schema": "CPU_MOCK_ONLY", "dataset": "EFVB", "model_id": 0, "input": model.provenance,
        "construction": {"construction": "mock", "settings": {"mock": True},
                         "splines": [{"max_sample_depth_error_m": 0.125}]},
        "certificate": {"scope": "not a scientific mesh"}, "fallback": "none", "algorithm_evidence": "CPU mock"}
    geometry = GeometryResult(tiny, report)
    native = root / "mock_native.txt"
    native.write_text("CPU mock only; no native execution\n")

    def estimate(path: Path, order: int, alpha: float) -> dict[str, object]:
        with np.load(path) as arrays:
            assert str(arrays["dataset"]) == "EFVB"
        return {"mesh": str(path), "mesh_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "order": order, "nodes": len(case.mesh.nodes_coords), "elements": len(case.element_vp),
                "alpha_max_s_inv": alpha, "dt_start_s": .001 / 15, "substeps_per_ms": 15}

    def factory(case_value: MeshCase, spec: runner.SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        assert case_value is case and worker.progress_counts(guard.status()) == (1, 0)
        return cast("SGModel[CUConfig]", MockModel(spec, fail_source and spec.source_x_m == 350.)), native, 0

    with ExitStack() as stack:
        stack.enter_context(patch.object(worker, "read_model", return_value=model))
        stack.enter_context(patch.object(worker, "build_curved", return_value=geometry))
        stack.enter_context(patch.object(worker, "build_sem", return_value=case))
        stack.enter_context(patch.object(worker, "estimate", side_effect=estimate))
        stack.enter_context(patch.object(runner, "make_cu_model", side_effect=factory))
        if fail_work_budget:
            expect_error(lambda: worker.run_reserved_model(root, root, guard, job), BudgetUnavailable)
        elif fail_source:
            expect_error(lambda: worker.run_reserved_model(root, root, guard, job), ValueError)
        else:
            worker.run_reserved_model(root, root, guard, job)
    work = root / "work/EFVB/00000"
    output = root / "chunks" / build_chunks()[0].name / "models/00000"
    if fail_work_budget:
        assert work.is_dir() and not (work / "mesh_q2.npz").exists()
        assert worker.progress_counts(guard.status()) == (1, 0)
        guard.fail(job.job_id, "intentional CPU test working budget rejection")
        return {"oversized_work_rejected_before_mesh_write": True}
    if fail_source:
        assert work.is_dir() and (work / "mesh_q2.npz").is_file()
        assert not (output / "model.json").exists()
        assert read_json(output / "x350/metadata.failed.json")["status"] == "failed"
        for source in (0., 175.):
            runner.verify_surface(output / f"x{int(source):03d}", runner.SurfaceSpec("EFVB", 0, source, 24))
        guard.fail(job.job_id, "intentional CPU test missing final source frame")
        assert worker.progress_counts(guard.status()) == (0, 1)
        expect_error(lambda: guard.reserve_next(1), BudgetUnavailable)
        return {"failure_keeps_work_and_reservation": True, "queue_blocks_after_failure": True}
    assert not work.exists() and worker.progress_counts(guard.status()) == (0, 0)
    for source in (0., 175., 350., 525., 700.):
        metadata = runner.verify_surface(output / f"x{int(source):03d}", runner.SurfaceSpec("EFVB", 0, source, 24))
        assert mapping(metadata["recorded_counts"])["surface"] == 2001
    summary = read_json(output / "model.json")
    assert len(sequence(summary["sources"])) == 5
    implementation = mapping(summary["implementation"])
    assert len(str(implementation["project_git_head"])) == 40
    assert "scripts/dev_1_4_server_preparation/worker.py" in mapping(implementation["code_sha256"])
    compact = read_json(output / "geometry.json")
    assert mapping(compact["construction"])["splines_max_sample_depth_error_m"] == .125
    retained = worker.directory_bytes(output)
    assert retained <= guard.config.model_budget_bytes
    expect_error(lambda: worker.run_reserved_model(root, root, guard, job), FileExistsError)
    metadata_bytes = sum(path.stat().st_size for path in output.rglob("*.json"))
    assert metadata_bytes < 256 * 1024
    return {"sources": 5, "times_per_source": 2001, "temporary_work_removed_only_after_verified_outputs": True,
            "allocated_model_bytes": retained, "model_budget_bytes": guard.config.model_budget_bytes,
            "all_json_metadata_bytes": metadata_bytes, "metadata_allowance_bytes": 256 * 1024}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root: Path = args.out.resolve()
    root.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    root.mkdir(parents=True, exist_ok=False)
    for minimum in range(1, 10000):
        actual = worker.select_substeps("EFVB", 0, minimum)
        assert actual % 8 == 0 and actual >= 1.6 * minimum and actual - 8 < 1.6 * minimum
    assert worker.select_substeps("ECVB", 12404, 40) == 64
    assert worker.select_substeps("ECVB", 65, 130) == 224
    expect_error(lambda: worker.select_substeps("ECVB", 65, 225), ValueError)
    expect_error(lambda: worker.select_substeps("EFVB", 0, 0), ValueError)
    assert worker.progress_counts({"progress": [{"status": "reserved", "models": 1}, {"status": "completed", "models": 3}]}) == (1, 0)
    passed = lifecycle(root / "completed_cpu_mock", False)
    failed = lifecycle(root / "failed_cpu_mock", True)
    work_budget = lifecycle(root / "working_budget_mock", False, True)
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
              "scope": "CPU mocked geometry/stability/solver; real five-source surface recorder/verifier, SQLite guard and filesystem lifecycle",
              "completed": passed, "failure": failed, "working_budget": work_budget,
              "sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("worker.py", "check_worker.py", "surface_runner.py", "chunk_plan.py")}}
    (root / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
