"""CPU mock-проверки записи и верификации Vx/Vy70; native/GPU не запускаются."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast
from unittest.mock import patch
import zipfile

import numpy as np

from semgpu2.model.model import SGModel
from semgpu2.model.physics import SGPhysicsModel
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.check_run import synthetic_case
from scripts.dev_1_3_point_source.physics import ricker
from scripts.dev_1_4_server_preparation import production_surface as runner
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import read_json

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


@dataclass(frozen=True)
class Event(SGBaseReceiverEventData):
    current_time: float


@dataclass
class Result:
    code: int
    steps_count: int
    current_time: float


class Control(SGReceiverRunControl):
    def abort(self, code: int) -> None:
        raise AssertionError("Unexpected mock abort")

    def complete(self, code: int) -> None:
        raise AssertionError("Unexpected mock complete")

    def save(self) -> None:
        raise AssertionError("Unexpected mock save")


class MockModel:
    def __init__(self, spec: runner.SurfaceSpec, missing_last: bool = False) -> None:
        self.spec = spec
        self.physics: list[SGPhysicsModel] = []
        self.receivers: list[SGReceiverModel] = []
        self.missing_last = missing_last

    def add_physics(self, physics: SGPhysicsModel) -> None:
        self.physics.append(physics)

    def add_receiver(self, receiver: SGReceiverModel) -> None:
        self.receivers.append(receiver)

    def prepare(self) -> None:
        assert [p.type_code for p in self.physics] == [11000, 32000, 30001, 20200]
        source = self.physics[-1]
        times = np.linspace(0.0, self.spec.t_end_s, self.spec.steps + 1)
        np.testing.assert_array_equal(source.values[:, 0, 0], 0)
        np.testing.assert_array_equal(source.values[:, 0, 1], -ricker(times))
        assert source.entities.size == 1
        assert len(self.receivers) == 1
        receiver = self.receivers[0]
        np.testing.assert_array_equal(receiver.comp_codes, [4, 5])
        assert receiver.sens_count == 70 and receiver.stride == self.spec.substeps
        assert receiver.begin_tact == 0 and receiver.end_tact == self.spec.steps

    def run(self, sync: bool) -> Result:
        assert sync
        receiver = self.receivers[0]
        times = runner.inclusive_times(self.spec.t_end_s, .001)
        if self.missing_last:
            times = times[:-1]
        for index, current in enumerate(times):
            data = np.vstack((np.full(70, current), np.full(70, -2 * current)))
            receiver.handler(data.ravel(), Event(0, index * self.spec.substeps, float(current)), Control())
        return Result(0, self.spec.steps, self.spec.t_end_s)


def expect_error(action: Callable[[], object], kind: type[Exception]) -> None:
    try:
        action()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def checks(root: Path) -> list[str]:
    passed: list[str] = []
    case = synthetic_case(4)
    case.metadata.update({"model_id": 0, "dataset": "EFVB", "kind": "curved"})
    default_spec = runner.SurfaceSpec("EFVB", 0, 350.0, 24)
    assert default_spec.kim_beta == 1. / 27. and default_spec.kim_gamma == .5
    assert default_spec.t_end_s == 1. and len(runner.SurfaceRecorder.create(default_spec).times) == 1001
    assert np.isclose(default_spec.alpha_max_s_inv,196.76847003046814,rtol=1e-14)
    models: list[MockModel] = []
    spec = runner.SurfaceSpec("EFVB", 0, 350.0, 24, .003)
    assert spec.steps == 72 and spec.dt_s == .001 / 24
    assert runner.suggested_substeps({"substeps_per_ms": 15}) == 24
    assert runner.suggested_substeps({"substeps_per_ms": 16}) == 32
    ids = runner.receiver_nodes(case)
    np.testing.assert_allclose(case.mesh.nodes_coords[ids], np.column_stack((np.arange(5, 700, 10), np.zeros(70))), atol=1e-10)
    mesh = root / "mock_mesh.dat"
    mesh.write_bytes(b"CPU mock mesh identity, not a production NPZ")
    stability = root / "mock_stability.json"
    valid: dict[str, object] = {"mesh": "/old/location/is/portable.npz", "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
        "dataset": "EFVB", "order": 4, "nodes": len(case.mesh.nodes_coords), "elements": len(case.element_vp),
        "alpha_max_s_inv": runner.ALPHA_MAX_S_INV, "dt_start_s": .001 / 15, "substeps_per_ms": 15}
    stability.write_text(json.dumps(valid))
    runner.validate_stability(case, spec, mesh, stability)
    for key, wrong in (("mesh_sha256", "0" * 64), ("order", 5), ("dataset", "ECVB"),
                       ("alpha_max_s_inv", 1.0), ("dt_start_s", 1e-8), ("substeps_per_ms", 30)):
        stability.write_text(json.dumps({**valid, key: wrong}))
        expect_error(lambda: runner.validate_stability(case, spec, mesh, stability), ValueError)
    stability.write_text(json.dumps(valid))
    passed.append("receiver coordinates and K policy exact; mesh SHA/group/order/alpha/dt/K mismatches rejected")

    recorder = runner.SurfaceRecorder.create(spec)
    expect_error(recorder.finish, ValueError)
    expect_error(lambda: recorder.append(np.zeros(140), Event(0, 0, .001)), ValueError)
    expect_error(lambda: recorder.append(np.full(140, np.nan), Event(0, 0, 0)), FloatingPointError)
    for index, current in enumerate(recorder.times):
        recorder.append(np.zeros(140), Event(0, index * spec.substeps, float(current)))
    recorder.finish()
    expect_error(lambda: recorder.append(np.zeros(140), Event(0, spec.steps, spec.t_end_s)), ValueError)
    passed.append("recorder rejects wrong time, nonfinite values, missing and extra frames")

    native = root / "mock_native.txt"
    native.write_text("CPU MOCK ONLY; no native solver loaded\n")
    writes = [0]

    def before_write() -> None:
        writes[0] += 1

    def factory(case_value: MeshCase, spec_value: runner.SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        assert case_value is case
        model = MockModel(spec_value)
        models.append(model)
        return cast("SGModel[CUConfig]", model), native, 0

    output = root / "completed_mock"
    with patch.object(runner, "make_cu_model", side_effect=factory):
        metadata = runner.run_case(case, spec, output, mesh, stability, before_write)
    assert metadata["status"] == "completed" and writes[0] >= 4
    np.testing.assert_array_equal(models[0].physics[2].values[:,0],runner.damping_alpha(case.mesh.nodes_coords,runner.ALPHA_MAX_S_INV))
    with np.load(output / "surface.npz") as arrays:
        assert arrays["velocity_m_s"].shape == (4, 70, 2)
        np.testing.assert_allclose(arrays["velocity_m_s"][-1, :, 0], .003)
        np.testing.assert_allclose(arrays["velocity_m_s"][-1, :, 1], -.006)
    with zipfile.ZipFile(output / "surface.npz") as archive:
        assert all(entry.compress_type == zipfile.ZIP_STORED for entry in archive.infolist())
    assert {p.name for p in output.iterdir()} == {"metadata.started.json", "metadata.json", "surface.npz"}
    runner.verify_surface(output, spec)
    expect_error(lambda: runner.run_case(case, spec, output, mesh, stability, before_write), FileExistsError)
    expect_error(lambda: runner.verify_surface(output, runner.SurfaceSpec("EFVB", 1, 350.0, 24, .003)), ValueError)
    with (output / "surface.npz").open("ab") as stream:
        stream.write(b"tampered")
    expect_error(lambda: runner.verify_surface(output, spec), ValueError)
    # This fixture is intentionally corrupted; scientific output directories are never used.
    passed.append("mock run registers only [4,5], preserves Ricker exactly, writes uncompressed velocity-only NPZ and rejects overwritten/tampered outputs")

    failed = root / "missing_frame_mock"

    def missing(case_value: MeshCase, spec_value: runner.SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        return cast("SGModel[CUConfig]", MockModel(spec_value, True)), native, 0

    with patch.object(runner, "make_cu_model", side_effect=missing):
        expect_error(lambda: runner.run_case(case, spec, failed, mesh, stability, before_write), ValueError)
    assert read_json(failed / "metadata.failed.json")["status"] == "failed"
    assert not (failed / "metadata.json").exists() and not (failed / "surface.npz").exists()
    passed.append("missing final frame yields failed metadata and never a completed surface file")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root: Path = args.out.resolve()
    root.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    root.mkdir(parents=True, exist_ok=False)
    passed = checks(root)
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(), "checks": passed,
              "scope": "CPU mocks only, no solver/native/GPU; intentionally corrupted fixture tests checksum rejection",
              "sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("production_surface.py", "check_production_surface.py")}}
    (root / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
