"""CPU-only tests of velocity field streaming, dual receiver layouts and failures."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np

from semgpu2.model.model import SGModel
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_4_server_preparation import velocity_fields_run as runner
from scripts.dev_1_4_server_preparation.check_surface_runner import Control, Event, MockModel, Result, expect_error
from scripts.dev_1_4_server_preparation.surface_runner import SurfaceSpec
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, read_json

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


class DualMock(MockModel):
    def __init__(self, case: MeshCase, spec: SurfaceSpec, missing: bool = False) -> None:
        super().__init__(spec)
        self.case = case
        self.missing = missing

    def prepare(self) -> None:
        expected = [11000, 32000, 20200] if isinstance(self.spec, runner.CoreSpec) else [11000, 32000, 30001, 20200]
        assert [physics.type_code for physics in self.physics] == expected
        assert len(self.receivers) == 2
        surface, fields = self.receivers
        np.testing.assert_array_equal(surface.comp_codes, [4, 5])
        np.testing.assert_array_equal(fields.comp_codes, [4, 5])
        assert surface.sens_count == 70 and surface.stride == 96
        assert fields.sens_count == len(self.case.mesh.nodes_coords) and fields.stride == 960
        assert surface.end_tact == fields.end_tact == self.spec.steps
        times = np.linspace(0., self.spec.t_end_s, self.spec.steps + 1)
        np.testing.assert_array_equal(self.physics[-1].values[:, 0, 1], -runner.ricker(times))

    def run(self, sync: bool) -> Result:
        assert sync
        for current in runner.inclusive_times(self.spec.t_end_s, .001):
            for receiver in self.receivers:
                step = round(current / self.spec.dt_s)
                if step % receiver.stride or (self.missing and receiver.sens_count != 70 and step == self.spec.steps):
                    continue
                frame = np.vstack((np.full(receiver.sens_count, current), np.full(receiver.sens_count, -2 * current)))
                receiver.handler(frame.ravel(), Event(0, step, float(current)), Control())
        return Result(0, self.spec.steps, self.spec.t_end_s)


def checks(output: Path) -> dict[str, object]:
    output.resolve().relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    output.mkdir(parents=True, exist_ok=False)
    native = output / "mock_native.txt"
    native.write_text("CPU MOCK, no solver or native loaded\n")

    def factory(case: MeshCase, spec: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        return cast("SGModel[CUConfig]", DualMock(case, spec)), native, 0

    for variant in ("padded", "core"):
        with patch.object(runner, "make_cu_model", side_effect=factory):
            result = runner.run(cast(runner.Variant, variant), output / variant, .01)
        assert result["status"] == "completed"
        assert result["recorded_counts"] == {"surface": 11, "fields": 2, "source_history": 0}
        fields = np.load(output / variant / "velocity_m_s.npy", mmap_mode="r")
        np.testing.assert_array_equal(fields[-1, :, 0], .01)
        np.testing.assert_array_equal(fields[-1, :, 1], -.02)
        assert mapping(result["field_files"])["velocity_m_s.npy"] == runner.stream_record(output / variant / "velocity_m_s.npy")

    small = output / "recorder"
    small.mkdir()
    recorder = runner.FieldRecorder.create(small, runner.SurfaceSpec("EFVB", 0, 350., 96, .01), 3)
    expect_error(recorder.finish, ValueError)
    expect_error(lambda: recorder.append(np.full(6, np.nan), Event(0, 0, 0.)), FloatingPointError)
    expect_error(lambda: recorder.append(np.zeros(6), Event(0, 0, .001)), ValueError)
    expect_error(lambda: recorder.append(np.ones(6), Event(0, 0, 0.)), ValueError)
    recorder.append(np.zeros(6), Event(0, 0, 0.))
    recorder.append(np.array([1., 2., 3., 4., 5., 6.]), Event(0, 960, .01))
    np.testing.assert_array_equal(recorder.values[1], [[1., 4.], [2., 5.], [3., 6.]])
    expect_error(lambda: recorder.append(np.zeros(6), Event(0, 1920, .02)), ValueError)
    recorder.finish()

    def missing(case: MeshCase, spec: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        return cast("SGModel[CUConfig]", DualMock(case, spec, True)), native, 0

    failed = output / "missing_frame"
    with patch.object(runner, "make_cu_model", side_effect=missing):
        expect_error(lambda: runner.run("core", failed, .01), ValueError)
    assert read_json(failed / "metadata.failed.json")["status"] == "failed"
    assert not (failed / "metadata.json").exists() and (failed / "velocity_m_s.partial.npy").exists()
    assert not any(name.endswith("._native") and "lin_elast_2d" in name for name in sys.modules)
    return {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU mocks only; actual EFVB geometry and receiver/physics construction, no native/GPU solution",
        "checks": ["Both physics variants preserve their source and ABC/mass-damping selection",
            "Two receivers use component-major [4,5], strides96/960 and all global node IDs",
            "Fields stream to float64 NPY and match surface values at common nodes/times",
            "Zero initial frame, missing/extra frames, wrong times and nonfinite values checked",
            "Component/node transpose checked using nonuniform node data",
            "Incomplete fields retain partial output and failed metadata, never completed"],
        "files": [runner.stream_record(Path(__file__)), runner.stream_record(Path(runner.__file__))]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = checks(args.out)
    with (args.out / "validation.json").open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
