"""CPU-проверка регистрации полей и строгого сравнения повторной сейсмограммы."""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
from semgpu2.model.model import SGModel

from scripts.dev_1_4_server_preparation import profile_golden_fields_run as runner
from scripts.dev_1_4_server_preparation.check_profile_sweep import ProfileMock
from scripts.dev_1_4_server_preparation.check_surface_runner import expect_error
from scripts.dev_1_4_server_preparation.surface_runner import SurfaceSpec
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


class FieldsMock(ProfileMock):
    def prepare(self) -> None:
        assert [item.type_code for item in self.physics] == [11000, 32000, 30001, 20200]
        coords = np.asarray(self.case.mesh.nodes_coords, dtype=np.float64)
        np.testing.assert_array_equal(self.physics[2].values.ravel(), runner.sweep.profile_alpha(coords, self.profile))
        assert self.profile.amplitude_factor == runner.FACTOR
        assert len(self.receivers) == 2
        for index, receiver in enumerate(self.receivers):
            np.testing.assert_array_equal(receiver.comp_codes, [4, 5])
            assert receiver.stride == (96 if index == 0 else 960)
            assert receiver.sens_count == (70 if index == 0 else len(coords))


def checks() -> None:
    output = runner.ROOT / "cpu_checks_v1"
    output.mkdir(parents=True, exist_ok=False)
    request, _, reference, _ = runner.selected()
    profile = runner.sweep.Profile(request.name, "power", 2., runner.FACTOR)
    spec = runner.sweep.spec_for(profile, .01)
    case, edges = runner.sweep.homogeneous.load_case("padded", spec)
    native = output / "mock_native.txt"
    native.write_text("CPU MOCK ONLY; no native/GPU run\n")

    def factory(mesh: MeshCase, specification: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        model = FieldsMock(mesh, specification, profile, True)
        return cast("SGModel[CUConfig]", model), native, 0

    mock = output / "two_frame_mock"
    mock.mkdir()
    with patch.object(runner.sweep, "make_cu_model", side_effect=factory):
        result = runner.sweep.execute(case, spec, mock, {"scope": "CPU MOCK ONLY"}, edges, profile, True)
    assert result["recorded_counts"] == {"surface": 11, "fields": 2, "source_history": 0}
    assert result["all_node_surface_values_exact"] is True
    assert np.load(mock / "velocity_m_s.npy", mmap_mode="r").shape == (2, 589541, 2)
    repeated = output / "surface_equality"
    repeated.mkdir()
    shutil.copyfile(reference / "surface.npz", repeated / "surface.npz")
    assert runner.compare_surfaces(repeated, reference)["exact_values"] is True
    with np.load(repeated / "surface.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    arrays["velocity_m_s"][700, 10, 0] += 1e-18
    np.savez(repeated / "surface.npz", **arrays)
    expect_error(lambda: runner.compare_surfaces(repeated, reference), ValueError)
    assert not any(name.endswith("._native") and "lin_elast_2d" in name for name in sys.modules)
    runner.golden.exclusive_json(output / "validation.json", {"status": "passed", "utc": runner.golden.now(),
        "scope": "CPU mocks only; no native/GPU calculation", "factor": runner.FACTOR,
        "checks": ["Selected result/source-prefix/native/mesh identity validated against immutable golden evidence",
            "Actual selected alpha and both receiver cadences/component orders checked on historical589541-node mesh",
            "Nonuniform mock all-node values match surface at common times; complete two-frame recording",
            "Exact saved surface copy accepted; single changed velocity sample rejected"],
        "code": [runner.file_record(Path(__file__)), runner.file_record(Path(runner.__file__))]})
    print("Golden full-field CPU checks passed; no native/GPU loaded", flush=True)


if __name__ == "__main__":
    checks()
