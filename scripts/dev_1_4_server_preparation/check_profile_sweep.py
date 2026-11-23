"""CPU mock и итоговая проверка двух амплитуд квадратичного слоя175м."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
from semgpu2.model.model import SGModel

from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.physics import damping_alpha, inclusive_times, ricker
from scripts.dev_1_4_server_preparation import profile_sweep_run as runner
from scripts.dev_1_4_server_preparation.check_surface_runner import Control, Event, MockModel, Result, expect_error
from scripts.dev_1_4_server_preparation.surface_runner import SurfaceSpec, receiver_nodes
from scripts.reporting import mapping, read_json

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


class ProfileMock(MockModel):
    def __init__(self, case: MeshCase, spec: SurfaceSpec, profile: runner.Profile,
                 save_fields: bool, missing_last: bool = False) -> None:
        super().__init__(spec, missing_last)
        self.case, self.profile, self.save_fields = case, profile, save_fields

    def prepare(self) -> None:
        assert [p.type_code for p in self.physics] == [11000, 32000, 30001, 20200]
        rho, vp, vs = 1000., 4087., 2431.62036
        mu, lam = rho*vs**2, rho*(vp**2-2*vs**2)
        expected = np.array([rho, mu*(3*lam+2*mu)/(lam+mu), lam/(2*(lam+mu))])
        actual = self.physics[0].values.reshape(-1, 3)
        np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=2e-15, atol=0.)
        impedance = self.physics[1].values.reshape(-1, 2)
        np.testing.assert_array_equal(impedance, np.broadcast_to([rho*vp, rho*vs], impedance.shape))
        assert self.physics[1].entities.size == 560
        coords = np.asarray(self.case.mesh.nodes_coords, dtype=np.float64)
        baseline = damping_alpha(coords, runner.ALPHA_MAX_S_INV)
        np.testing.assert_array_equal(self.physics[2].values.ravel(), self.profile.amplitude_factor*baseline)
        np.testing.assert_array_equal(self.physics[2].values.ravel(), runner.profile_alpha(coords, self.profile))
        times = np.linspace(0, self.spec.t_end_s, self.spec.steps+1)
        np.testing.assert_array_equal(self.physics[-1].values[:, 0, 1], -ricker(times))
        np.testing.assert_array_equal(self.physics[-1].values[:, 0, 0], 0.)
        assert len(self.receivers) == (2 if self.save_fields else 1)
        for i, receiver in enumerate(self.receivers):
            np.testing.assert_array_equal(receiver.comp_codes, [4, 5])
            assert receiver.sens_count == (70 if i == 0 else len(coords))
            assert receiver.stride == (96 if i == 0 else 960)
            assert receiver.end_tact == self.spec.steps

    def run(self, sync: bool) -> Result:
        assert sync
        for time in inclusive_times(self.spec.t_end_s, .001):
            step = round(time / self.spec.dt_s)
            for receiver in self.receivers:
                if step % receiver.stride or (self.missing_last and step == self.spec.steps):
                    continue
                ids = receiver_nodes(self.case) if receiver.sens_count == 70 else np.arange(len(self.case.mesh.nodes_coords))
                values = time*(ids+1)
                frame = np.vstack((values, -2*values)).astype(np.float64)
                receiver.handler(frame.ravel(), Event(0, step, float(time)), Control())
        return Result(0, self.spec.steps, self.spec.t_end_s)


def checks() -> None:
    output = runner.ROOT / "cpu_checks_v1"
    output.mkdir(parents=True, exist_ok=False)
    native = output / "mock_native.txt"
    native.write_text("CPU MOCK; no native or GPU solution\n")
    selected = runner.PROFILES["q2_half"]
    save_fields = False
    missing = False

    def factory(case: MeshCase, spec: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        return cast("SGModel[CUConfig]", ProfileMock(case, spec, selected, save_fields, missing)), native, 0

    for selected in runner.PROFILES.values():
        with patch.object(runner, "make_cu_model", side_effect=factory):
            result = runner.run(selected, output / selected.name, .01)
        assert result["recorded_counts"] == {"surface": 11, "fields": 0, "source_history": 0}
        assert result["profile"] == runner.profile_record(selected)
        assert mapping(result["profile_stability"])["positive_nodal_increment_over_reference_s_inv"] == 0.
        assert not list((output/selected.name).glob("*.npy"))
    save_fields = True
    with patch.object(runner, "make_cu_model", side_effect=factory):
        result = runner.run(selected, output / "optional_fields_cpu_only", .01, True)
    assert result["recorded_counts"] == {"surface": 11, "fields": 2, "source_history": 0}
    assert result["all_node_surface_values_exact"] is True
    save_fields, missing = False, True
    with patch.object(runner, "make_cu_model", side_effect=factory):
        expect_error(lambda: runner.run(selected, output / "missing_last", .01), ValueError)
    assert read_json(output / "missing_last/metadata.failed.json")["status"] == "failed"
    assert not (output / "missing_last/metadata.json").exists()
    expect_error(lambda: runner.Profile("bad_negative", "power", 2., -1.), ValueError)
    expect_error(lambda: runner.Profile("bad_strong", "power", 2., 2.), ValueError)
    expect_error(lambda: runner.Profile("bad_path/escape", "power", 2., .5), ValueError)
    expect_error(lambda: runner.profile_alpha(np.array([[-176., 0.]]), selected), ValueError)
    assert not any(name.endswith("._native") and "lin_elast_2d" in name for name in sys.modules)
    runner.homogeneous.write_json(output / "validation.json", {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU mocks on actual historical mesh; no native/GPU solution",
        "checks": ["Both nodal arrays exactly equal reference alpha times0.5/0.25",
            "Actual constant material and rhoVp/rhoVs ABC coefficients unchanged;560 edges",
            "Same downward Ricker force and exact 70 receiver positions/component order",
            "Surface-only mode emits no all-node arrays; optional dual mode agrees for nonuniform node-dependent data",
            "Missing final output fails; invalid profile parameters/domain rejected",
            "Both profiles pointwise dominated by reference; K96 conservative bound unchanged"],
        "code": [runner.file_record(Path(__file__)), runner.file_record(Path(runner.__file__))]})
    print("Profile-sweep CPU checks passed; native/GPU not loaded", flush=True)


def completed() -> None:
    outputs: dict[str, object] = {}
    for name, profile in runner.PROFILES.items():
        output = runner.ROOT / "runs" / name
        metadata = runner.verify(output, profile)
        with np.load(output / "surface.npz", allow_pickle=False) as archive:
            values = archive["velocity_m_s"]
            vx = float(np.linalg.norm(values[:, :, 0]+values[:, ::-1, 0])/np.linalg.norm(values[:, :, 0]))
            vy = float(np.linalg.norm(values[:, :, 1]-values[:, ::-1, 1])/np.linalg.norm(values[:, :, 1]))
            assert max(vx, vy) < 1e-8
        outputs[name] = {"metadata": runner.file_record(output / "metadata.json"), "recorded_counts": metadata["recorded_counts"],
            "wall_seconds": metadata["wall_seconds"], "symmetry": {"vx_odd_relative_l2": vx, "vy_even_relative_l2": vy},
            "source_native_nodes_equal_reference": True, "profile": metadata["profile"]}
    runner.homogeneous.write_json(runner.ROOT / "completion_validation.json", {"status": "passed",
        "utc": datetime.now(timezone.utc).isoformat(), "variants": outputs,
        "scope": "Complete finite surface records, exact source/native/mesh, alpha/ABC configuration and symmetry; not independent solver validation"})
    print("Profile-sweep completion checks passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed", action="store_true")
    arguments = parser.parse_args()
    completed() if arguments.completed else checks()
