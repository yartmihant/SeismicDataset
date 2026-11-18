"""CPU mock: actual uniform material/ABC coefficients, dual recording, no native."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
from semgpu2.model.model import SGModel

from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.mesh import load_frozen
from scripts.dev_1_4_server_preparation import homogeneous_run as runner
from scripts.dev_1_4_server_preparation import velocity_fields_run as fields
from scripts.dev_1_4_server_preparation.check_surface_runner import expect_error
from scripts.dev_1_4_server_preparation.check_velocity_fields import DualMock
from scripts.dev_1_4_server_preparation.surface_runner import SurfaceSpec

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


class HomogeneousMock(DualMock):
    def prepare(self) -> None:
        super().prepare()
        rho, vp, vs = runner.DENSITY_KG_M3, runner.VP_M_S, runner.VS_M_S
        mu, lam = rho * vs**2, rho * (vp**2 - 2 * vs**2)
        material = np.array([rho, mu * (3 * lam + 2 * mu) / (lam + mu), lam / (2 * (lam + mu))])
        actual = self.physics[0].values.reshape(-1, 3)
        np.testing.assert_allclose(actual, np.broadcast_to(material, actual.shape), rtol=2e-15, atol=0.)
        impedance = self.physics[1].values.reshape(-1, 2)
        np.testing.assert_array_equal(impedance, np.broadcast_to([rho * vp, rho * vs], impedance.shape))
        assert np.all(self.case.element_vp == vp) and np.all(self.case.element_vs == vs)
        assert self.case.metadata["is_original_dataset_model"] is False


def checks() -> None:
    output = runner.ROOT / "cpu_checks_v1"
    output.mkdir(parents=True, exist_ok=False)
    native = output / "mock_native.txt"
    native.write_text("CPU MOCK; not a native solution\n")

    def factory(case: MeshCase, spec: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        return cast("SGModel[CUConfig]", HomogeneousMock(case, spec)), native, 0

    for variant in ("padded", "core"):
        selected = cast(fields.Variant, variant)
        frozen = load_frozen(runner.mesh_inputs(selected)[0])
        bad = replace(frozen, vp=np.full_like(frozen.vp, 3500.))
        expect_error(lambda: runner.check_material(bad), ValueError)
        with patch.object(fields, "make_cu_model", side_effect=factory):
            result = runner.run(selected, output / variant, .01)
        assert result["status"] == "completed" and result["physical_model"] == runner.PHYSICAL_MODEL
        assert result["recorded_counts"] == {"surface": 11, "fields": 2, "source_history": 0}
        assert result["is_original_dataset_model"] is False and result["all_node_surface_values_exact"] is True
    assert not any(name.endswith("._native") and "lin_elast_2d" in name for name in sys.modules)
    runner.write_json(output / "validation.json", {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU mocked model; actual homogeneous mesh and assembled material/ABC coefficients, no GPU/native solution",
        "physical_model": runner.PHYSICAL_MODEL,
        "checks": ["Vp/Vs/rho constant in every element and boundary coefficient",
            "Actual rho/E/nu and rhoVp/rhoVs passed to model agree with the requested material",
            "Padded/core ABC and mass-damping selection preserve preceding boundary experiments",
            "Two recorders preserve Vx/Vy ordering and exact values at common nodes/times",
            "Explicit synthetic material identity, new mesh and stability hashes; invalid material rejected"],
        "files": [fields.stream_record(Path(__file__)), fields.stream_record(Path(runner.__file__))]})
    print("Homogeneous CPU checks passed; no native/GPU loaded", flush=True)


if __name__ == "__main__":
    checks()
