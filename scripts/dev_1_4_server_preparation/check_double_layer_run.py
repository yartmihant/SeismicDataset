"""CPU mock: actual uniform material/ABC coefficients, dual recording, no native."""

from __future__ import annotations

import argparse
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
from scripts.dev_1_4_server_preparation import double_layer_run as runner
from scripts.dev_1_4_server_preparation import velocity_fields_run as fields
from scripts.dev_1_4_server_preparation.check_surface_runner import expect_error
from scripts.dev_1_4_server_preparation.check_velocity_fields import DualMock
from scripts.dev_1_4_server_preparation.surface_runner import SurfaceSpec

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


class HomogeneousMock(DualMock):
    def __init__(self, case: MeshCase, spec: SurfaceSpec, profile: runner.Profile) -> None:
        super().__init__(case,spec)
        self.profile=profile

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
        np.testing.assert_array_equal(self.physics[2].values.ravel(),runner.profile_alpha(np.asarray(self.case.mesh.nodes_coords,dtype=np.float64),self.profile))
        edges=runner.abc_edges(self.case)
        np.testing.assert_array_equal(self.physics[1].entities,edges)
        assert len(edges)==700


def checks() -> None:
    output = runner.ROOT / "cpu_checks_v1"
    output.mkdir(parents=True, exist_ok=False)
    native = output / "mock_native.txt"
    native.write_text("CPU MOCK; not a native solution\n")

    def factory(case: MeshCase, spec: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        return cast("SGModel[CUConfig]", HomogeneousMock(case, spec, selected)), native, 0

    for profile in ("stretched350", "fixed175plateau"):
        selected = cast(runner.Profile, profile)
        frozen = load_frozen(runner.mesh_inputs("padded")[0])
        bad = replace(frozen, vp=np.full_like(frozen.vp, 3500.))
        expect_error(lambda: runner.check_material(bad), ValueError)
        with patch.object(runner, "make_cu_model", side_effect=factory):
            result = runner.run("padded", output / profile, .01, selected)
        assert result["status"] == "completed" and result["physical_model"] == runner.PHYSICAL_MODEL
        assert result["recorded_counts"] == {"surface": 11, "fields": 2, "source_history": 0}
        assert result["is_original_dataset_model"] is False and result["all_node_surface_values_exact"] is True
    assert not any(name.endswith("._native") and "lin_elast_2d" in name for name in sys.modules)
    runner.write_json(output / "validation.json", {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CPU mocked model; actual homogeneous mesh and assembled material/ABC coefficients, no GPU/native solution",
        "physical_model": runner.PHYSICAL_MODEL,
        "checks": ["Vp/Vs/rho constant in every element and boundary coefficient",
            "Actual rho/E/nu and rhoVp/rhoVs passed to model agree with the requested material",
            "Both350m profiles use actual requested nodal alpha and700 exterior ABC edges; free top",
            "Two recorders preserve Vx/Vy ordering and exact values at common nodes/times",
            "Explicit synthetic material identity, new mesh and stability hashes; invalid material rejected"],
        "files": [fields.stream_record(Path(__file__)), fields.stream_record(Path(runner.__file__))]})
    print("Double-layer CPU checks passed; no native/GPU loaded", flush=True)


def completed() -> None:
    from scripts.reporting import read_json,mapping,number
    parent=runner.PROJECT_ROOT/"data/dev_1_4_server_preparation/homogeneous_3060_v1/padded/metadata.json"
    reference=read_json(parent)
    variants:dict[str,object]={}
    for name,profile in (("padded350","stretched350"),("held175_profile","fixed175plateau")):
        output=runner.ROOT/name
        metadata=runner.verify(output,runner.spec_for("padded"))
        assert metadata["source_input_sha256"]==reference["source_input_sha256"]
        assert mapping(metadata["native"])["sha256"]==mapping(reference["native"])["sha256"]
        assert metadata["damping_profile"]==runner.profile_record(cast(runner.Profile,profile))
        with np.load(output/"surface.npz") as archive:
            values=np.asarray(archive["velocity_m_s"],dtype=np.float64)
            reflected=values[:,::-1,:]
            vx=float(np.linalg.norm(values[:,:,0]+reflected[:,:,0])/np.linalg.norm(values[:,:,0]))
            vy=float(np.linalg.norm(values[:,:,1]-reflected[:,:,1])/np.linalg.norm(values[:,:,1]))
            assert max(vx,vy)<1e-8
        variants[name]={"wall_seconds":number(metadata["wall_seconds"]),"recorded_counts":metadata["recorded_counts"],
            "field_shape":metadata["field_shape"],"symmetry":{"vx_odd_relative_l2":vx,"vy_even_relative_l2":vy},
            "source_and_native_equal_reference":True,"metadata":runner.file_record(output/"metadata.json")}
    runner.write_json(runner.ROOT/"completion_validation.json",{"status":"passed","utc":datetime.now(timezone.utc).isoformat(),
        "variants":variants,"reference":runner.file_record(parent),"scope":"Complete finite records, material/profile/boundary configuration, same input/native and symmetry; not independent solver accuracy validation"})
    print("Double-layer complete validation passed",flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--completed",action="store_true")
    args=parser.parse_args()
    completed() if args.completed else checks()
