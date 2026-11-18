"""CPU-only core-runner checks using actual geometry and a mocked CU model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, cast
from unittest.mock import patch

import numpy as np
from semgpu2.model.model import SGModel

from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
from scripts.dev_1_3_point_source.physics import ricker
from scripts.dev_1_4_server_preparation import no_padding_run as runner
from scripts.dev_1_4_server_preparation.check_surface_runner import MockModel, expect_error
from scripts.dev_1_4_server_preparation.surface_runner import SurfaceSpec
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, read_json

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig


class CoreMock(MockModel):
    def __init__(self, case: MeshCase, spec: SurfaceSpec) -> None:
        super().__init__(spec)
        self.case = case

    def prepare(self) -> None:
        assert [physics.type_code for physics in self.physics] == [11000, 32000, 20200]
        edges = runner.core_abc_edges(self.case)
        assert len(edges) == 420
        np.testing.assert_array_equal(self.physics[1].entities, edges)
        expected_abc = 1000 * np.column_stack((self.case.boundary_vp[edges], self.case.boundary_vs[edges]))
        np.testing.assert_array_equal(self.physics[1].values.reshape(-1, 2), np.repeat(expected_abc, 5, axis=0))
        times = np.linspace(0., self.spec.t_end_s, self.spec.steps + 1)
        np.testing.assert_array_equal(self.physics[2].values[:, 0, 0], 0.)
        np.testing.assert_array_equal(self.physics[2].values[:, 0, 1], -ricker(times))
        assert self.physics[2].entities.size == 1
        assert len(self.receivers) == 1
        receiver = self.receivers[0]
        np.testing.assert_array_equal(receiver.comp_codes, [4, 5])
        assert receiver.sens_count == 70 and receiver.stride == 96
        assert receiver.begin_tact == 0 and receiver.end_tact == self.spec.steps


def checks(mesh: Path, stability: Path, output: Path) -> dict[str, object]:
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    output.mkdir(parents=True, exist_ok=False)
    case = build_sem(load_frozen(mesh), 4)
    case.metadata["dataset"] = "EFVB"
    spec = runner.CoreSpec("EFVB", 0, 350., 96)
    edges = runner.validate_core(case, spec, mesh, stability)
    coords = case.mesh.nodes_coords
    boundary = case.mesh.strata[2]
    edge_nodes = boundary.p_nodes.reshape(-1, 5)
    assert len(edges) == 420 and len(edge_nodes) == 560
    assert not np.any(np.all(np.abs(coords[edge_nodes[edges], 1]) < 1e-9, axis=1))
    top = next(ids for ids in edge_nodes if np.all(np.abs(coords[ids, 1]) < 1e-9))
    node = int(top[1])
    previous = float(coords[node, 1])
    coords[node, 1] = -.25
    try:
        expect_error(lambda: runner.core_abc_edges(case), ValueError)
    finally:
        coords[node, 1] = previous
    certificate = read_json(stability)
    for key, value in (("alpha_max_s_inv", 1.), ("dt_start_s", 1e-7)):
        invalid = output / f"invalid_{key}.json"
        invalid.write_text(json.dumps({**certificate, key: value}))
        expect_error(lambda: runner.validate_core(case, spec, mesh, invalid), ValueError)
    native = output / "mock_native.txt"
    native.write_text("CPU MOCK; no solver/native execution\n")

    def factory(actual_case: MeshCase, actual_spec: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
        return cast("SGModel[CUConfig]", CoreMock(actual_case, actual_spec)), native, 0

    with patch.object(runner, "make_cu_model", side_effect=factory):
        completed = runner.run_core(mesh, stability, output / "completed_cpu_mock")
    assert completed["status"] == "completed"
    assert completed["physics_ids"] == [11000, 32000, 20200]
    assert completed["recorded_counts"] == {"surface": 2001, "source_history": 0, "fields": 0}
    reference = read_json(PROJECT_ROOT / "data/dev_1_4_server_preparation/benchmark_3060_v1/EFVB_00000/x350/metadata.json")
    assert completed["source_input_sha256"] == reference["source_input_sha256"]
    with np.load(output / "completed_cpu_mock/surface.npz", allow_pickle=False) as archive:
        assert archive["velocity_m_s"].shape == (2001, 70, 2)
        np.testing.assert_array_equal(archive["velocity_m_s"][-1, :, 0], 2.)
        np.testing.assert_array_equal(archive["velocity_m_s"][-1, :, 1], -4.)
    assert not any(name.startswith("semgpu2.solvers.") and name.endswith("._native") for name in sys.modules)
    return {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(),
            "scope": "CPU mock using actual core geometry; no native/GPU solution and no ABC accuracy/convergence claim",
            "physics_ids_exact": [11000, 32000, 20200], "mass_damping_absent": True,
            "abc_only_core_sides_bottom": True, "abc_edge_count": 420, "free_top_edge_count": 140,
            "bad_boundary_and_certificate_rejected": True, "K96_satisfies_core_certificate": True,
            "source_input_sha_matches_padded_run": True, "surface_shape": [2001, 70, 2],
            "core_nodes": mapping(completed["mesh"])["nodes"],
            "sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ("no_padding_run.py", "check_no_padding_run.py", "surface_runner.py")}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    record = checks(args.mesh, args.stability, args.out)
    (args.out / "validation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
