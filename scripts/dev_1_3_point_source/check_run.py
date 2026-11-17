"""Малый native-smoke регистратора; синтетическая среда не заменяет два опыта."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Sequence
import unittest

import numpy as np

from semgpu2.model import SGPETALineModel
from semgpu2.tools.mesh.box import rectangle_quad_fast
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_3_point_source.run import RunSpec, abc_edges, exact_surface_nodes, run_case, validate_stability
from scripts.project_paths import PROJECT_ROOT


def synthetic_case(order: int) -> MeshCase:
    """Редкая по вертикали сетка только для проверки интерфейсов и записей."""
    mesh = rectangle_quad_fast(sizes=(1050.0, 875.0), elements=(210, 2),
                               pets=[SGPETALineModel("GLL", order=order)])
    mesh.nodes_coords[:] += np.array([350.0, -437.5])
    mesh.centers = mesh.build_centers()
    count = mesh.strata[1].p_node_offsets.size - 1
    boundary_count = mesh.strata[2].p_node_offsets.size - 1
    return MeshCase(mesh, np.full(count, 2000.0), np.full(count, 1000.0),
                    np.full(boundary_count, 2000.0), np.full(boundary_count, 1000.0),
                    {"synthetic_smoke_only": True, "physical_model_id": None,
                     "geometry": "rectangle_210_by_2_not_a_scientific_mesh", "order": order})


class RecorderChecks(unittest.TestCase):
    def test_schedule_and_invalid_request(self) -> None:
        spec = RunSpec(12404, "raster", 4, 350.0, 10)
        self.assertEqual(spec.steps, 20000)
        self.assertAlmostEqual(spec.dt_s, 0.0001)
        with self.assertRaises(ValueError):
            RunSpec(12404, "raster", 4, 350.0, 10, t_end_s=0.011)

    def test_surface_nodes_and_abc_do_not_touch_top(self) -> None:
        for order in (3, 4, 5):
            case = synthetic_case(order)
            ids = exact_surface_nodes(case)
            np.testing.assert_allclose(case.mesh.nodes_coords[ids, 0], np.arange(141) * 5)
            np.testing.assert_allclose(case.mesh.nodes_coords[ids, 1], 0)
            self.assertEqual(abc_edges(case).size, 214)

    def test_stability_rejects_mismatches_before_output(self) -> None:
        case = synthetic_case(4)
        spec = RunSpec(12404, "raster", 4, 350.0, 20)
        with TemporaryDirectory(prefix="tmp_run_stability_", dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            mesh = root / "mesh.npz"
            evidence = root / "stability.json"
            valid: dict[str, object] = {"mesh": str(mesh), "order": 4,
                "nodes": case.mesh.nodes_coords.shape[0], "elements": case.element_vp.size,
                "alpha_max_s_inv": spec.alpha_max_s_inv, "dt_start_s": 0.0001}
            evidence.write_text(json.dumps(valid))
            validate_stability(case, spec, mesh, evidence)
            for key, wrong in (("order", 5), ("dt_start_s", 0.00001),
                               ("alpha_max_s_inv", 1.0), ("mesh", str(root / "other.npz"))):
                evidence.write_text(json.dumps({**valid, key: wrong}))
                with self.assertRaises(ValueError):
                    validate_stability(case, spec, mesh, evidence)


def smoke(root: Path, backend: str, resource: str, layout: str = "full",
          orders: Sequence[int] = (4, 5)) -> None:
    """Реальный native-сигнал с пиком Рикера: проверить все каналы и времена."""
    if not orders or len(set(orders)) != len(orders) or any(order not in (3, 4, 5) for order in orders):
        raise ValueError("Нужны различные порядки из p3/p4/p5")
    for order in orders:
        output = root / f"synthetic_p{order}"
        spec = RunSpec(12404, "raster", order, 350.0, 20, 0.06, backend, resource, field_layout=layout)
        record = run_case(synthetic_case(order), spec, output)
        if record["status"] != "completed":
            raise AssertionError("Native smoke did not complete")
        with np.load(output / "surface.npz") as surface:
            if surface["displacement_m"].shape != (61, 141, 2):
                raise AssertionError("Wrong smoke surface shape")
            if np.max(np.abs(surface["velocity_m_s"])) <= 0:
                raise AssertionError("Point source produced no signal")
        with np.load(output / "source_history.npz") as source:
            if source["time_s"].size != 1201 or not np.isclose(source["force_y_n_per_m"].min(), -1, atol=1e-12):
                raise AssertionError("Wrong source amplitude or internal schedule")
        with np.load(output / "energy_balance.npz") as energy:
            if energy["time_s"].size != 7 or np.max(energy["kinetic_j_per_m"]) <= 0:
                raise AssertionError("Missing energy frames")
            if np.any(energy["elastic_j_per_m"] < -1e-20):
                raise AssertionError("Negative elastic energy")
        with np.load(output / "fields/frame_0006.npz") as field:
            if ("displacement_m" in field.files) != (layout == "full"):
                raise AssertionError("Field layout was not applied")
            if field["velocity_m_s"].dtype != np.float64 or field["elastic_energy_density_j_m3"].dtype != np.float64:
                raise AssertionError("Field precision changed")
    print("Native smoke " + "/".join(f"p{order}" for order in orders) +
          ": source, U/V, energy and inclusive schedules OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--backend", choices=("ceed", "cu"), default="ceed")
    parser.add_argument("--ceed-resource", default="/cpu/self/opt/blocked")
    parser.add_argument("--field-layout", choices=("full", "compact"), default="full")
    parser.add_argument("--orders", type=int, nargs="+", choices=(3, 4, 5), default=[4, 5],
                        help="Порядки native-smoke; по умолчанию сохранены p4/p5")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "data/dev_1_3_point_source/smoke_v1")
    args = parser.parse_args()
    result = unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(RecorderChecks))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.native:
        smoke(args.out, args.backend, args.ceed_resource, args.field_layout, args.orders)


if __name__ == "__main__":
    main()
