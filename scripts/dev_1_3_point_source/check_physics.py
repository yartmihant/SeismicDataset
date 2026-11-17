"""Адресные CPU-проверки физических соглашений; решатель не импортируется."""

from __future__ import annotations

import unittest

import numpy as np

from scripts.dev_1_3_point_source.physics import (
    ExperimentSpec, damping_alpha, inclusive_times,
    material_coordinates, nodal_point_force, ricker, source_work,
)


class PhysicsChecks(unittest.TestCase):
    def test_observation_grids(self) -> None:
        spec = ExperimentSpec()
        self.assertEqual(spec.force_amplitude_n_per_m, 1.0)
        for interval, count in ((spec.output_interval_s, 2001), (spec.field_interval_s, 201)):
            times = inclusive_times(spec.end_s, interval)
            self.assertEqual(times.size, count)
            self.assertEqual(times[0], 0.0)
            self.assertEqual(times[-1], 2.0)
        with self.assertRaises(ValueError):
            inclusive_times(2.0, 0.03)

    def test_ricker_peak_and_zero_crossings(self) -> None:
        spec = ExperimentSpec()
        offset = 1.0 / (np.sqrt(2.0) * np.pi * 30.0)
        values = ricker(np.array([spec.center_s - offset, spec.center_s, spec.center_s + offset]))
        np.testing.assert_allclose(values, [0.0, 1.0, 0.0], atol=2e-15)

    def test_point_force_is_downward_and_exact(self) -> None:
        x = np.arange(141, dtype=np.float64) * 5.0
        nodes = np.column_stack((x, np.zeros_like(x)))
        for source in ExperimentSpec().source_x_m:
            load = nodal_point_force(nodes, source, 3.0)
            self.assertEqual(nodes[load.node_id, 0], source)
            self.assertEqual(load.force_x, 0.0)
            self.assertEqual(load.force_y, -3.0)
        with self.assertRaises(ValueError):
            nodal_point_force(nodes, 172.5, 1.0)
        with self.assertRaises(ValueError):
            nodal_point_force(np.vstack((nodes, [[175.0, 0.0]])), 175.0, 1.0)

    def test_damping_interfaces_and_corner(self) -> None:
        points = np.array([[0.0, 0.0], [700.0, -700.0], [350.0, -350.0],
                           [-87.5, -350.0], [-175.0, -350.0], [350.0, -875.0],
                           [875.0, -875.0]])
        np.testing.assert_allclose(damping_alpha(points, 12.0), [0, 0, 0, 3, 12, 12, 24])
        with self.assertRaises(ValueError):
            damping_alpha(np.array([[0.0, -876.0]]), 12.0)
        np.testing.assert_allclose(damping_alpha(np.array([[875.0 + 1e-12, -875.0 - 1e-12]]), 12.0), [24.0])

    def test_material_extension_preserves_tangential_coordinate(self) -> None:
        points = np.array([[-30.0, -123.0], [750.0, -456.0], [234.0, -800.0],
                           [-175.0, -875.0], [350.0, -350.0]])
        np.testing.assert_array_equal(material_coordinates(points),
                                      [[0, -123], [700, -456], [234, -700], [0, -700], [350, -350]])

    def test_work_includes_negative_power_and_linear_scaling(self) -> None:
        # Constant downward force and affine velocity: the first half of the
        # interval returns energy, and trapezoidal quadrature is exact.
        times = np.linspace(0.0, 2.0, 41)
        force = np.column_stack((np.zeros_like(times), -np.ones_like(times)))
        velocity = np.column_stack((np.zeros_like(times), 0.5 - times))
        work = source_work(times, force, velocity)
        self.assertAlmostEqual(work, 1.0, places=14)
        amplitude = 3.0
        self.assertAlmostEqual(source_work(times, force * amplitude, velocity * amplitude), 9.0, places=13)
        with self.assertRaises(ValueError):
            source_work(times[::-1], force, velocity)


if __name__ == "__main__":
    unittest.main()
