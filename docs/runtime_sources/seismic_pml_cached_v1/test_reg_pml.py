"""SEMGPU Test Card

id
    package: semgpu2
    suite: lin_elast_2d_seismic_newmark_kim_cu
    test_name: pml
    level: reg
    tags: [cuda, pml, restart]

Назначение
    Проверить направленное растяжение, сложение профилей и отражённый импульс.

Постановка
    Однородный прямоугольник, свободные грани, продольная волна при nu=0.
    Сравнение записи с удлинённой областью той же пространственной дискретизации.

Машинный критерий
    Ошибка амплитуды в окне отражения меньше 1% падающего импульса.
    Без слоя в этом окне отражение превышает 90%. Первый приход совпадает.
    Продолжение со всей памятью воспроизводит непрерывный расчёт.

Артефакты
    pytest report

Запуск
    targeted:
        .venv/bin/python -m pytest semgpu2/tests/tests_lin_elast_2d/seismic_newmark_kim/test_reg_pml.py
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import numpy as np
import pytest

from semgpu2.model import SGModel, SGPETALineModel, SGPhysicsModel
from semgpu2.solvers import lin_elast_2d_seismic_newmark_kim_cu as seismic
from semgpu2.solvers.solver_api import Float64Array, SGPhysicsData, SGReceiverData
from semgpu2.tools.mesh.box import rectangle_quad


def pulse_trace(length: int, axis: int, with_layer: bool) -> Float64Array:
    sizes = (float(length), .25) if axis == 0 else (.25, float(length))
    grid = (length * 4, 1) if axis == 0 else (1, length * 4)
    mesh = rectangle_quad(sizes=sizes, elements=grid, pets=[SGPETALineModel("line", order=3)])
    solver = seismic.SGSolver(seismic.SGSolverConfig(t_end=11., steps_count=1100, steps_per_tact=5))
    model = SGModel(mesh, solver)
    model.add_physics(SGPhysicsModel.from_constant(
        "material", int(seismic.SGPhysicsTypeEnum.MATERIAL_ISO_PLANE_STRAIN), mesh, 1,
        mesh.elements_filter(1), np.array([1., 1., 0.]),
    ))
    start = 5. - length / 2.
    if with_layer:
        elements = mesh.elements_filter(1, lambda c: bool(c[axis] > start))

        def profile(coord: Float64Array, time: float | None, node_id: int, elem_id: int | None) -> Float64Array:
            values = np.zeros(2)
            values[axis] = 5. * (max(0., coord[axis] - start) / 3.) ** 2
            return values

        model.add_physics(SGPhysicsModel.from_formula(
            "layer", int(seismic.SGPhysicsTypeEnum.PML), mesh, 1, elements, profile,
        ))
    model.prepare()
    distance = mesh.nodes_coords[:, axis] + length / 2.
    # Right-going Gaussian displacement. rho=E=1, nu=0 gives cp=1 and
    # zero traction on the two longitudinal sides for this exact solution.
    offset = distance - 2.
    width = .35
    shape = np.exp(-(offset / width) ** 2)
    solver.solutions[0][axis, :] = shape
    solver.solutions[0][axis + 2, :] = 2. * offset * shape / width**2
    solver.solutions[0][axis + 4, :] = (4. * offset**2 / width**4 - 2. / width**2) * shape
    station = int(np.argmin(np.abs(distance - 4.)))
    probe = SGReceiverData(0, np.array([4 + axis], dtype=np.int32), np.array([station]),
                           np.array([0]), np.array([1.]), 1, 0, -1, 1, 8)
    trace: list[float] = []
    solver.run([probe], lambda _, frame, event, control: trace.append(float(frame.item())), sync=True)
    return np.asarray(trace)


@pytest.mark.parametrize("axis", [0, 1])
def test_reflected_p_wave(axis: int, record_property: Callable[[str, float], None]) -> None:
    reference = pulse_trace(16, axis, False)
    reflected = pulse_trace(8, axis, False)
    layer = pulse_trace(8, axis, True)
    times = np.arange(reference.size) * .05
    direct = (times > 1.) & (times < 3.)
    incident = np.max(np.abs(reference[direct]))
    assert incident > 1.
    np.testing.assert_allclose(layer[direct], reference[direct], rtol=0., atol=1e-5 * incident)
    # Start before an entrance reflection (t=4); include the wall return (t=10).
    late = times >= 3.5
    record_property("pml_relative_echo", float(np.max(np.abs(layer[late] - reference[late])) / incident))
    record_property("free_relative_echo", float(np.max(np.abs(reflected[late] - reference[late])) / incident))
    assert np.max(np.abs(reflected[late] - reference[late])) > .9 * incident
    assert np.max(np.abs(layer[late] - reference[late])) < .01 * incident


def corner_model(config: seismic.SGSolverConfig) -> seismic.SGSolver:
    mesh = rectangle_quad(sizes=(2., 1.), elements=(2, 1), pets=[SGPETALineModel("line", order=2)])
    solver = seismic.SGSolver(config)
    model = SGModel(mesh, solver)
    for element, density in enumerate([2., 3.]):
        model.add_physics(SGPhysicsModel.from_constant(
            "material", int(seismic.SGPhysicsTypeEnum.MATERIAL_ISO_PLANE_STRAIN), mesh, 1,
            np.array([element]), np.array([density, 6., .25]),
        ))
    return solver


def strip(dx: float, dy: float) -> SGPhysicsData:
    return SGPhysicsData(int(seismic.SGPhysicsTypeEnum.PML), 1, np.array([1]),
                         np.repeat([[dx], [dy]], 9, axis=1))


@pytest.mark.parametrize("defect", ["negative", "nan", "carrier", "components", "dynamic", "cfl"])
def test_invalid_pml(defect: str) -> None:

    cfg = seismic.SGSolverConfig(cfl_safety=.5, t_end=1., tacts_count=1) if defect == "cfl" else seismic.SGSolverConfig()
    solver = corner_model(cfg)
    pml = strip(2., 1.)
    if defect == "negative":
        pml.values[0, 0] = -1.
    elif defect == "nan":
        pml.values[1, -1] = np.nan
    elif defect == "carrier":
        pml = replace(pml, space_id=2)
    elif defect == "components":
        pml = replace(pml, values=pml.values[:1])
    elif defect == "dynamic":
        pml = replace(pml, values=pml.values[None, :, :], t_begin=0., t_end=1., interp_code=0)
    solver.physics[2] = pml
    with pytest.raises(seismic.SGPrepareError) as error:
        solver.prepare()
    expected = seismic.SGPrepareErrorCodeEnum.SPECTRUM_FAILURE if defect == "cfl" else seismic.SGPrepareErrorCodeEnum.PHYSICS_FAILURE
    assert error.value.code == expected


def test_overlapping_strips_corner_inertia() -> None:
    # A rigid translation has zero gradient. Only the inertial correction remains.
    # At shared nodes the adjacent, different material also contributes to MASS.
    dt = .001
    solver = corner_model(seismic.SGSolverConfig(t_end=dt, steps_count=1, steps_per_tact=1))
    solver.physics[2] = strip(2., 0.)
    solver.physics[3] = strip(0., 3.)
    solver.prepare()
    xy = solver.geometries[0][:2, :]
    share = np.where(xy[0] > 1e-12, 1., np.where(np.abs(xy[0]) < 1e-12, 3./5., 0.))
    shifts = np.array([[.5], [-1.]])
    velocity = np.array([[.2], [.1]])
    solver.solutions[0][:2, :] = np.broadcast_to(shifts, (2, share.size)).copy()
    solver.solutions[0][2:4, :] = np.broadcast_to(velocity, (2, share.size)).copy()
    solver.run([], lambda *_: None, sync=True)
    expected = -(5. * velocity + 6. * (shifts + dt * velocity)) * share
    np.testing.assert_allclose(solver.solutions[0][4:, :], expected, atol=1e-11, rtol=1e-12)


def test_directional_flux_virtual_work() -> None:
    # Linear test functions recover each physical component of the weak flux.
    dt = .001
    cfg = seismic.SGSolverConfig(t_end=dt, steps_count=1, steps_per_tact=1)
    ordinary = corner_model(cfg)
    layer = corner_model(cfg)
    dx, dy = 2., 3.
    layer.physics[2] = strip(dx, dy)
    gradient = np.array([[.2, -.3], [.4, .1]])  # [displacement component, direction]
    for solver in (ordinary, layer):
        solver.prepare()
        xy = solver.geometries[0][:2, :]
        solver.solutions[0][:2, :] = gradient @ xy
        solver.run([], lambda *_: None, sync=True)
    mass = layer.materials[0][0, :]
    delta_force = (layer.solutions[0][4:, :] - ordinary.solutions[0][4:, :]) * mass
    # The layered element is a unit square of density 3. Remove its inertial contribution.
    fraction = np.where(xy[0] > 1e-12, 1., np.where(np.abs(xy[0]) < 1e-12, 3./5., 0.))
    delta_force += dx * dy * (gradient @ xy) * mass * fraction
    mu = 6. / (2 * 1.25)
    lam = 6. * .25 / (1.25 * .5)
    history_factors = np.array([-np.expm1(-dt * d) / d for d in (dx, dy)])
    stiffness = np.array([[lam + 2*mu, mu], [mu, lam + 2*mu]])
    flux = stiffness * gradient * history_factors * np.array([dy - dx, dx - dy])
    np.testing.assert_allclose(delta_force @ xy.T, -flux, atol=5e-9, rtol=2e-6)


def surface_trace(half_width: float, depth: float, use_pml: bool) -> Float64Array:
    mesh = rectangle_quad(sizes=(2. * half_width, depth), elements=(int(8 * half_width), int(4 * depth)),
                          pets=[SGPETALineModel("line", order=4)])
    solver = seismic.SGSolver(seismic.SGSolverConfig(t_end=11., steps_count=2200, steps_per_tact=10))
    model = SGModel(mesh, solver)
    # rho=lambda=mu=1, cp=sqrt(3), cs=1. The top is traction-free.
    model.add_physics(SGPhysicsModel.from_constant(
        "rock", 11000, mesh, 1, mesh.elements_filter(1), np.array([1., 2.5, .25]),
    ))
    coords = mesh.nodes_coords
    top = depth / 2.
    source = int(np.argmin(np.square(coords[:, 0]) + np.square(coords[:, 1] - top)))
    station = int(np.argmin(np.square(coords[:, 0] - 1.) + np.square(coords[:, 1] - top)))
    time = np.linspace(0., 1.6, 321)
    phase = (np.pi * 1.5 * (time - .8)) ** 2
    loads = np.zeros((time.size, 2, 1))
    loads[:, 1, 0] = (1. - 2. * phase) * np.exp(-phase)
    solver.physics[10] = SGPhysicsData(20200, 0, np.array([source]), loads, t_begin=0., t_end=1.6, interp_code=1)
    if use_pml:
        start_x = half_width - 2.
        start_y = -top + 2.
        elements = mesh.elements_filter(1, lambda c: bool(abs(c[0]) > start_x or c[1] < start_y))

        def profile(coord: Float64Array, time: float | None, node_id: int, elem_id: int | None) -> Float64Array:
            penetration = np.maximum([abs(coord[0]) - start_x, start_y - coord[1]], 0.) / 2.
            return 8. * penetration**2

        model.add_physics(SGPhysicsModel.from_formula(
            "sides and bottom", int(seismic.SGPhysicsTypeEnum.PML), mesh, 1, elements, profile,
        ))
    model.prepare()
    probe = SGReceiverData(0, np.array([5], dtype=np.int32), np.array([station]),
                           np.array([0]), np.array([1.]), 1, 0, -1, 1, 8)
    trace: list[float] = []
    solver.run([probe], lambda _, frame, event, control: trace.append(float(frame.item())), sync=True)
    return np.asarray(trace)


def test_surface_source_half_space(record_property: Callable[[str, float], None]) -> None:
    reference = surface_trace(12., 12., False)
    free_box = surface_trace(4.5, 4.5, False)
    layer = surface_trace(4.5, 4.5, True)
    times = np.arange(reference.size) * .05
    amplitude = np.max(np.abs(reference[times < 3.]))
    # Includes body-wave and surface-wave returns from both sides and the bottom.
    late = times >= 3.
    record_property("pml_relative_echo", float(np.max(np.abs(layer[late] - reference[late])) / amplitude))
    record_property("free_relative_echo", float(np.max(np.abs(free_box[late] - reference[late])) / amplitude))
    assert np.max(np.abs(free_box[late] - reference[late])) > .1 * amplitude
    assert np.max(np.abs(layer[late] - reference[late])) < .03 * amplitude


def test_compact_history_restart() -> None:
    # Two spatially varying profiles are supplied in opposite entity orders.
    def make_solver(start: float, end: float, steps: int) -> seismic.SGSolver:
        solver = corner_model(seismic.SGSolverConfig(t_begin=start, t_end=end, steps_count=steps,
                                                     steps_per_tact=4, kim_beta=1./27.))
        samples = np.linspace(0., 1., 18)
        solver.physics[2] = SGPhysicsData(int(seismic.SGPhysicsTypeEnum.PML), 1, np.array([1, 0]),
                                         np.vstack([samples, np.zeros(18)]))
        solver.physics[3] = SGPhysicsData(int(seismic.SGPhysicsTypeEnum.PML), 1, np.array([0, 1]),
                                         np.vstack([np.zeros(18), 2. - samples]))
        solver.prepare()
        return solver

    continuous = make_solver(0., .02, 16)
    first = make_solver(0., .01, 8)
    initial = np.random.default_rng(102).normal(scale=.02, size=first.solutions[0].shape)
    for solver in (continuous, first):
        solver.solutions[0][:, :] = initial
        solver.run([], lambda *_: None, sync=True)
    assert first.pml_history is not None and continuous.pml_history is not None
    assert first.pml_history.shape == (4, 18)
    assert np.all(np.max(np.abs(first.pml_history[:, :]), axis=1) > 1e-5)
    second = make_solver(.01, .02, 8)
    assert second.pml_history is not None
    np.testing.assert_array_equal(first.pml_entity_ids, [0, 1])
    np.testing.assert_array_equal(second.pml_entity_ids, first.pml_entity_ids)
    second.solutions[0][:, :] = first.solutions[0][:, :]
    second.pml_history[:, :] = first.pml_history[:, :]
    second.run([], lambda *_: None, sync=True)
    np.testing.assert_allclose(second.solutions[0][:, :], continuous.solutions[0][:, :], atol=1e-12)
    np.testing.assert_allclose(second.pml_history[:, :], continuous.pml_history[:, :], atol=1e-13)


def test_reprepare_uses_rounded_step_for_overlapping_pml_history() -> None:
    config = seismic.SGSolverConfig(t_end=.024, step_dt=.009, tacts_count=1, max_steps=1)
    solver = corner_model(config)
    # Element-local Q values include zero and a stretch so small that exp(-d*h) rounds to 1.
    q = np.array([0., 1e-320, 1e-16, .02, .03, .04, .05, .06, .07])
    first = np.vstack((np.r_[2*q, 3*q], np.r_[4*q, q]))
    second = np.vstack((np.r_[q, 6*q], np.r_[2*q, 3*q]))
    code = int(seismic.SGPhysicsTypeEnum.PML)
    solver.physics[2] = SGPhysicsData(code, 1, np.array([1, 0]), first.copy())
    solver.physics[3] = SGPhysicsData(code, 1, np.array([0, 1]), second.copy())

    # In canonical element order [0, 1], sum both profiles before exponentiation.
    stretches = np.vstack((np.r_[4*q, 8*q], np.r_[3*q, 7*q]))
    history_d = np.repeat(stretches, 2, axis=0)
    initial_history = np.linspace(-.2, .4, history_d.size).reshape(history_d.shape)
    gradient = np.array([[.3, -.2], [.1, .4]])
    history_gradient = gradient.T.reshape(4, 1)

    for requested, h in ((.009, .008), (.0061, .006)):
        solver.config.step_dt = requested
        solver.prepare()
        assert isinstance(solver.spectrum, seismic.SGSpectrumData)
        assert solver.spectrum.step_dt == pytest.approx(h, rel=0., abs=1e-16)
        np.testing.assert_array_equal(solver.pml_entity_ids, [0, 1])
        assert solver.pml_history is not None
        np.testing.assert_array_equal(solver.pml_history[:, :], np.zeros_like(initial_history))
        solver.pml_history[:, :] = initial_history
        solver.solutions[0][:, :] = np.zeros(solver.solutions[0].shape)
        solver.solutions[0][:2, :] = gradient @ solver.geometries[0][:2, :]

        # STEP_LIMIT saves the state after the first internal step.
        with pytest.raises(seismic.SGRunError) as error:
            solver.run([], lambda *_: None, sync=True)
        assert error.value.code == seismic.SGRunCodeEnum.STEP_LIMIT
        a = np.exp(-history_d * h)
        b = np.full_like(history_d, h)
        np.divide(-np.expm1(-history_d * h), history_d, out=b, where=history_d * h >= 1e-16)
        expected = a * initial_history + b * history_gradient
        np.testing.assert_allclose(solver.pml_history[:, :], expected, rtol=0., atol=2e-14)
        np.testing.assert_array_equal(solver.physics[2].values, first)
        np.testing.assert_array_equal(solver.physics[3].values, second)
        solver.reset()


def test_zero_profile_preserves_cfl_and_solution() -> None:
    cfg = seismic.SGSolverConfig(t_end=.05, cfl_safety=.4, tacts_count=1)
    ordinary = corner_model(cfg)
    layer = corner_model(cfg)
    layer.physics[2] = strip(0., 0.)
    for solver in (ordinary, layer):
        solver.prepare()
        solver.solutions[0][1, 0] = .25
        solver.run([], lambda *_: None, sync=True)
    assert isinstance(ordinary.spectrum, seismic.SGSpectrumData) and isinstance(layer.spectrum, seismic.SGSpectrumData)
    assert ordinary.spectrum.steps_count == layer.spectrum.steps_count
    np.testing.assert_allclose(layer.solutions[0][:, :], ordinary.solutions[0][:, :], atol=1e-12)
