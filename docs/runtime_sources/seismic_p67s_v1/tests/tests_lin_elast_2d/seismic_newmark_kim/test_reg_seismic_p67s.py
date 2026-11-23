"""SEMGPU Test Card

id
    package: semgpu2
    suite: lin_elast_2d_seismic_newmark_kim_p67s_cu
    test_name: seismic_p67s
    level: reg
    tags: [cuda, fp64, p6, seven-shot, pml, source, save, receiver]

Назначение
    Зафиксировать p67s как семь p6-задач со встроенным источником на
    фиксированной верхней полосе p6/h10.

Постановка
    Тест сам строит компактную полосу шириной 210 p6-элементов. Верхняя строка
    имеет номера 0..1260; источники расположены в 315 + 105*s. Обычный p6
    получает независимую таблицу Рикера, p67s — только материал и PML.

Машинный критерий
    Семь слоёв совпадают с последовательными p6-эталонами. Покрыты фаза с
    ненулевым t_begin, reprepare с округлённым dt, SAVE, PML-history,
    смешанный MAIN-приёмник, внешняя сила и фиксированная геометрия.
    При остановке посреди такта состояние и история сравниваются с шестью
    отдельными шагами. Изменённые после prepare массы проверяются по p6,
    а непредставимая обратная масса — по нулевому ускорению свободного от силы узла.

Артефакты
    pytest report

Запуск
    targeted: .venv/bin/python -m pytest semgpu2/tests/tests_lin_elast_2d/seismic_newmark_kim/test_reg_seismic_p67s.py
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pytest

from semgpu2.model import SGMeshModel, SGMeshStratumData, SGPETALineModel
from semgpu2.solvers import lin_elast_2d_seismic_newmark_kim_p67s_cu as p67s
from semgpu2.solvers import lin_elast_2d_seismic_newmark_kim_p6_cu as p6
from semgpu2.solvers.solver_api import Float64Array, SGBaseReceiverEventData, SGPhysicsData, SGReceiverData, SGReceiverMainHandler, SGReceiverRunControl
from semgpu2.tools.mesh.numbering import with_leading_nodes


RTOL, ATOL = 1.e-10, 1.e-12
ORDER, SHOTS = 6, 7
Q = 49
NX, NY = 210, 3
NODES_X, NODES_Y = NX * ORDER + 1, NY * ORDER + 1
N_COUNT, TOP_FIRST = NODES_X * NODES_Y, 0
SOURCE_NODES = np.asarray([315 + 105 * shot for shot in range(SHOTS)], dtype=np.int64)
SOURCE_X = np.arange(175., 1226., 175.)


@dataclass(frozen=True, slots=True)
class _Topology:
    coords: np.ndarray
    tags: np.ndarray
    p_nodes: np.ndarray
    offsets: np.ndarray
    centers: np.ndarray


def _topology() -> _Topology:
    """Target h10 topology, deliberately self-contained and mapped below y=0."""
    pet = SGPETALineModel("gll-p6", order=ORDER)
    gll = (pet.p_nodes.coords[:, 0] + 1.) / 2.
    x = np.append(-350. + (np.arange(NX)[:, None] * 10. + 10. * gll[:-1]).reshape(-1), 1750.)
    y = np.append(-NY * 10. + (np.arange(NY)[:, None] * 10. + 10. * gll[:-1]).reshape(-1), 0.)[::-1]
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack((xx.ravel(), yy.ravel())).astype(np.float64, copy=False)
    depth = -coords[:, 1] / (NY * 10.)
    coords[:, 0] += .025 * depth * np.sin((coords[:, 0] + 350.) * np.pi / 2100.)
    coords[:, 1] -= .015 * depth * (1. - depth) * np.cos((coords[:, 0] + 350.) * np.pi / 2100.)
    ex, ey = np.meshgrid(np.arange(NX), np.arange(NY))
    lx, ly = np.meshgrid(np.arange(7), np.arange(7))
    # N storage is top-to-bottom, whereas each PET's reference y is bottom-to-top.
    p_nodes = ((NY * 6 - (ey.ravel()[:, None, None] * 6 + ly[None])) * NODES_X + ex.ravel()[:, None, None] * 6 + lx[None]).reshape(-1).astype(np.int64)
    elements = NX * NY
    return _Topology(coords, np.zeros((N_COUNT, 3), dtype=np.int64), p_nodes,
                     np.arange(elements + 1, dtype=np.int64) * Q,
                     np.column_stack((-350. + (ex.ravel() + .5) * 10., -NY * 10. + (ey.ravel() + .5) * 10.)).astype(np.float64))


@pytest.fixture(scope="module")
def target() -> _Topology:
    result = _topology()
    assert result.coords.shape == (N_COUNT, 2)
    gll = (SGPETALineModel("gll-p6", order=ORDER).p_nodes.coords[:, 0] + 1.) / 2.
    expected_x = np.append(-350. + (np.arange(NX)[:, None] * 10. + 10. * gll[:-1]).reshape(-1), 1750.)
    np.testing.assert_allclose(result.coords[TOP_FIRST:TOP_FIRST + NODES_X, 0], expected_x, rtol=0., atol=1.e-9)
    np.testing.assert_allclose(result.coords[TOP_FIRST:TOP_FIRST + NODES_X, 1], 0., rtol=0., atol=1.e-9)
    np.testing.assert_allclose(result.coords[SOURCE_NODES, 0], SOURCE_X, rtol=0., atol=1.e-9)
    return result


def mesh(topology: _Topology, coords: np.ndarray | None = None) -> SGMeshModel:
    elements = NX * NY
    stratum = SGMeshStratumData((2, 2, 2), topology.p_nodes, topology.offsets,
                                np.zeros(2 * elements, dtype=np.int32), np.arange(0, 2 * elements + 1, 2, dtype=np.int64),
                                np.ones(elements, dtype=np.int64))
    return SGMeshModel(topology.coords if coords is None else coords, topology.tags,
                       [SGPETALineModel("gll-p6", order=ORDER)], {1: stratum}, centers={1: topology.centers})


def material() -> SGPhysicsData:
    ids = np.arange(NX * NY, dtype=np.int64)
    q = np.arange(ids.size * Q, dtype=np.float64)
    values = np.vstack((2200. + .01 * (q % Q), 9.e9 + 1.e5 * q, .25 + 1.e-5 * (q % Q)))
    return SGPhysicsData(11000, 1, ids, values)


def pml() -> SGPhysicsData:
    # Elements containing the source nodes make the PML histories nonzero in T1.
    ids = (NY - 1) * NX + SOURCE_NODES // ORDER
    q = np.arange(ids.size * Q, dtype=np.float64)
    return SGPhysicsData(31100, 1, ids, np.vstack((.1 + .001 * q, .2 + .0015 * q[::-1])))


def configure(solver: p6.SGSolver | p67s.SGSolver, domain: SGMeshModel, with_pml: bool = True) -> None:
    solver.compile(len(domain.nodes_coords), domain.build_pets_data(), {1: domain.build_pq_space_data(1)})
    solver.mesh(domain.build_n_coords())
    solver.physics[0] = material()
    if with_pml:
        solver.physics[1] = pml()


def ricker(times: np.ndarray) -> np.ndarray:
    """Independent acquisition specification: f=30 Hz, t0=.05, Fy=-1."""
    phase = np.pi * 30. * (times - .05)
    return -(1. - 2. * phase**2) * np.exp(-phase**2)


def add_reference_force(solver: p6.SGSolver, node: int, begin: float, end: float, steps: int) -> None:
    values = np.zeros((steps + 1, 2, 1))
    values[:, 1, 0] = ricker(np.linspace(begin, end, steps + 1))
    solver.physics[9] = SGPhysicsData(20200, 0, np.asarray([node]), values, t_begin=begin, t_end=end, interp_code=1)


def config(begin: float, end: float, steps: int) -> p67s.SGSolverConfig:
    return p67s.SGSolverConfig(t_begin=begin, t_end=end, steps_count=steps, steps_per_tact=1, kim_beta=1 / 27, kim_gamma=.5)


def test_p67s_matches_seven_p6_ricker_tables_and_pml(target: _Topology) -> None:
    domain = mesh(target)
    begin, end, steps = .048, .052, 4
    actual = p67s.SGSolver(config(begin, end, steps))
    configure(actual, domain)
    actual.prepare()
    assert actual.pml_history is not None and actual.pml_history.shape == (28, SHOTS * Q)
    assert actual.run([], lambda *_: None, sync=True).code == int(p67s.SGRunCodeEnum.TIME_END)
    actual_history = actual.pml_history[:, :].reshape(SHOTS, 4, -1)
    assert np.all(np.max(np.abs(actual_history), axis=(1, 2)) > 0.)
    for shot, node in enumerate(SOURCE_NODES):
        reference = p6.SGSolver(p6.SGSolverConfig(t_begin=begin, t_end=end, steps_count=steps, steps_per_tact=1, kim_beta=1 / 27, kim_gamma=.5))
        configure(reference, domain)
        add_reference_force(reference, int(node), begin, end, steps)
        reference.prepare()
        assert reference.run([], lambda *_: None, sync=True).code == int(p6.SGRunCodeEnum.TIME_END)
        np.testing.assert_allclose(actual.solutions[0][6 * shot:6 * shot + 6, :], reference.solutions[0][:, :], rtol=RTOL, atol=ATOL)
        assert reference.pml_history is not None
        np.testing.assert_allclose(actual.pml_history[4 * shot:4 * shot + 4, :], reference.pml_history[:, :], rtol=RTOL, atol=ATOL)


def test_p67s_save_reprepare_rounded_dt_and_main_receiver(target: _Topology) -> None:
    domain = mesh(target)
    full = p67s.SGSolver(config(.048, .052, 4))
    configure(full, domain)
    full.prepare()
    assert full.run([], lambda *_: None, sync=True).code == int(p67s.SGRunCodeEnum.TIME_END)
    assert full.pml_history is not None
    expected, expected_history = full.solutions[0][:, :].copy(), full.pml_history[:, :].copy()

    first = p67s.SGSolver(p67s.SGSolverConfig(t_begin=.048, t_end=.05, steps_count=2,
                                                 steps_per_tact=2, kim_beta=1 / 27, kim_gamma=.5))
    configure(first, domain)
    first.prepare()
    local_count = 3 * Q
    entities = pml().entity_ids[:3]
    receiver = SGReceiverData(1, np.asarray([5, 4, 17], dtype=np.int32), entities,
                              np.arange(local_count, dtype=np.int64) % 2,
                              np.linspace(-.75, .5, local_count), 2, 0, -1, 1, 3)
    frames: list[Float64Array] = []
    def save(_: int, frame: Float64Array, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        if event.tact == 1:
            frames.append(frame.copy())
            control.save()
    handler = cast(SGReceiverMainHandler, save)
    assert first.run([receiver], handler, sync=True).code == int(p67s.SGRunCodeEnum.TIME_END)
    assert first.pml_history is not None
    checkpoint, checkpoint_history = first.solutions[0][:, :].copy(), first.pml_history[:, :].copy()
    sampled_nodes = target.p_nodes.reshape(-1, Q)[entities].ravel()
    assert len(frames) == 1
    for component, row in enumerate((3, 2, 15)):
        weighted = checkpoint[row, sampled_nodes] * receiver.sens_weights
        expected_frame = np.bincount(receiver.sens_frame_idxs, weights=weighted, minlength=2)
        np.testing.assert_allclose(frames[0].reshape(3, 2)[component], expected_frame, rtol=RTOL, atol=1.e-22)
    resumed = p67s.SGSolver(config(.05, .052, 2))
    configure(resumed, domain)
    resumed.prepare()
    assert resumed.pml_history is not None
    resumed.solutions[0][:, :], resumed.pml_history[:, :] = checkpoint, checkpoint_history
    assert resumed.run([], lambda *_: None, sync=True).code == int(p67s.SGRunCodeEnum.TIME_END)
    np.testing.assert_allclose(resumed.solutions[0][:, :], expected, rtol=RTOL, atol=ATOL)
    np.testing.assert_allclose(resumed.pml_history[:, :], expected_history, rtol=RTOL, atol=ATOL)

    rounded = p67s.SGSolver(p67s.SGSolverConfig(t_begin=.048, t_end=.052, step_dt=.002, steps_per_tact=1, kim_beta=1 / 27, kim_gamma=.5))
    configure(rounded, domain)
    rounded.prepare()
    assert isinstance(rounded.spectrum, p67s.SGSpectrumData) and rounded.spectrum.steps_count == 2
    rounded.config.step_dt = .0013
    rounded.prepare()
    assert isinstance(rounded.spectrum, p67s.SGSpectrumData) and rounded.spectrum.steps_count == 4
    assert rounded.run([], lambda *_: None, sync=True).code == int(p67s.SGRunCodeEnum.TIME_END)
    np.testing.assert_allclose(rounded.solutions[0][:, :], expected, rtol=RTOL, atol=ATOL)


def test_p67s_rejects_external_force_and_wrong_surface(target: _Topology) -> None:
    domain = mesh(target)
    rejected = p67s.SGSolver(config(.048, .049, 1))
    configure(rejected, domain)
    rejected.physics[9] = SGPhysicsData(20200, 0, SOURCE_NODES[:1], np.zeros((2, 2, 1)), t_begin=.048, t_end=.049, interp_code=1)
    with pytest.raises(p67s.SGPrepareError):
        rejected.prepare()
    # Keep all physical elements intact and move the top row to the end.
    legacy, _ = with_leading_nodes(domain, np.arange(NODES_X, N_COUNT, dtype=np.int64))
    invalid = p67s.SGSolver(config(.048, .049, 1))
    invalid.compile(N_COUNT, legacy.build_pets_data(), {1: legacy.build_pq_space_data(1)})
    with pytest.raises(p67s.SGMeshError, match="upper-surface node 0"):
        invalid.mesh(legacy.build_n_coords())

    # The frozen geometry includes surface nodes that carry no source.
    broken = target.coords.copy()
    broken[1, 0] += 2.e-9
    malformed = p67s.SGSolver(config(.048, .049, 1))
    malformed.compile(N_COUNT, domain.build_pets_data(), {1: domain.build_pq_space_data(1)})
    with pytest.raises(p67s.SGMeshError, match="upper-surface node 1"):
        malformed.mesh(mesh(target, broken).build_n_coords())


def test_p67s_reads_updated_nodal_mass_and_elastic_rows_on_pml_elements(target: _Topology) -> None:
    domain = mesh(target)
    begin, end, steps = .049, .05, 4
    actual = p67s.SGSolver(config(begin, end, steps))
    reference = p6.SGSolver(p6.SGSolverConfig(t_begin=begin, t_end=end, steps_count=steps,
                                               steps_per_tact=1, kim_beta=1 / 27, kim_gamma=.5))
    configure(actual, domain)
    configure(reference, domain)
    add_reference_force(reference, int(SOURCE_NODES[0]), begin, end, steps)
    for solver in (actual, reference):
        solver.prepare()
        node = int(SOURCE_NODES[0])
        solver.materials[0][0, node] = 2. * solver.materials[0][0, node]
        element = int(pml().entity_ids[0])
        columns = slice(Q * element, Q * (element + 1))
        solver.materials[1][1:3, columns] = .5 * solver.materials[1][1:3, columns]
        solver.run([], lambda *_: None, sync=True)
    np.testing.assert_allclose(actual.solutions[0][:6, :], reference.solutions[0][:, :], rtol=RTOL, atol=ATOL)


def test_p67s_cfl_ignores_pml_and_rejects_profile_overflow(target: _Topology) -> None:
    domain = mesh(target)
    spectra = []
    for enabled in (False, True):
        solver = p67s.SGSolver(p67s.SGSolverConfig(t_end=.01, cfl_safety=.8, tacts_count=2))
        configure(solver, domain, with_pml=enabled)
        solver.prepare()
        assert isinstance(solver.spectrum, p67s.SGSpectrumData)
        spectra.append(solver.spectrum)
    assert spectra[0].step_dt == spectra[1].step_dt
    assert spectra[0].lambda_max == pytest.approx(spectra[1].lambda_max, rel=1.e-12)

    profile = pml()
    excessive = SGPhysicsData(31100, 1, profile.entity_ids, np.full_like(profile.values, np.finfo(np.float64).max))
    solver.physics[1] = excessive
    solver.physics[2] = excessive
    with pytest.raises(p67s.SGPrepareError, match="Summed PML"):
        solver.prepare()


def test_p67s_step_limit_keeps_corrected_state(target: _Topology) -> None:
    domain = mesh(target)
    capped = p67s.SGSolver(p67s.SGSolverConfig(t_begin=.048, t_end=.0488, steps_count=8,
                                                 steps_per_tact=4, max_steps=6, kim_beta=1 / 27))
    reference = p67s.SGSolver(p67s.SGSolverConfig(t_begin=.048, t_end=.0486, steps_count=6,
                                                    steps_per_tact=1, check_finite_each_step=True, kim_beta=1 / 27))
    for solver in (capped, reference):
        configure(solver, domain)
        solver.prepare()
    with pytest.raises(p67s.SGRunError) as stopped:
        capped.run([], lambda *_: None, sync=True)
    assert stopped.value.code == p67s.SGRunCodeEnum.STEP_LIMIT
    assert reference.run([], lambda *_: None, sync=True).code == p67s.SGRunCodeEnum.TIME_END
    np.testing.assert_allclose(capped.solutions[0][:, :], reference.solutions[0][:, :], rtol=RTOL, atol=1.e-17)
    assert capped.pml_history is not None and reference.pml_history is not None
    np.testing.assert_allclose(capped.pml_history[:, :], reference.pml_history[:, :], rtol=RTOL, atol=1.e-23)


def test_p67s_unloaded_node_with_unrepresentable_inverse_mass(target: _Topology) -> None:
    solver = p67s.SGSolver(p67s.SGSolverConfig(t_begin=.048, t_end=.0483, steps_count=3,
                                                 steps_per_tact=3, kim_beta=1 / 27))
    configure(solver, mesh(target))
    solver.prepare()
    # No wave reaches node zero in three steps. Its acceleration must stay zero.
    tiny_mass = np.nextafter(0., 1.)
    solver.materials[0][0, 0] = tiny_mass
    assert solver.run([], lambda *_: None, sync=True).code == p67s.SGRunCodeEnum.TIME_END
    assert np.isfinite(solver.solutions[0][:, :]).all()
    np.testing.assert_array_equal(solver.solutions[0][:, 0], 0.)
    assert solver.materials[0][0, 0] == tiny_mass
