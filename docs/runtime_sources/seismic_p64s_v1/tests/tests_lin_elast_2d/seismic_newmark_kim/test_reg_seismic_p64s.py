"""SEMGPU Test Card

id
    package: semgpu2
    suite: lin_elast_2d_seismic_newmark_kim_p64s_cu
    test_name: seismic_p64s
    level: reg
    tags: [cuda, fp64, p6, four-shot, pml, source, receiver]

Назначение
    Сравнить четыре опыта на полной сетке с соответствующими опытами p67s.

Постановка
    Криволинейная сетка p6/h10, переменный материал, PML вблизи источников.
    Узловой приёмник снимает восемь компонент скорости на семи позициях.

Машинный критерий
    Первая ненулевая скорость каждого опыта локализована в заданном узле.
    Совпадают кадры, все 24 строки решения и 16 строк истории слоя.
    Попытка зарегистрировать компоненту пятого опыта отклоняется.

Артефакты
    pytest report

Запуск
    targeted: .venv/bin/python -m pytest semgpu2/tests/tests_lin_elast_2d/seismic_newmark_kim/test_reg_seismic_p64s.py
"""
from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from semgpu2.model import SGMeshModel
from semgpu2.solvers import lin_elast_2d_seismic_newmark_kim_p64s_cu as p64s
from semgpu2.solvers import lin_elast_2d_seismic_newmark_kim_p67s_cu as p67s
from semgpu2.solvers.solver_api import Float64Array, SGReceiverData, SGReceiverMainHandler
from tests_lin_elast_2d.seismic_newmark_kim.test_reg_seismic_p67s import _topology, material, mesh, pml


def prepare(solver: p64s.SGSolver | p67s.SGSolver, domain: SGMeshModel) -> None:
    solver.compile(len(domain.nodes_coords), domain.build_pets_data(), {1: domain.build_pq_space_data(1)})
    solver.mesh(domain.build_n_coords())
    solver.physics[0] = material()
    solver.physics[1] = pml()
    solver.prepare()


def test_four_sources_match_first_four_p67s_layers() -> None:
    assert p64s.SOLVER_ID == "lin_elast_2d_seismic_newmark_kim_p64s_cu"
    assert p64s.SHOT_COUNT == 4 and p67s.SHOT_COUNT == 7
    assert len(p64s.SGNSolutionComponentEnum) == 24
    assert len(p64s.SGNResponseComponentEnum) == 8
    codes = np.array(list(p64s.SGReceiverComponentEnum), dtype=np.int32)
    np.testing.assert_array_equal(codes, [4, 5, 10, 11, 16, 17, 22, 23])
    domain = mesh(_topology())
    nodes = np.array([270, 390, 510, 630, 750, 870, 990], dtype=np.int64)
    np.testing.assert_allclose(domain.nodes_coords[nodes[:4], 0], [100., 300., 500., 700.])
    receiver = SGReceiverData(0, codes, nodes, np.arange(7, dtype=np.int64), np.ones(7), 7, 0, -1, 1, 2)
    four = p64s.SGSolver(p64s.SGSolverConfig(t_begin=.048, t_end=.0484, steps_count=4, steps_per_tact=1, kim_beta=1/27))
    seven = p67s.SGSolver(p67s.SGSolverConfig(t_begin=.048, t_end=.0484, steps_count=4, steps_per_tact=1, kim_beta=1/27))
    traces: list[list[Float64Array]] = []
    for solver in (four, seven):
        prepare(solver, domain)
        if solver is four:
            invalid = replace(receiver, comp_codes=np.array([p67s.SGReceiverComponentEnum.VX_S5], dtype=np.int32))
            with pytest.raises(ValueError, match="shot velocity"):
                four.run([invalid], lambda *_: None, sync=True)
        frames: list[Float64Array] = []
        handler = cast(SGReceiverMainHandler, lambda _, frame, event, control: frames.append(frame.copy()))
        assert solver.run([receiver], handler, sync=True).code == p64s.SGRunCodeEnum.TIME_END
        traces.append(frames)
    assert len(traces[0]) == len(traces[1]) == 5
    first_vy = traces[0][1].reshape(4, 2, 7)[:, 1]
    np.testing.assert_array_equal(first_vy != 0., np.eye(4, 7, dtype=bool))
    assert np.all(first_vy.diagonal() < 0.)
    np.testing.assert_allclose(traces[0], traces[1], rtol=1.e-10, atol=1.e-23)
    assert four.solutions[0].shape == (24, len(domain.nodes_coords))
    np.testing.assert_allclose(four.solutions[0][:, :], seven.solutions[0][:24, :], rtol=1.e-10, atol=1.e-19)
    assert four.pml_history is not None and seven.pml_history is not None
    assert four.pml_history.shape == (16, seven.pml_history.shape[1])
    assert np.any(four.pml_history[:, :])
    np.testing.assert_allclose(four.pml_history[:, :], seven.pml_history[:16, :], rtol=1.e-10, atol=1.e-23)


def test_p64s_cfl_and_restart() -> None:
    domain = mesh(_topology())
    full = p64s.SGSolver(p64s.SGSolverConfig(t_begin=.048, t_end=.052, tacts_count=2, cfl_safety=.8, kim_beta=1/27))
    seven = p67s.SGSolver(p67s.SGSolverConfig(t_begin=.048, t_end=.052, tacts_count=2, cfl_safety=.8, kim_beta=1/27))
    for solver in (full, seven):
        prepare(solver, domain)
    assert isinstance(full.spectrum, p64s.SGSpectrumData) and isinstance(seven.spectrum, p67s.SGSpectrumData)
    assert full.spectrum.step_dt == seven.spectrum.step_dt
    steps = full.spectrum.steps_count // 2
    assert steps > 1 and 2 * steps == full.spectrum.steps_count
    assert full.run([], lambda *_: None, sync=True).code == p64s.SGRunCodeEnum.TIME_END

    first = p64s.SGSolver(p64s.SGSolverConfig(t_begin=.048, t_end=.050, steps_count=steps, steps_per_tact=steps, kim_beta=1/27))
    second = p64s.SGSolver(p64s.SGSolverConfig(t_begin=.050, t_end=.052, steps_count=steps, steps_per_tact=steps, kim_beta=1/27))
    for part in (first, second):
        prepare(part, domain)
    assert first.run([], lambda *_: None, sync=True).code == p64s.SGRunCodeEnum.TIME_END
    assert first.pml_history is not None and second.pml_history is not None and full.pml_history is not None
    second.solutions[0][:, :] = first.solutions[0][:, :]
    second.pml_history[:, :] = first.pml_history[:, :]
    assert second.run([], lambda *_: None, sync=True).code == p64s.SGRunCodeEnum.TIME_END
    np.testing.assert_allclose(second.solutions[0][:, :], full.solutions[0][:, :], rtol=1.e-10, atol=1.e-18)
    np.testing.assert_allclose(second.pml_history[:, :], full.pml_history[:, :], rtol=1.e-10, atol=1.e-23)
