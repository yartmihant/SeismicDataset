"""Shared Python DTOs and errors for fixed-order seismic CUDA solvers."""

from __future__ import annotations

from dataclasses import dataclass

from semgpu2.solvers.common.time_grid import SGDynamicSolverConfig
from semgpu2.solvers.solver_api import Float64Array, SGBaseError, SGBaseReceiverEventData, SGBaseRunResultData, SGBaseSpectrumData


class SGFixedSolverConfig(SGDynamicSolverConfig):
    """Fields supplied by the concrete solver configuration and its native defaults."""

    __slots__ = ()
    kim_beta: float
    kim_gamma: float
    check_finite_each_step: bool


@dataclass(frozen=True, slots=True)
class SGSpectrumData(SGBaseSpectrumData):
    lambda_max_by_main_element: Float64Array
    lambda_max: float = 0.0
    step_dt: float = 0.0
    tact_dt: float = 0.0
    tacts_count: int = 0
    steps_count: int = 0
    steps_per_tact: int = 0
    lambda_max_damping: float = 0.0


@dataclass(frozen=True, slots=True)
class SGReceiverEventData(SGBaseReceiverEventData):
    current_time: float
    steps_count: int = 0


@dataclass(frozen=True, slots=True)
class SGRunResultData(SGBaseRunResultData):
    tact: int
    current_time: float
    steps_count: int = 0


class SGCompileError(SGBaseError):
    pass


class SGMeshError(SGBaseError):
    pass


class SGPrepareError(SGBaseError):
    pass


class SGRunError(SGBaseError):
    tact: int
    current_time: float
    steps_count: int

    def __init__(self, code: int, message: str, tact: int, current_time: float, steps_count: int = 0) -> None:
        self.tact = tact
        self.current_time = current_time
        self.steps_count = steps_count
        super().__init__(code, message)
