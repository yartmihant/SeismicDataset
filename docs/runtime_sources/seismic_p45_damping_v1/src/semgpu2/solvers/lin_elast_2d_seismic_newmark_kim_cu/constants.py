"""Native CUDA 2D seismic Newmark–Kim codes and public data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semgpu2.solvers.solver_api import (
    Float64Array,
    SGBaseError,
    SGBaseReceiverEventData,
    SGBaseRunResultData,
    SGBaseSpectrumData,
)

from semgpu2.solvers.common.time_grid import SGDynamicSolverConfig

from ._native import (
    SOLVER_ID as SOLVER_ID,
    THREADS_MIN as THREADS_MIN,
    THREADS_MAX as THREADS_MAX,
    THREADS_PER_BLOCK as THREADS_PER_BLOCK,
    LANCZOS_STEPS as LANCZOS_STEPS,
    LANCZOS_MAX_BLOCKS as LANCZOS_MAX_BLOCKS,
    SPECTRUM_BISECTION_STEPS as SPECTRUM_BISECTION_STEPS,
    LANCZOS_BREAKDOWN_TOL as LANCZOS_BREAKDOWN_TOL,
    SPECTRUM_STURM_TOL as SPECTRUM_STURM_TOL,
    DEFAULT_T_BEGIN as DEFAULT_T_BEGIN,
    DEFAULT_T_END as DEFAULT_T_END,
    DEFAULT_KIM_BETA as DEFAULT_KIM_BETA,
    DEFAULT_KIM_GAMMA as DEFAULT_KIM_GAMMA,
    SGSpaceEnum as SGSpaceEnum,
    SGPhysicsTypeEnum as SGPhysicsTypeEnum,
    SGReceiverComponentEnum as SGReceiverComponentEnum,
    SGNSolutionComponentEnum as SGNSolutionComponentEnum,
    SGNGeometryComponentEnum as SGNGeometryComponentEnum,
    SGNMaterialsComponentEnum as SGNMaterialsComponentEnum,
    SGNResponseComponentEnum as SGNResponseComponentEnum,
    SGQMainGeometryComponentEnum as SGQMainGeometryComponentEnum,
    SGQMainMaterialsComponentEnum as SGQMainMaterialsComponentEnum,
    SGQBoundaryGeometryComponentEnum as SGQBoundaryGeometryComponentEnum,
    SGCompileErrorCodeEnum as SGCompileErrorCodeEnum,
    SGMeshErrorCodeEnum as SGMeshErrorCodeEnum,
    SGPrepareErrorCodeEnum as SGPrepareErrorCodeEnum,
    SGRunCodeEnum as SGRunCodeEnum,
)

Float = np.float64


@dataclass(slots=True)
class SGSolverConfig(SGDynamicSolverConfig):
    kim_beta: float = DEFAULT_KIM_BETA
    kim_gamma: float = DEFAULT_KIM_GAMMA
    check_finite_each_step: bool = False


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


__all__ = [
    "Float",
    "SOLVER_ID",
    "SGCompileError",
    "SGCompileErrorCodeEnum",
    "SGMeshError",
    "SGMeshErrorCodeEnum",
    "SGNGeometryComponentEnum",
    "SGNMaterialsComponentEnum",
    "SGNResponseComponentEnum",
    "SGNSolutionComponentEnum",
    "SGPhysicsTypeEnum",
    "SGPrepareError",
    "SGPrepareErrorCodeEnum",
    "SGQBoundaryGeometryComponentEnum",
    "SGQMainGeometryComponentEnum",
    "SGQMainMaterialsComponentEnum",
    "SGReceiverComponentEnum",
    "SGReceiverEventData",
    "SGRunCodeEnum",
    "SGRunError",
    "SGRunResultData",
    "SGSolverConfig",
    "SGSpectrumData",
    "SGSpaceEnum",
    "THREADS_PER_BLOCK",
]
