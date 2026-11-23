"""Native CUDA 2D seismic Newmark–Kim codes and public data types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from semgpu2.solvers.common.seismic_fixed_types import (
    SGCompileError,
    SGFixedSolverConfig,
    SGMeshError,
    SGPrepareError,
    SGReceiverEventData,
    SGRunError,
    SGRunResultData,
    SGSpectrumData,
)

from ._native import (
    SHOT_COUNT as SHOT_COUNT,
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
    SGCompileErrorCodeEnum as SGCompileErrorCodeEnum,
    SGMeshErrorCodeEnum as SGMeshErrorCodeEnum,
    SGPrepareErrorCodeEnum as SGPrepareErrorCodeEnum,
    SGRunCodeEnum as SGRunCodeEnum,
)

Float = np.float64


@dataclass(slots=True)
class SGSolverConfig(SGFixedSolverConfig):
    kim_beta: float = DEFAULT_KIM_BETA
    kim_gamma: float = DEFAULT_KIM_GAMMA
    check_finite_each_step: bool = False


__all__ = [
    "Float",
    "SHOT_COUNT",
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
