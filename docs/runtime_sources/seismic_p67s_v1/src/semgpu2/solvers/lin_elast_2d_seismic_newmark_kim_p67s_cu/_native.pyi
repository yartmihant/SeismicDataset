from enum import IntEnum
from typing import Final

from semgpu2.solvers.common.seismic_fixed_native import (
    DEFAULT_KIM_BETA as DEFAULT_KIM_BETA,
    DEFAULT_KIM_GAMMA as DEFAULT_KIM_GAMMA,
    DEFAULT_T_BEGIN as DEFAULT_T_BEGIN,
    DEFAULT_T_END as DEFAULT_T_END,
    Handle as Handle,
    LANCZOS_BREAKDOWN_TOL as LANCZOS_BREAKDOWN_TOL,
    LANCZOS_MAX_BLOCKS as LANCZOS_MAX_BLOCKS,
    LANCZOS_STEPS as LANCZOS_STEPS,
    NativeCompiledState as NativeCompiledState,
    PreparedState as PreparedState,
    ReceiverPackages as ReceiverPackages,
    SGCompileErrorCodeEnum as SGCompileErrorCodeEnum,
    SGMeshErrorCodeEnum as SGMeshErrorCodeEnum,
    SGNGeometryComponentEnum as SGNGeometryComponentEnum,
    SGNMaterialsComponentEnum as SGNMaterialsComponentEnum,
    SGPrepareErrorCodeEnum as SGPrepareErrorCodeEnum,
    SGQMainGeometryComponentEnum as SGQMainGeometryComponentEnum,
    SGQMainMaterialsComponentEnum as SGQMainMaterialsComponentEnum,
    SGRunCodeEnum as SGRunCodeEnum,
    SOLVER_ID as SOLVER_ID,
    SPECTRUM_BISECTION_STEPS as SPECTRUM_BISECTION_STEPS,
    SPECTRUM_STURM_TOL as SPECTRUM_STURM_TOL,
    Session as Session,
    StateArrayAdapter as StateArrayAdapter,
    THREADS_MAX as THREADS_MAX,
    THREADS_MIN as THREADS_MIN,
    THREADS_PER_BLOCK as THREADS_PER_BLOCK,
    _validate_state_range as _validate_state_range,
    compile_structure as compile_structure,
    empty_spectrum as empty_spectrum,
    mesh_geometry as mesh_geometry,
    prepare_model as prepare_model,
    prepare_receivers as prepare_receivers,
    validate_structure as validate_structure,
)

SHOT_COUNT: Final[int]

class SGSpaceEnum(IntEnum):
    N = 0
    MAIN = 1

class SGPhysicsTypeEnum(IntEnum):
    MATERIAL_ISO_PLANE_STRAIN = 11000
    PML = 31100

class SGReceiverComponentEnum(IntEnum):
    VX = 4
    VY = 5
    VX_S2 = 10
    VY_S2 = 11
    VX_S3 = 16
    VY_S3 = 17
    VX_S4 = 22
    VY_S4 = 23
    VX_S5 = 28
    VY_S5 = 29
    VX_S6 = 34
    VY_S6 = 35
    VX_S7 = 40
    VY_S7 = 41

class SGNSolutionComponentEnum(IntEnum):
    UX = 0
    UY = 1
    VX = 2
    VY = 3
    AX = 4
    AY = 5
    UX_S2 = 6
    UY_S2 = 7
    VX_S2 = 8
    VY_S2 = 9
    AX_S2 = 10
    AY_S2 = 11
    UX_S3 = 12
    UY_S3 = 13
    VX_S3 = 14
    VY_S3 = 15
    AX_S3 = 16
    AY_S3 = 17
    UX_S4 = 18
    UY_S4 = 19
    VX_S4 = 20
    VY_S4 = 21
    AX_S4 = 22
    AY_S4 = 23
    UX_S5 = 24
    UY_S5 = 25
    VX_S5 = 26
    VY_S5 = 27
    AX_S5 = 28
    AY_S5 = 29
    UX_S6 = 30
    UY_S6 = 31
    VX_S6 = 32
    VY_S6 = 33
    AX_S6 = 34
    AY_S6 = 35
    UX_S7 = 36
    UY_S7 = 37
    VX_S7 = 38
    VY_S7 = 39
    AX_S7 = 40
    AY_S7 = 41

class SGNResponseComponentEnum(IntEnum):
    FORCE_X = 0
    FORCE_Y = 1
    FORCE_X_S2 = 2
    FORCE_Y_S2 = 3
    FORCE_X_S3 = 4
    FORCE_Y_S3 = 5
    FORCE_X_S4 = 6
    FORCE_Y_S4 = 7
    FORCE_X_S5 = 8
    FORCE_Y_S5 = 9
    FORCE_X_S6 = 10
    FORCE_Y_S6 = 11
    FORCE_X_S7 = 12
    FORCE_Y_S7 = 13
