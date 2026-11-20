"""Typed private boundary of the C++/CUDA solver."""

from enum import IntEnum, auto
from typing import Callable, Final, Mapping, Sequence

from semgpu2.solvers.solver_api import Float64Array, PhysicsId, SGPETData, SGPQSpaceData, SGPhysicsData, SGReceiverData, SpaceId
from .constants import SGSolverConfig, SGSpectrumData

SOLVER_ID: Final[str]

class NativeCompiledState:
    @property
    def n_count(self) -> int: ...
    @property
    def meshed(self) -> bool: ...
    @property
    def solutions(self) -> Mapping[SpaceId, StateArrayAdapter]: ...
    @property
    def geometries(self) -> Mapping[SpaceId, StateArrayAdapter]: ...
    @property
    def materials(self) -> Mapping[SpaceId, StateArrayAdapter]: ...

class StateArrayAdapter:
    @property
    def shape(self) -> tuple[int, int]: ...
    def read(self, row_begin: int, row_count: int, row_step: int, column_begin: int, column_count: int, column_step: int) -> Float64Array: ...
    def write(self, row_begin: int, row_count: int, row_step: int, column_begin: int, column_count: int, column_step: int, values: Float64Array) -> None: ...

def compile_structure(
    n_count: int, pets: SGPETData, pq_spaces: Mapping[SpaceId, SGPQSpaceData],
) -> NativeCompiledState: ...

def validate_structure(
    n_count: int, pets: SGPETData, pq_spaces: Mapping[SpaceId, SGPQSpaceData],
) -> None: ...

def _validate_state_range(
    rows: int, columns: int, row_begin: int, row_count: int, row_step: int,
    column_begin: int, column_count: int, column_step: int,
) -> None: ...

def mesh_geometry(state: NativeCompiledState, n_coords: Float64Array) -> None: ...

def empty_spectrum(state: NativeCompiledState) -> SGSpectrumData: ...

class PreparedState:
    @property
    def spectrum(self) -> SGSpectrumData: ...

class ReceiverPackages: ...

def prepare_model(
    state: NativeCompiledState, physics: Mapping[PhysicsId, SGPhysicsData],
    config: SGSolverConfig,
) -> PreparedState: ...

def prepare_receivers(
    state: NativeCompiledState, receivers: Sequence[SGReceiverData],
) -> ReceiverPackages: ...

class Handle:
    def save(self) -> None: ...
    def complete(self, code: int) -> None: ...
    def abort(self, code: int) -> None: ...
    def _handler_failure(self) -> None: ...
    def wait(self) -> tuple[int, int, float, int, bool, bool, str]: ...

class Session:
    def __init__(
        self, prepared: PreparedState,
        receivers: ReceiverPackages | None,
        callback: Callable[[int, float, int, int, int, Float64Array], None],
        finished: Callable[[bool, bool], None],
    ) -> None: ...
    def handle(self) -> Handle: ...
    def start(self, sync: bool) -> None: ...

THREADS_MIN: Final[int]
THREADS_MAX: Final[int]
THREADS_PER_BLOCK: Final[int]
LANCZOS_STEPS: Final[int]
LANCZOS_MAX_BLOCKS: Final[int]
SPECTRUM_BISECTION_STEPS: Final[int]
LANCZOS_BREAKDOWN_TOL: Final[float]
SPECTRUM_STURM_TOL: Final[float]
DEFAULT_T_BEGIN: Final[float]
DEFAULT_T_END: Final[float]
DEFAULT_KIM_BETA: Final[float]
DEFAULT_KIM_GAMMA: Final[float]


class SGSpaceEnum(IntEnum):
    N = auto()
    MAIN = auto()
    BOUNDARY = auto()


class SGPhysicsTypeEnum(IntEnum):
    MATERIAL_ISO_PLANE_STRAIN = 11000
    N_FORCE_DYNAMIC = 20200
    N_MASS_DAMPING_STATIC = 30001
    ABC = 32000




class SGReceiverComponentEnum(IntEnum):
    VX = 4
    VY = 5


class SGNSolutionComponentEnum(IntEnum):
    UX = auto()
    UY = auto()
    VX = auto()
    VY = auto()
    AX = auto()
    AY = auto()


class SGNGeometryComponentEnum(IntEnum):
    X = auto()
    Y = auto()
    SIZE = auto()


class SGNMaterialsComponentEnum(IntEnum):
    MASS = auto()


class SGNResponseComponentEnum(IntEnum):
    FORCE_X = 0
    FORCE_Y = 1


class SGQMainGeometryComponentEnum(IntEnum):
    SIZE = auto()
    DIFGEOMETRY_XX = auto()
    DIFGEOMETRY_XY = auto()
    DIFGEOMETRY_YX = auto()
    DIFGEOMETRY_YY = auto()


class SGQMainMaterialsComponentEnum(IntEnum):
    DENSITY = 0
    LAMBDA = 1
    MU = 2




class SGQBoundaryGeometryComponentEnum(IntEnum):
    SIZE = auto()
    NORMAL_X = auto()
    NORMAL_Y = auto()


class SGCompileErrorCodeEnum(IntEnum):
    STRUCTURE_FAILURE = auto()
    DEVICE_FAILURE = auto()


class SGMeshErrorCodeEnum(IntEnum):
    GEOMETRY_FAILURE = auto()
    DEVICE_FAILURE = auto()


class SGPrepareErrorCodeEnum(IntEnum):
    PHYSICS_FAILURE = auto()
    SPECTRUM_FAILURE = auto()
    DEVICE_FAILURE = auto()


class SGRunCodeEnum(IntEnum):
    TIME_END = auto()
    UNDERCONFIGURED = auto()
    NUMERICAL_DIVERGENCE = auto()
    DEVICE_FAILURE = auto()
    HANDLER_FAILURE = auto()
    STEP_LIMIT = auto()
