"""SolverAPI facade for the fixed-order CUDA extension."""

from semgpu2.solvers.common.seismic_fixed_solver import FixedOrderSolver

from . import _native
from .constants import SGSolverConfig


class SGSolver(FixedOrderSolver[SGSolverConfig]):
    def __init__(self, config: SGSolverConfig) -> None:
        super().__init__(config, _native)


__all__ = ["SGSolver"]
