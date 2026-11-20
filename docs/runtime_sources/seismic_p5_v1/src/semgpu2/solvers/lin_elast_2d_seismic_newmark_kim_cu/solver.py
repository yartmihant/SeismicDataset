"""SolverAPI facade; preparation, numerical orchestration and FIFO are native."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, overload

import numpy as np

from semgpu2.solvers.common.time_grid import validate_time_config
from semgpu2.solvers.solver_api import (
    Float64Array, SGBaseRunResultData, SGBaseSolver, SGPETData, SGPQSpaceData,
    SGReceiverMainHandler,
    SGReceiverData, SGReceiverRunControl, SGRunHandle, SGSolverLifecycleState,
    SGStateArray, SpaceId,
)

from . import _native
from .constants import (
    SGPrepareError, SGPrepareErrorCodeEnum, SGReceiverEventData, SGRunCodeEnum, SGRunError, SGRunResultData,
    SGSolverConfig, SGSpectrumData,
)


@dataclass(slots=True)
class _CauseBox:
    error: BaseException | None = None


class _RunControl(SGReceiverRunControl):
    def __init__(self, native: _native.Handle) -> None:
        self._native = native

    def save(self) -> None:
        self._native.save()

    def complete(self, code: int) -> None:
        self._native.complete(code)

    def abort(self, code: int) -> None:
        self._native.abort(code)


class _RunHandle(SGRunHandle):
    def __init__(self, native: _native.Handle, cause: _CauseBox) -> None:
        self._native = native
        self._cause = cause

    def abort(self, code: int) -> None:
        self._native.abort(code)

    def wait(self) -> SGBaseRunResultData:
        code, tact, time, steps_count, aborted, failed, message = self._native.wait()
        if failed or aborted:
            error = SGRunError(code, message or "run aborted", tact, time, steps_count)
            if code == int(SGRunCodeEnum.HANDLER_FAILURE) and self._cause.error is not None:
                raise error from self._cause.error
            raise error
        return SGRunResultData(code, tact, time, steps_count)


class _NativeStateArray(SGStateArray):
    """Copying state transport over one native canonical field."""

    def __init__(self, native: _native.StateArrayAdapter) -> None:
        self._native = native

    @property
    def shape(self) -> tuple[int, int]:
        rows, columns = self._native.shape
        return int(rows), int(columns)

    @staticmethod
    def _native_range(index: range) -> tuple[int, int, int]:
        """Represent a Python-selected range within the native int64 ABI."""
        count = len(index)
        if count == 0:
            return 0, 0, 1
        if count == 1:
            return index.start, 1, 1
        return index.start, count, index.step

    def _read(self, rows: range, columns: range) -> Float64Array:
        row_begin, row_count, row_step = self._native_range(rows)
        column_begin, column_count, column_step = self._native_range(columns)
        return self._native.read(
            row_begin, row_count, row_step,
            column_begin, column_count, column_step,
        )

    def _write(self, rows: range, columns: range, values: Float64Array) -> None:
        row_begin, row_count, row_step = self._native_range(rows)
        column_begin, column_count, column_step = self._native_range(columns)
        self._native.write(
            row_begin, row_count, row_step,
            column_begin, column_count, column_step, values,
        )


class SGSolver(SGBaseSolver[SGSolverConfig]):
    def __init__(self, config: SGSolverConfig) -> None:
        super().__init__(config)
        self.spectrum = SGSpectrumData(np.empty(0, dtype=np.float64))
        self.lifecycle = SGSolverLifecycleState.INIT
        self.solutions: Mapping[SpaceId, SGStateArray] = MappingProxyType({})
        self.geometries: Mapping[SpaceId, SGStateArray] = MappingProxyType({})
        self.materials: Mapping[SpaceId, SGStateArray] = MappingProxyType({})
        self._state: _native.NativeCompiledState | None = None
        self._prepared: _native.PreparedState | None = None
        self._session: _native.Session | None = None

    def compile(self, n_count: int, pets: SGPETData, pq_spaces: Mapping[SpaceId, SGPQSpaceData]) -> None:
        if self.lifecycle != SGSolverLifecycleState.INIT:
            raise RuntimeError("compile is allowed only in INIT")
        state = _native.compile_structure(n_count, pets, pq_spaces)
        self._state = state
        self.solutions = MappingProxyType({space: _NativeStateArray(field) for space, field in state.solutions.items()})
        self.geometries = MappingProxyType({space: _NativeStateArray(field) for space, field in state.geometries.items()})
        self.materials = MappingProxyType({space: _NativeStateArray(field) for space, field in state.materials.items()})
        self.spectrum = _native.empty_spectrum(state)
        self.lifecycle = SGSolverLifecycleState.COMPILED

    def mesh(self, n_coords: Float64Array) -> None:
        if self.lifecycle not in (SGSolverLifecycleState.COMPILED, SGSolverLifecycleState.MESHED, SGSolverLifecycleState.READY):
            raise RuntimeError("mesh requires COMPILED, MESHED or READY")
        assert self._state is not None
        self.lifecycle = SGSolverLifecycleState.COMPILED
        self._prepared = None
        _native.mesh_geometry(self._state, n_coords)
        self.lifecycle = SGSolverLifecycleState.MESHED

    def _require_prepare_stage(self) -> None:
        if self.lifecycle not in (SGSolverLifecycleState.MESHED, SGSolverLifecycleState.READY):
            raise RuntimeError("prepare requires MESHED or READY")

    def prepare(self) -> None:
        self._require_prepare_stage()
        assert self._state is not None
        self.lifecycle = SGSolverLifecycleState.MESHED
        self._prepared = None
        try:
            validate_time_config(self.config)
        except ValueError as error:
            raise SGPrepareError(int(SGPrepareErrorCodeEnum.SPECTRUM_FAILURE), str(error)) from error
        prepared = _native.prepare_model(self._state, self.physics, self.config)
        self._prepared = prepared
        self.spectrum = prepared.spectrum
        self.lifecycle = SGSolverLifecycleState.READY

    @overload
    def run(
        self, receivers: Sequence[SGReceiverData], main_handler: SGReceiverMainHandler,
        *, sync: Literal[False] = False,
    ) -> SGRunHandle: ...

    @overload
    def run(
        self, receivers: Sequence[SGReceiverData], main_handler: SGReceiverMainHandler,
        *, sync: Literal[True],
    ) -> SGBaseRunResultData: ...

    def run(
        self, receivers: Sequence[SGReceiverData], main_handler: SGReceiverMainHandler,
        *, sync: bool = False,
    ) -> SGRunHandle | SGBaseRunResultData:
        if self.lifecycle != SGSolverLifecycleState.READY:
            raise RuntimeError("run requires READY")
        assert self._prepared is not None
        assert self._state is not None
        # Argument acceptance precedes the run stage and leaves READY on failure.
        packages = _native.prepare_receivers(self._state, receivers)
        # Native runtime delivers the negative terminal frame and drains the
        # handler FIFO before selecting the default UNDERCONFIGURED outcome.
        cause = _CauseBox()
        native_handle: _native.Handle

        def callback(tact: int, time: float, steps_count: int, event: int, index: int, frame: Float64Array) -> None:
            self.lifecycle = SGSolverLifecycleState.WAITING
            try:
                main_handler(index, frame, SGReceiverEventData(event, tact, time, steps_count), _RunControl(native_handle))
            except BaseException as error:
                if cause.error is None:
                    cause.error = error
                native_handle._handler_failure()
            finally:
                self.lifecycle = SGSolverLifecycleState.RUNNING

        def finished(aborted: bool, failed: bool) -> None:
            self.lifecycle = SGSolverLifecycleState.FAILED if aborted or failed else SGSolverLifecycleState.DONE

        try:
            session = _native.Session(self._prepared, packages, callback, finished)
            native_handle = session.handle()
            handle = _RunHandle(native_handle, cause)
            self._session = session
            self.lifecycle = SGSolverLifecycleState.RUNNING
            session.start(sync)
        except Exception as error:
            self.lifecycle = SGSolverLifecycleState.FAILED
            raise SGRunError(int(SGRunCodeEnum.DEVICE_FAILURE), "Failed to start native run", 0, self.config.t_begin) from error
        if sync:
            return handle.wait()
        return handle

    def reset(self) -> None:
        if self.lifecycle not in (SGSolverLifecycleState.DONE, SGSolverLifecycleState.FAILED):
            raise RuntimeError("reset requires DONE or FAILED")
        self._session = None
        self.lifecycle = SGSolverLifecycleState.READY


__all__ = ["SGSolver"]
