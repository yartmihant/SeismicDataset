"""CU-опыт главы 1.4: только Vx/Vy на 70 точных поверхностных узлах."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import hashlib
from importlib.metadata import version
import json
from math import ceil, isfinite, log
import os
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback
from typing import TYPE_CHECKING, Callable, cast

import numpy as np
from numpy.typing import NDArray

from semgpu2.model.model import SGModel
from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.physics import ExperimentSpec, damping_alpha, inclusive_times, nodal_point_force, ricker
from scripts.dev_1_3_point_source.run import TimedResult, abc_edges, checked_time, file_record, write_json
from scripts.dev_1_4_server_preparation.chunk_plan import Dataset, GIB
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
DAMPING_FACTOR = 0.5684038628158912
ALPHA_MAX_S_INV = DAMPING_FACTOR * (-3.0 * 4385.0 * log(0.01) / 175.0)
RECEIVER_X_M = np.arange(5.0, 700.0, 10.0, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SurfaceSpec:
    dataset: Dataset
    model_id: int
    source_x_m: float
    substeps: int
    t_end_s: float = 1.0
    order: int = 4
    kind: str = field(default="curved", init=False)
    backend: str = field(default="cu", init=False)
    alpha_max_s_inv: float = field(default=ALPHA_MAX_S_INV, init=False)

    def __post_init__(self) -> None:
        if self.dataset not in ("ECVB", "EFVB") or isinstance(self.model_id, bool) or not isinstance(self.model_id, int) or not 0 <= self.model_id < 30000:
            raise ValueError("Нужны ECVB/EFVB и ID 00000..29999")
        if self.order != 4 or isinstance(self.substeps, bool) or not isinstance(self.substeps, int) or self.substeps < 1:
            raise ValueError("Нужны p4 и положительное целое число подшагов")
        if self.source_x_m not in ExperimentSpec().source_x_m:
            raise ValueError("Недопустимое положение источника")
        if not isfinite(self.t_end_s) or self.t_end_s > 1.0:
            raise ValueError("Длительность должна быть конечной и не больше 1 с")
        inclusive_times(self.t_end_s, 0.001)

    @property
    def steps(self) -> int:
        return round(self.t_end_s / 0.001) * self.substeps

    @property
    def dt_s(self) -> float:
        return 0.001 / self.substeps


def suggested_substeps(stability: dict[str, object]) -> int:
    """Новые модели: ceil(1.6 Kmin), округлённое вверх до кратного восьми."""
    minimum = int(number(stability["substeps_per_ms"]))
    if minimum < 1 or minimum != number(stability["substeps_per_ms"]):
        raise ValueError("Неверный Kmin")
    return 8 * ((minimum + 4) // 5)


def receiver_nodes(case: MeshCase) -> IntArray:
    coords = case.mesh.nodes_coords
    surface = np.flatnonzero(np.abs(coords[:, 1]) <= 1e-9)
    result: list[int] = []
    for x in RECEIVER_X_M:
        matched = surface[np.abs(coords[surface, 0] - x) <= 1e-9]
        if matched.size != 1:
            raise ValueError(f"Приёмник ({x},0): {matched.size} точных узлов")
        result.append(int(matched[0]))
    return np.asarray(result, dtype=np.int64)


def validate_stability(case: MeshCase, spec: SurfaceSpec, mesh_path: Path,
                       stability_path: Path) -> dict[str, object]:
    """Проверить SHA сетки и параметры; абсолютный старый путь может переноситься."""
    record = read_json(stability_path)
    if record.get("mesh_sha256") != file_record(mesh_path)["sha256"]:
        raise ValueError("Сертификат не совпадает с SHA256 фактической сетки")
    for key, expected in (("order", spec.order), ("nodes", case.mesh.nodes_coords.shape[0]),
                          ("elements", case.element_vp.size)):
        if number(record[key]) != expected:
            raise ValueError(f"Сертификат не совпадает по {key}")
    if case.metadata.get("model_id") != spec.model_id or case.metadata.get("kind") != "curved" or case.metadata.get("order") != 4:
        raise ValueError("Метаданные фактической сетки не соответствуют опыту")
    if case.metadata.get("dataset", spec.dataset) != spec.dataset or record.get("dataset", spec.dataset) != spec.dataset:
        raise ValueError("Группа сетки/сертификата не совпадает с опытом")
    alpha = number(record["alpha_max_s_inv"])
    if not np.isclose(alpha, ALPHA_MAX_S_INV, atol=1e-10, rtol=1e-12):
        raise ValueError("Сертификат использует другое демпфирование")
    dt = number(record["dt_start_s"])
    minimum = number(record["substeps_per_ms"])
    if not isfinite(dt) or dt <= 0 or not isfinite(minimum) or minimum < 1 or minimum != int(minimum):
        raise ValueError("Неверный предел временного шага")
    if spec.substeps < int(minimum) or spec.dt_s > dt * (1 + 1e-12):
        raise ValueError("Временной шаг превышает сертификат устойчивости")
    if int(minimum) < ceil(0.001 / dt - 1e-10):
        raise ValueError("Kmin сертификата противоречит dt_start_s")
    receiver_nodes(case)
    nodal_point_force(np.asarray(case.mesh.nodes_coords, dtype=np.float64), spec.source_x_m, 1.0)
    return record


@dataclass
class SurfaceRecorder:
    times: FloatArray
    velocity: FloatArray
    count: int = 0

    @classmethod
    def create(cls, spec: SurfaceSpec) -> SurfaceRecorder:
        times = inclusive_times(spec.t_end_s, 0.001)
        return cls(times, np.empty((times.size, 70, 2), dtype=np.float64))

    def append(self, frame: FloatArray, event: SGBaseReceiverEventData) -> None:
        if self.count >= self.times.size:
            raise ValueError("Получен лишний кадр поверхности")
        if frame.size != 140 or frame.dtype != np.float64:
            raise ValueError("Неверное число компонент/приёмников")
        checked_time(frame, event, float(self.times[self.count]))
        self.velocity[self.count] = frame.reshape(2, 70).T
        self.count += 1

    def finish(self) -> None:
        if self.count != self.times.size:
            raise ValueError(f"Неполная запись: {self.count}/{self.times.size}")
        if not np.isfinite(self.velocity).all():
            raise FloatingPointError("Неконечные значения в завершённой сейсмограмме")


def _check_write(path: Path, expected_bytes: int, before_write: Callable[[], None] | None) -> None:
    if before_write is not None:
        before_write()
    existing = path.resolve()
    while not existing.exists():
        existing = existing.parent
    info = os.statvfs(existing)
    if info.f_bavail * info.f_frsize < 20 * GIB + expected_bytes:
        raise OSError("Недостаточно места: требуется сохранить резерв 20 ГиБ")


def _write_metadata(path: Path, metadata: dict[str, object], before_write: Callable[[], None] | None) -> None:
    expected = len(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False).encode()) + 4096
    _check_write(path.parent, expected, before_write)
    write_json(path, metadata)


def make_cu_model(case: MeshCase, spec: SurfaceSpec) -> tuple[SGModel[CUConfig], Path, int]:
    """Native CU импортируется только при прямом расчётном вызове."""
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu import _native
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig, SGRunCodeEnum
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.solver import SGSolver
    config = SGSolverConfig(t_begin=0.0, t_end=spec.t_end_s, steps_count=spec.steps, steps_per_tact=1)
    return SGModel(case.mesh, SGSolver(config)), Path(_native.__file__), int(SGRunCodeEnum.TIME_END)


def _execute(case: MeshCase, spec: SurfaceSpec, output: Path, metadata: dict[str, object],
             before_write: Callable[[], None] | None) -> dict[str, object]:
    experiment = ExperimentSpec()
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    source = nodal_point_force(coords, spec.source_x_m, experiment.force_amplitude_n_per_m)
    receivers = receiver_nodes(case)
    edges = abc_edges(case)
    model, native_path, success_code = make_cu_model(case, spec)
    source_times = np.linspace(0, spec.t_end_s, spec.steps + 1)
    values = np.zeros((source_times.size, 1, 2), dtype=np.float64)
    values[:, 0, 1] = source.force_y * ricker(source_times)
    model.add_physics(SGPhysicsModel.from_entity_values("material", 11000, case.mesh, 1,
        np.arange(case.element_vp.size, dtype=np.int64), plane_strain_parameters(case.element_vp, case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values("outer_abc", 32000, case.mesh, 2, edges,
        np.column_stack((experiment.density_kg_m3 * case.boundary_vp[edges], experiment.density_kg_m3 * case.boundary_vs[edges]))))
    node_ids = np.arange(coords.shape[0], dtype=np.int64)
    model.add_physics(SGPhysicsModel("mass_damping", 30001, case.mesh, 0, node_ids,
                                    damping_alpha(coords, ALPHA_MAX_S_INV)[:, None]))
    model.add_physics(SGPhysicsModel("point_ricker", 20200, case.mesh, 0,
        np.asarray([source.node_id], dtype=np.int64), values,
        time=SGPhysicsTimeData(0.0, spec.t_end_s, source_times.size, 3)))
    metadata["source_node_id"] = source.node_id
    metadata["source_input_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    metadata["source_input_first_force_y_n_per_m"] = float(values[0, 0, 1])
    metadata["native"] = file_record(native_path)
    metadata["nodes_sha256"] = hashlib.sha256(coords.tobytes()).hexdigest()
    recorder = SurfaceRecorder.create(spec)

    def handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        recorder.append(frame, event)
        if (recorder.count - 1) % 500 == 0:
            print(json.dumps({"stage": "surface", "time_s": float(recorder.times[recorder.count - 1])}), flush=True)

    model.add_receiver(SGReceiverModel.from_positions("surface_velocity", case.mesh, 0, receivers,
        np.asarray([4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps,
        stride=spec.substeps, handler=handler))
    model.prepare()
    print(json.dumps({"stage": "prepared", "nodes": len(coords), "steps": spec.steps}), flush=True)
    result = cast(TimedResult, model.run(sync=True))
    if int(result.code) != success_code or result.steps_count != spec.steps or not np.isclose(result.current_time, spec.t_end_s, atol=2e-10, rtol=0):
        raise RuntimeError(f"Расчёт не достиг заданного конца: {result}")
    recorder.finish()
    expected = recorder.velocity.nbytes + recorder.times.nbytes + receivers.nbytes + 70 * 2 * 8 + 4096
    _check_write(output, expected, before_write)
    partial = output / "surface.partial"
    with partial.open("xb") as stream:
        np.savez(stream, time_s=recorder.times, coordinates_m=coords[receivers], node_ids=receivers,
                 velocity_m_s=recorder.velocity)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(output / "surface.npz")
    return {"runtime": {"result_code": int(result.code), "time_end_code": success_code,
                         "steps_count": result.steps_count, "current_time_s": result.current_time},
            "recorded_counts": {"surface": recorder.count, "source_history": 0, "fields": 0},
            "surface_file": file_record(output / "surface.npz"), "surface_file_bytes": (output / "surface.npz").stat().st_size}


def run_case(case: MeshCase, spec: SurfaceSpec, out: Path, mesh_path: Path,
             stability_path: Path, before_write: Callable[[], None] | None = None) -> dict[str, object]:
    """Один источник; не перезаписывать даже незавершённый каталог."""
    out = out.resolve()
    out.relative_to((PROJECT_ROOT / "data/dev_1_4_server_preparation").resolve())
    validate_stability(case, spec, mesh_path, stability_path)
    _check_write(out.parent, 65536, before_write)
    out.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    metadata: dict[str, object] = {
        "status": "started", "run_spec": asdict(spec), "experiment_spec": {**asdict(ExperimentSpec()), "end_s": spec.t_end_s},
        "dt_s": spec.dt_s, "steps_count": spec.steps, "mesh": case.metadata,
        "mesh_evidence": file_record(mesh_path), "stability_evidence": file_record(stability_path),
        "short_test": spec.t_end_s != 1.0, "components": ["Vx", "Vy"], "receiver_count": 70,
        "initial_state": "native U0=V0=A0=0; first force assembly at dt; same convention as chapter 1.3",
        "storage": "uncompressed NPZ; float64 velocity; no displacements, source history, energy or all-node fields",
        "provenance": {"python": sys.executable, "python_version": sys.version,
            "numpy": np.__version__, "semgpu2": version("semgpu2"), "platform": platform.platform(),
            "files": [file_record(Path(__file__)), file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source/physics.py"),
                      file_record(PROJECT_ROOT / "scripts/dev_1_3_point_source/run.py")]},
    }
    try:
        _write_metadata(out / "metadata.started.json", metadata, before_write)
        metadata.update(_execute(case, spec, out, metadata, before_write))
        metadata["status"] = "completed"
        metadata["wall_seconds"] = perf_counter() - started
        _write_metadata(out / "metadata.json", metadata, before_write)
        return verify_surface(out, spec)
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["wall_seconds"] = perf_counter() - started
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["traceback"] = traceback.format_exc()
        try:
            _write_metadata(out / "metadata.failed.json", metadata, before_write)
        except BaseException as diagnostic_error:
            print(json.dumps({"status": "failed", "error": str(error), "failure_metadata_not_written": str(diagnostic_error)}), file=sys.stderr, flush=True)
        raise


def verify_surface(out: Path, spec: SurfaceSpec) -> dict[str, object]:
    """Проверить завершение, SHA и фактическую схему единственной сейсмограммы."""
    metadata = read_json(out / "metadata.json")
    if metadata.get("status") != "completed" or metadata.get("run_spec") != asdict(spec):
        raise ValueError("Опыт не завершён или спецификация отличается")
    if (out / "metadata.failed.json").exists() or (out / "surface.partial").exists():
        raise ValueError("Есть свидетельство ошибки или незавершённая запись")
    native = mapping(metadata["native"])
    native_sha = native.get("sha256")
    if not isinstance(native.get("path"), str) or not isinstance(native_sha, str) or len(native_sha) != 64 or any(c not in "0123456789abcdef" for c in native_sha):
        raise ValueError("Не зафиксирована фактическая native-библиотека")
    if not isinstance(mapping(metadata["provenance"]).get("semgpu2"), str):
        raise ValueError("Не зафиксирована версия semgpu2")
    runtime = mapping(metadata["runtime"])
    if number(runtime["result_code"]) != number(runtime["time_end_code"]) or number(runtime["steps_count"]) != spec.steps:
        raise ValueError("Неверный код/число шагов")
    if not np.isclose(number(runtime["current_time_s"]), spec.t_end_s, atol=2e-10, rtol=0):
        raise ValueError("Неверное конечное время")
    expected_times = inclusive_times(spec.t_end_s, 0.001)
    counts = mapping(metadata["recorded_counts"])
    if counts != {"surface": len(expected_times), "source_history": 0, "fields": 0}:
        raise ValueError("Неполная или лишняя запись")
    if mapping(metadata["surface_file"])["sha256"] != file_record(out / "surface.npz")["sha256"]:
        raise ValueError("SHA256 surface.npz изменился")
    with np.load(out / "surface.npz", allow_pickle=False) as data:
        if set(data.files) != {"time_s", "coordinates_m", "node_ids", "velocity_m_s"}:
            raise ValueError("Лишние или отсутствующие массивы surface.npz")
        times, coords, ids, velocity = (data[key] for key in ("time_s", "coordinates_m", "node_ids", "velocity_m_s"))
        if times.dtype != np.float64 or coords.dtype != np.float64 or ids.dtype != np.int64 or velocity.dtype != np.float64:
            raise ValueError("Тип хранения должен быть float64/int64")
        if times.shape != expected_times.shape or not np.array_equal(times, expected_times):
            raise ValueError("Неверная временная ось")
        if coords.shape != (70, 2) or ids.shape != (70,) or velocity.shape != (len(times), 70, 2):
            raise ValueError("Неверная форма приёмников/скорости")
        if not np.isfinite(coords).all() or not np.isfinite(velocity).all() or not np.all(ids >= 0) or len(np.unique(ids)) != 70:
            raise ValueError("Неконечные данные или неверные ID узлов")
        if not np.allclose(coords[:, 0], RECEIVER_X_M, atol=1e-9, rtol=0) or not np.allclose(coords[:, 1], 0, atol=1e-9, rtol=0):
            raise ValueError("Приёмники должны иметь координаты x=5,15,…,695, y=0")
        if not np.array_equal(velocity[0], np.zeros((70, 2))):
            raise ValueError("Нарушено нулевое начальное состояние скорости")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("ECVB", "EFVB"), required=True)
    parser.add_argument("--model", type=int, required=True)
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--stability", type=Path, required=True)
    parser.add_argument("--source", type=float, choices=ExperimentSpec().source_x_m, default=350.0)
    parser.add_argument("--substeps", type=int, help="Иначе ceil(1.6 Kmin), вверх до кратного 8; для согласования пилотов явно 64/224")
    parser.add_argument("--t-end", type=float, default=1.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
    frozen = load_frozen(args.mesh)
    case = build_sem(frozen, 4)
    with np.load(args.mesh, allow_pickle=False) as mesh_data:
        if "dataset" in mesh_data:
            case.metadata["dataset"] = str(mesh_data["dataset"])
        elif args.dataset != "ECVB" or args.model not in (65, 12404):
            raise ValueError("Сетка полной серии должна содержать dataset")
    substeps = args.substeps if args.substeps is not None else suggested_substeps(read_json(args.stability))
    spec = SurfaceSpec(cast(Dataset, args.dataset), args.model, args.source, substeps, args.t_end)
    result = run_case(case, spec, args.out, args.mesh, args.stability)
    print(json.dumps({"status": result["status"], "out": str(args.out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
