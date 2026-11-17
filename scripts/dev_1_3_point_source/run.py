"""Запуск отдельного опыта 1.3 с точечной силой и потоковой записью полей."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
from importlib.metadata import version
import json
from math import isfinite, log
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter
import traceback
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from semgpu2.model.model import SGModel
from semgpu2.model.physics import SGPhysicsModel, SGPhysicsTimeData
from semgpu2.model.receiver import SGReceiverModel
from semgpu2.solvers.solver_api import SGBaseReceiverEventData, SGReceiverRunControl
from scripts.dev_1_2_smoothing.pilot_mesh import MeshCase
from scripts.dev_1_2_smoothing.pilot_run import plane_strain_parameters
from scripts.dev_1_3_point_source.physics import ExperimentSpec, damping_alpha, inclusive_times, nodal_point_force, ricker
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import number, read_json

if TYPE_CHECKING:
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed.constants import SGSolverConfig as CEEDConfig
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
FIELD_COMPONENTS = np.asarray([2, 3, 4, 5, 100, 101, 110, 111], dtype=np.int32)


class TimedEvent(Protocol):
    current_time: float


class TimedResult(Protocol):
    code: int
    current_time: float
    steps_count: int


@dataclass(frozen=True, slots=True)
class RunSpec:
    model_id: int
    kind: str
    order: int
    source_x_m: float
    substeps: int
    t_end_s: float = 2.0
    backend: str = "ceed"
    ceed_resource: str = "/cpu/self/opt/blocked"
    alpha_max_s_inv: float = -3.0 * 4385.0 * log(0.01) / 175.0
    fields: bool = True
    field_layout: str = "full"

    def __post_init__(self) -> None:
        if self.model_id not in (65, 12404) or self.kind not in ("raster", "curved"):
            raise ValueError("Ожидается ECVB 65/12404 и raster/curved")
        if self.order not in (3, 4, 5) or self.substeps < 1:
            raise ValueError("Ожидается p3/p4/p5 и положительное число подшагов")
        if self.source_x_m not in ExperimentSpec().source_x_m:
            raise ValueError("Источник должен находиться в одном из пяти заданных положений")
        if self.backend not in ("ceed", "cu"):
            raise ValueError("Backend должен быть ceed/cu")
        if self.field_layout not in ("full", "compact"):
            raise ValueError("Формат полей должен быть full/compact")
        if not isfinite(self.alpha_max_s_inv) or self.alpha_max_s_inv < 0:
            raise ValueError("Демпфирование должно быть конечным и неотрицательным")
        if self.t_end_s > 2.0:
            raise ValueError("Глава 1.3 ограничена интервалом до 2 с")
        inclusive_times(self.t_end_s, 0.01)

    @property
    def steps(self) -> int:
        return round(self.t_end_s / 0.001) * self.substeps

    @property
    def dt_s(self) -> float:
        return 0.001 / self.substeps


def write_json(path: Path, record: dict[str, object]) -> None:
    """Записать законченный JSON только в ещё не существующий файл."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def file_record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def loaded_ceed_libraries() -> list[Path]:
    """Фактические shared libraries Linux, а не предположение по build-dir."""
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths: set[Path] = set()
    for line in maps.read_text(encoding="utf-8").splitlines():
        columns = line.split(maxsplit=5)
        if len(columns) == 6 and columns[5].startswith("/"):
            path = Path(columns[5])
            if "libceed" in path.name.lower() and path.is_file():
                paths.add(path.resolve())
    return sorted(paths)


def validate_stability(case: MeshCase, spec: RunSpec, mesh_path: Path | None,
                       stability_path: Path | None) -> None:
    """Сверить оценку с фактическим запуском до создания каталога результатов."""
    if stability_path is None:
        if mesh_path is not None:
            raise ValueError("Для расчётной сетки требуется --stability")
        return  # Synthetic recorder smoke has no frozen production mesh.
    if mesh_path is None:
        raise ValueError("Оценка устойчивости требует пути исходной сетки")
    record = read_json(stability_path)
    stored_mesh = record.get("mesh")
    if not isinstance(stored_mesh, str) or Path(stored_mesh).resolve() != mesh_path.resolve():
        raise ValueError("Оценка устойчивости относится к другой сетке")
    expected = {"order": spec.order, "nodes": case.mesh.nodes_coords.shape[0],
                "elements": case.element_vp.size}
    for key, value in expected.items():
        if number(record[key]) != value:
            raise ValueError(f"Оценка устойчивости не совпадает по {key}")
    alpha = number(record["alpha_max_s_inv"])
    if not np.isclose(alpha, spec.alpha_max_s_inv, atol=1e-10, rtol=1e-12):
        raise ValueError("Оценка устойчивости использует другой коэффициент демпфирования")
    dt = number(record["dt_start_s"])
    if not isfinite(dt) or dt <= 0 or spec.dt_s > dt * (1 + 1e-12):
        raise ValueError(f"Выбранный шаг {spec.dt_s:g} с превышает оценку {dt:g} с")


def exact_surface_nodes(case: MeshCase) -> IntArray:
    """Однозначные глобальные узлы заданных 141 приёмников, без интерполяции."""
    coords = case.mesh.nodes_coords
    surface = np.flatnonzero(np.abs(coords[:, 1]) <= 1e-9)
    ids: list[int] = []
    for x in np.arange(141, dtype=np.float64) * 5.0:
        matched = surface[np.abs(coords[surface, 0] - x) <= 1e-9]
        if matched.size != 1:
            raise ValueError(f"Приёмник ({x},0) имеет {matched.size} совпадающих узлов")
        ids.append(int(matched[0]))
    return np.asarray(ids, dtype=np.int64)


def abc_edges(case: MeshCase) -> IntArray:
    """Только внешние бока и дно; внутренние границы не должны быть boundary."""
    boundary = case.mesh.strata[2]
    selected: list[int] = []
    for edge in range(boundary.p_node_offsets.size - 1):
        ids = boundary.p_nodes[boundary.p_node_offsets[edge]:boundary.p_node_offsets[edge + 1]]
        xy = case.mesh.nodes_coords[ids]
        if np.all(np.abs(xy[:, 1]) < 1e-8):
            continue
        if not (np.all(np.abs(xy[:, 0] + 175.0) < 1e-8) or
                np.all(np.abs(xy[:, 0] - 875.0) < 1e-8) or
                np.all(np.abs(xy[:, 1] + 875.0) < 1e-8)):
            raise ValueError(f"Грань {edge} не принадлежит внешнему прямоугольнику")
        selected.append(edge)
    return np.asarray(selected, dtype=np.int64)


def make_model(case: MeshCase, spec: RunSpec) -> tuple[SGModel[CEEDConfig] | SGModel[CUConfig], Path, int]:
    """Импортировать фактический native только при явно вызванном расчёте."""
    if spec.backend == "ceed":
        from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed import _native as native_ceed
        from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed.constants import SGSolverConfig as CEEDConfig, SGRunCodeEnum as CEEDCode
        from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_ceed.solver import SGSolver as CEEDSolver
        config = CEEDConfig(t_begin=0.0, t_end=spec.t_end_s, steps_count=spec.steps,
                            steps_per_tact=1, ceed_resource=spec.ceed_resource)
        return SGModel(case.mesh, CEEDSolver(config)), Path(native_ceed.__file__), int(CEEDCode.TIME_END)
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu import _native as native_cu
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.constants import SGSolverConfig as CUConfig, SGRunCodeEnum as CUCode
    from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu.solver import SGSolver as CUSolver
    cu_config = CUConfig(t_begin=0.0, t_end=spec.t_end_s, steps_count=spec.steps, steps_per_tact=1)
    return SGModel(case.mesh, CUSolver(cu_config)), Path(native_cu.__file__), int(CUCode.TIME_END)


def checked_time(frame: FloatArray, event: SGBaseReceiverEventData, expected: float) -> float:
    current = float(cast(TimedEvent, event).current_time)
    if not np.isfinite(frame).all():
        raise FloatingPointError(f"Неконечный кадр при t={current}")
    if not np.isclose(current, expected, atol=2e-10, rtol=0):
        raise ValueError(f"Неверное время кадра: {current}, ожидалось {expected}")
    return current


def run_case(case: MeshCase, spec: RunSpec, output: Path, mesh_path: Path | None = None,
             stability_path: Path | None = None) -> dict[str, object]:
    """Выполнить один опыт; не перезаписывать даже незавершённый каталог."""
    output = output.resolve()
    output.relative_to((PROJECT_ROOT / "data/dev_1_3_point_source").resolve())
    validate_stability(case, spec, mesh_path, stability_path)
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    experiment = ExperimentSpec()
    metadata: dict[str, object] = {
        "status": "started", "run_spec": asdict(spec), "experiment_spec": asdict(experiment),
        "dt_s": spec.dt_s, "mesh": case.metadata, "steps_count": spec.steps,
        "coordinate_convention": "x right, y upward, surface y=0, main source peak downward",
        "short_test": spec.t_end_s != 2.0,
        "field_components": (["ux", "uy", "vx", "vy", "kinetic_energy", "elastic_energy", "kinetic_density", "elastic_density"]
                             if spec.field_layout == "full" else ["vx", "vy", "kinetic_density", "elastic_density"]),
        "field_layout_note": "compact omits all-node displacement and redundant nodal energies; full float64 velocity and both energy densities remain. Total energies are still summed from native nodal energies and saved separately.",
        "energy_limitations": "Integrated source work and total energies are measured. Separate core/layer balance and cumulative ABC/mass-damping losses are not implemented; no energy-balance closure is claimed.",
    }
    if stability_path is not None:
        metadata["stability_evidence"] = file_record(stability_path)
    try:
        result = _execute(case, spec, output, mesh_path, metadata)
        metadata.update(result)
        metadata["status"] = "completed"
        metadata["wall_seconds"] = perf_counter() - started
        write_json(output / "metadata.json", metadata)
        return metadata
    except BaseException as error:
        metadata["status"] = "failed"
        metadata["wall_seconds"] = perf_counter() - started
        metadata["error"] = f"{type(error).__name__}: {error}"
        metadata["traceback"] = traceback.format_exc()
        write_json(output / "metadata.failed.json", metadata)
        raise


def _execute(case: MeshCase, spec: RunSpec, output: Path, mesh_path: Path | None,
             metadata: dict[str, object]) -> dict[str, object]:
    experiment = ExperimentSpec()
    coords = np.asarray(case.mesh.nodes_coords, dtype=np.float64)
    source = nodal_point_force(coords, spec.source_x_m, experiment.force_amplitude_n_per_m)
    receivers = exact_surface_nodes(case)
    boundary_ids = abc_edges(case)
    main_count = case.element_vp.size
    model, native_path, success_code = make_model(case, spec)
    source_times = np.linspace(0, spec.t_end_s, spec.steps + 1)
    values = np.zeros((source_times.size, 1, 2), dtype=np.float64)
    values[:, 0, 1] = source.force_y * ricker(source_times)
    model.add_physics(SGPhysicsModel.from_entity_values("material", 11000, case.mesh, 1,
        np.arange(main_count, dtype=np.int64), plane_strain_parameters(case.element_vp, case.element_vs)))
    model.add_physics(SGPhysicsModel.from_entity_values("outer_abc", 32000, case.mesh, 2, boundary_ids,
        np.column_stack((experiment.density_kg_m3 * case.boundary_vp[boundary_ids], experiment.density_kg_m3 * case.boundary_vs[boundary_ids]))))
    node_ids = np.arange(coords.shape[0], dtype=np.int64)
    alpha = damping_alpha(coords, spec.alpha_max_s_inv)
    model.add_physics(SGPhysicsModel("mass_damping", 30001, case.mesh, 0, node_ids, alpha[:, None]))
    model.add_physics(SGPhysicsModel("point_ricker", 20200, case.mesh, 0,
        np.asarray([source.node_id], dtype=np.int64), values,
        time=SGPhysicsTimeData(0.0, spec.t_end_s, source_times.size, 3)))
    files = [native_path, *loaded_ceed_libraries(), Path(__file__), Path(__file__).with_name("physics.py"), Path(__file__).with_name("mesh.py")]
    if mesh_path is not None:
        files.append(mesh_path)
    metadata["provenance"] = {"files": [file_record(p) for p in files if p.is_file()],
        "python": sys.executable, "python_version": sys.version, "numpy": np.__version__, "semgpu2": version("semgpu2"),
        "platform": platform.platform(), "nodes_sha256": hashlib.sha256(coords.tobytes()).hexdigest()}
    metadata["source_node_id"] = source.node_id
    saved_components = 8 if spec.field_layout == "full" else 4
    metadata["storage_uncompressed_fields_bytes"] = (round(spec.t_end_s / .01) + 1) * coords.shape[0] * saved_components * 8 if spec.fields else 0
    write_json(output / "metadata.started.json", metadata)
    surface_times = inclusive_times(spec.t_end_s, 0.001)
    surface = np.empty((surface_times.size, 141, 4), dtype=np.float64)
    source_history = np.empty((source_times.size, 2), dtype=np.float64)
    surface_count = 0
    source_count = 0
    field_count = 0
    field_times = inclusive_times(spec.t_end_s, 0.01)
    energy = np.empty((field_times.size, 2), dtype=np.float64)
    fields = output / "fields"
    if spec.fields:
        fields.mkdir()
        np.savez_compressed(fields / "geometry.npz", coordinates_m=coords,
            cells=case.mesh.strata[1].p_nodes.reshape(-1, (spec.order + 1)**2), order=spec.order,
            element_vp=case.element_vp, element_vs=case.element_vs,
            reference_nodes_1d=case.mesh.pets[0].p_nodes.coords[:, 0],
            physical_node_mask=(coords[:, 0] >= 0) & (coords[:, 0] <= 700) & (coords[:, 1] >= -700),
            damping_alpha_s_inv=alpha)

    def surface_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        nonlocal surface_count
        checked_time(frame, event, float(surface_times[surface_count]))
        surface[surface_count] = frame.reshape(4, 141).T
        surface_count += 1
        if (surface_count - 1) % 100 == 0:
            print(json.dumps({"stage": "surface", "time_s": float(surface_times[surface_count - 1])}), flush=True)

    def source_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        nonlocal source_count
        checked_time(frame, event, float(source_times[source_count]))
        source_history[source_count] = frame.reshape(2)
        source_count += 1

    def field_handler(frame: FloatArray, event: SGBaseReceiverEventData, control: SGReceiverRunControl) -> None:
        nonlocal field_count
        current = checked_time(frame, event, float(field_times[field_count]))
        data = frame.reshape(8, coords.shape[0])
        energy[field_count] = data[4:6].sum(axis=1)
        # Reserve at least the uncompressed frame plus a small filesystem margin.
        if shutil.disk_usage(fields).free < data.nbytes + 100_000_000:
            raise OSError("Недостаточно места для следующего кадра; уже сохранённые данные оставлены")
        frame_path = fields / f"frame_{field_count:04d}.npz"
        if spec.field_layout == "full":
            np.savez_compressed(frame_path, time_s=current,
                displacement_m=data[:2].T, velocity_m_s=data[2:4].T,
                kinetic_energy_j_per_m=data[4], elastic_energy_j_per_m=data[5],
                kinetic_energy_density_j_m3=data[6], elastic_energy_density_j_m3=data[7])
        else:
            np.savez_compressed(frame_path, time_s=current, velocity_m_s=data[2:4].T,
                kinetic_energy_density_j_m3=data[6], elastic_energy_density_j_m3=data[7])
        field_count += 1

    model.add_receiver(SGReceiverModel.from_positions("surface", case.mesh, 0, receivers,
        np.asarray([2, 3, 4, 5], dtype=np.int32), begin_tact=0, end_tact=spec.steps,
        stride=spec.substeps, handler=surface_handler))
    model.add_receiver(SGReceiverModel.from_positions("source_work", case.mesh, 0,
        np.asarray([source.node_id], dtype=np.int64), np.asarray([5, 13], dtype=np.int32),
        begin_tact=0, end_tact=spec.steps, stride=1, handler=source_handler))
    if spec.fields:
        model.add_receiver(SGReceiverModel.from_positions("all_node_fields", case.mesh, 0, node_ids,
            FIELD_COMPONENTS, begin_tact=0, end_tact=spec.steps, stride=10 * spec.substeps,
            handler=field_handler))
    model.prepare()
    print(json.dumps({"stage": "prepared", "nodes": coords.shape[0], "steps": spec.steps}), flush=True)
    result = cast(TimedResult, model.run(sync=True))
    if int(result.code) != success_code or result.steps_count != spec.steps or not np.isclose(result.current_time, spec.t_end_s, atol=2e-10, rtol=0):
        raise RuntimeError(f"Расчёт не достиг заданного конца: {result}")
    if surface_count != surface_times.size or source_count != source_times.size or (spec.fields and field_count != field_times.size):
        raise RuntimeError(f"Неполная запись: surface={surface_count}, source={source_count}, fields={field_count}")
    np.savez_compressed(output / "surface.npz", time_s=surface_times, coordinates_m=coords[receivers],
        node_ids=receivers, displacement_m=surface[:, :, :2], velocity_m_s=surface[:, :, 2:])
    power = source_history[:, 0] * source_history[:, 1]
    work = np.r_[0.0, np.cumsum((power[1:] + power[:-1]) * 0.5 * np.diff(source_times))]
    np.savez_compressed(output / "source_history.npz", time_s=source_times,
        velocity_y_m_s=source_history[:, 0], force_y_n_per_m=source_history[:, 1],
        power_w_per_m=power, work_j_per_m=work)
    if spec.fields:
        np.savez_compressed(output / "energy_balance.npz", time_s=field_times,
            kinetic_j_per_m=energy[:, 0], elastic_j_per_m=energy[:, 1],
            source_work_j_per_m=np.interp(field_times, source_times, work))
    return {"runtime": {"result_code": int(result.code), "steps_count": result.steps_count,
                "current_time_s": result.current_time},
            "recorded_counts": {"surface": surface_count, "source": source_count, "fields": field_count},
            "source_work_final_j_per_m": float(work[-1])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=int, choices=(65, 12404), required=True)
    parser.add_argument("--kind", choices=("raster", "curved"), required=True)
    parser.add_argument("--order", type=int, choices=(3, 4, 5), required=True)
    parser.add_argument("--source-x", type=float, choices=ExperimentSpec().source_x_m, default=350.0)
    parser.add_argument("--substeps", type=int, required=True)
    parser.add_argument("--t-end", type=float, default=2.0)
    parser.add_argument("--backend", choices=("ceed", "cu"), default="ceed")
    parser.add_argument("--ceed-resource", default="/cpu/self/opt/blocked")
    parser.add_argument("--alpha-max", type=float, default=-3.0 * 4385.0 * log(0.01) / 175.0)
    parser.add_argument("--no-fields", action="store_true")
    parser.add_argument("--field-layout", choices=("full", "compact"), default="full",
                        help="compact сохраняет float64 Vx/Vy и две плотности энергии; суммарные энергии отдельно")
    parser.add_argument("--mesh", type=Path)
    parser.add_argument("--stability", type=Path, help="Проверенная оценка устойчивости для provenance")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    spec = RunSpec(args.model, args.kind, args.order, args.source_x, args.substeps, args.t_end,
                   args.backend, args.ceed_resource, args.alpha_max, not args.no_fields, args.field_layout)
    from scripts.dev_1_3_point_source.mesh import build_sem, load_frozen
    mesh_path = args.mesh or PROJECT_ROOT / f"data/dev_1_3_point_source/meshes_v1/{args.model:05d}_{args.kind}_q2.npz"
    frozen = load_frozen(mesh_path)
    if frozen.model_id != spec.model_id or frozen.kind != spec.kind:
        raise ValueError("Метаданные сетки не соответствуют выбранному опыту")
    case = build_sem(frozen, spec.order)
    result = run_case(case, spec, args.out, mesh_path, args.stability)
    print(json.dumps({"status": result["status"], "out": str(args.out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
