"""Сопоставить завершённые записи главы 1.3; расчётный решатель не загружается."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from scripts.project_paths import PROJECT_ROOT, image_directory
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from scripts.reporting import mapping, number, read_json, sequence

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
COMPONENTS = ("Ux", "Uy", "Vx", "Vy")
UNITS = ("м", "м", "м/с", "м/с")
RECEIVER_COORDINATE_TOLERANCE_M = 1e-9


def receiver_mask(coordinates_m: FloatArray, source_x_m: float, radius_m: float) -> BoolArray:
    """Включить граничные приёмники радиуса с тем же допуском, что проверка узлов.

    Q2/GLL вычисления могут дать x=340+5.7e-14 вместо 340: это не повод
    исключать физический приёмник на расстоянии ровно 10 м от x=350.
    """
    if coordinates_m.ndim != 2 or coordinates_m.shape[1] != 2 or not np.isfinite(coordinates_m).all():
        raise ValueError("Нужны конечные координаты (n,2)")
    if not np.isfinite(source_x_m) or not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("Нужны конечные источник и неотрицательный радиус")
    return np.abs(coordinates_m[:, 0] - source_x_m) >= radius_m - RECEIVER_COORDINATE_TOLERANCE_M


@dataclass(frozen=True, slots=True)
class RunData:
    """Полная запись четырёх поверхностных компонент одного выстрела."""

    path: Path
    model_id: int
    kind: str
    order: int
    source_x_m: float
    substeps: int
    end_s: float
    dt_s: float
    time_s: FloatArray
    coordinates_m: FloatArray
    surface: FloatArray
    metadata: dict[str, object]

    @property
    def label(self) -> str:
        return f"ECVB {self.model_id:05d} {self.kind}, p{self.order}, xₛ={self.source_x_m:g} м, K={self.substeps}"

    @property
    def slug(self) -> str:
        return f"m{self.model_id:05d}_{self.kind}_p{self.order}_x{self.source_x_m:g}_k{self.substeps}"


def sha256(path: Path) -> str:
    """Хеш компактного артефакта без привязки к изменяемому checkout решателя."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_run(path: Path, allow_short: bool = False) -> RunData:
    """Принять только завершённый запуск с конечными данными и точными отсчётами."""
    path = path.resolve()
    metadata = read_json(path / "metadata.json")
    if metadata.get("status") != "completed":
        raise ValueError(f"Нет успешного статуса завершения: {path}")
    spec = mapping(metadata["run_spec"])
    model_id = int(number(spec["model_id"]))
    order = int(number(spec["order"]))
    substeps = int(number(spec["substeps"]))
    source_x = number(spec["source_x_m"])
    end_s = number(spec["t_end_s"])
    dt_s = number(metadata["dt_s"])
    kind = spec["kind"]
    if not isinstance(kind, str) or kind not in ("raster", "curved"):
        raise ValueError(f"Неизвестная геометрия: {path}")
    if model_id not in (65, 12404) or order not in (3, 4, 5) or substeps < 1:
        raise ValueError(f"Неожиданный состав основной серии: {path}")
    if source_x not in (0.0, 175.0, 350.0, 525.0, 700.0):
        raise ValueError(f"Неожиданный источник: {path}")
    if not 0 < end_s <= 2 or (not allow_short and end_s != 2.0):
        raise ValueError(f"Нужна полная запись 0–2 с: {path}")
    if not np.isclose(dt_s, 0.001 / substeps, rtol=1e-12, atol=0):
        raise ValueError(f"Внутренний шаг не согласован с K: {path}")
    count = round(end_s / 0.001) + 1
    with np.load(path / "surface.npz", allow_pickle=False) as data:
        time = np.asarray(data["time_s"], dtype=np.float64)
        coordinates = np.asarray(data["coordinates_m"], dtype=np.float64)
        displacement = np.asarray(data["displacement_m"], dtype=np.float64)
        velocity = np.asarray(data["velocity_m_s"], dtype=np.float64)
    if time.shape != (count,) or not np.allclose(time, np.linspace(0, end_s, count), rtol=0, atol=1e-12):
        raise ValueError(f"Неверная временная сетка поверхности: {path}")
    expected_coordinates = np.column_stack((np.arange(141) * 5, np.zeros(141)))
    if coordinates.shape != (141, 2) or not np.allclose(coordinates, expected_coordinates, rtol=0, atol=1e-9):
        raise ValueError(f"Нужны все 141 приёмник через 5 м: {path}")
    if displacement.shape != (count, 141, 2) or velocity.shape != displacement.shape:
        raise ValueError(f"Неверные размеры U/V: {path}")
    surface = np.concatenate((displacement, velocity), axis=2)
    if not np.isfinite(surface).all():
        raise ValueError(f"Неконечные поверхностные значения: {path}")
    return RunData(path, model_id, kind, order, source_x, substeps, end_s, dt_s,
                   time, coordinates, surface, metadata)


def compatible(left: RunData, right: RunData, comparison: str) -> None:
    """Исключить подмену физики или времени при сравнении дискретизаций."""
    if comparison not in ("p", "dt", "geometry"):
        raise ValueError("Неизвестный вид сравнения")
    if (left.model_id, left.source_x_m, left.end_s) != (right.model_id, right.source_x_m, right.end_s):
        raise ValueError("Разные модель, источник или длительность")
    for a, b in ((left.time_s, right.time_s), (left.coordinates_m, right.coordinates_m)):
        if a.shape != b.shape or not np.allclose(a, b, rtol=0, atol=1e-12):
            raise ValueError("Разные координаты сравнения")
    spec_left, spec_right = mapping(left.metadata["run_spec"]), mapping(right.metadata["run_spec"])
    for key in ("backend", "ceed_resource", "alpha_max_s_inv"):
        if spec_left.get(key) != spec_right.get(key):
            raise ValueError(f"Разные параметры {key}")
    if "experiment_spec" not in left.metadata or "experiment_spec" not in right.metadata:
        raise ValueError("Нет полного физического описания experiment_spec")
    if left.metadata["experiment_spec"] != right.metadata["experiment_spec"]:
        raise ValueError("Разные физические постановки experiment_spec")
    if comparison == "p" and (left.kind != right.kind or left.substeps != right.substeps or left.order >= right.order):
        raise ValueError("Для p-сравнения нужны одинаковые геометрия/шаг и меньший порядок как value")
    if comparison == "dt" and (left.kind != right.kind or left.order != right.order):
        raise ValueError("Для временного контроля нужны одинаковые геометрия и порядок")
    if comparison == "geometry" and (left.order != right.order or left.substeps != right.substeps):
        raise ValueError("Для геометрического эффекта нужны одинаковые p и шаг")
    if comparison in ("p", "dt"):
        hashes: list[str | None] = []
        for run in (left, right):
            provenance = mapping(run.metadata.get("provenance", {}))
            matches = [mapping(value) for value in sequence(provenance.get("files", []))
                       if Path(str(mapping(value)["path"])).name.endswith("_q2.npz")]
            if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
                if run.end_s == 2.0:
                    raise ValueError(f"Нет однозначного SHA256 замороженной Q2-сетки: {run.path}")
                hashes.append(None)
            else:
                hashes.append(str(matches[0]["sha256"]))
        if hashes[0] != hashes[1]:
            raise ValueError("В p/dt-сравнении изменена замороженная Q2-геометрия")


def metrics(value: RunData, reference: RunData, comparison: str) -> dict[str, object]:
    """Дискретные L2-нормы, максимумы и близкая к нулю опорная компонента."""
    compatible(value, reference, comparison)
    windows: list[dict[str, object]] = []
    full_norms = np.linalg.norm(reference.surface.reshape(-1, 4), axis=0)
    scale_by_family = np.array([max(full_norms[:2])] * 2 + [max(full_norms[2:])] * 2)
    for begin, end in ((0.0, reference.end_s), (1.0, reference.end_s)):
        if end <= begin:
            continue
        selected_time = (reference.time_s >= begin) & (reference.time_s <= end)
        for radius in (0.0, 10.0, 20.0):
            selected_receiver = receiver_mask(reference.coordinates_m, reference.source_x_m, radius)
            actual = value.surface[selected_time][:, selected_receiver]
            ref = reference.surface[selected_time][:, selected_receiver]
            difference = actual - ref
            records: dict[str, object] = {}
            for component, name in enumerate(COMPONENTS):
                numerator = float(np.linalg.norm(difference[:, :, component]))
                denominator = float(np.linalg.norm(ref[:, :, component]))
                near_zero = denominator <= max(float(scale_by_family[component]) * 1e-12, np.finfo(float).tiny)
                records[name] = {
                    "unit": UNITS[component], "difference_discrete_l2": numerator,
                    "reference_discrete_l2": denominator,
                    "relative_l2": None if near_zero else numerator / denominator,
                    "reference_near_zero": near_zero,
                    "max_abs_difference": float(np.max(np.abs(difference[:, :, component]))),
                    "reference_max_abs": float(np.max(np.abs(ref[:, :, component]))),
                }
            windows.append({"time_window_s": [begin, end], "excluded_radius_m": radius,
                            "receiver_count": int(selected_receiver.sum()), "components": records})
    main_window = next(mapping(record["components"]) for record in windows
                       if record["time_window_s"] == [0.0, reference.end_s] and record["excluded_radius_m"] == 20.0)
    relative = [mapping(main_window[name])["relative_l2"] for name in COMPONENTS]
    threshold = 0.01 if comparison == "p" else 0.001 if comparison == "dt" else None
    preliminary_pass = (None if threshold is None or reference.end_s != 2.0 or any(value is None for value in relative)
                        else all(number(value) <= threshold for value in relative))
    return {"comparison": comparison, "value": value.slug, "reference": reference.slug,
            "norm_definition": "unweighted_discrete_l2_common_time_and_receiver_samples",
            "receiver_mask_boundary_tolerance_m": RECEIVER_COORDINATE_TOLERANCE_M,
            "source_point_included_only_when_radius_zero": True, "windows": windows,
            "preliminary_full_far20_threshold": threshold,
            "preliminary_full_far20_all_components_pass": preliminary_pass,
            "threshold_limit": "Предварительный порог по полной записи вне 20 м; это не подтверждение временного запаса 1/5 и не проверка отражений."}


def render_surface_pair(value: RunData, reference: RunData, destination: Path,
                        begin_s: float = 0.0, exclude_radius_m: float = 0.0) -> None:
    """Четыре компоненты: общая шкала пары и отдельная шкала разности."""
    selected = reference.time_s >= begin_s
    actual, ref = value.surface[selected], reference.surface[selected]
    difference = actual - ref
    if exclude_radius_m > 0:
        hidden = ~receiver_mask(reference.coordinates_m, reference.source_x_m, exclude_radius_m)
        # Mask copies only; raw arrays and all-receiver norms remain untouched.
        actual = np.where(hidden[None, :, None], np.nan, actual)
        ref = np.where(hidden[None, :, None], np.nan, ref)
        difference = np.where(hidden[None, :, None], np.nan, difference)
    fig, axes = plt.subplots(4, 3, figsize=(13, 14), layout="constrained")
    for component, (name, unit) in enumerate(zip(COMPONENTS, UNITS, strict=True)):
        common = float(max(np.nanmax(np.abs(actual[:, :, component])), np.nanmax(np.abs(ref[:, :, component]))))
        delta = float(np.nanmax(np.abs(difference[:, :, component])))
        for column, field in enumerate((ref, actual, difference)):
            limit = (delta if column == 2 else common) or 1.0
            ax = axes[component, column]
            im = ax.imshow(field[:, :, component], extent=(0, 700, reference.end_s, begin_s),
                           aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
            ax.axvline(reference.source_x_m, linewidth=0.7, linestyle=":", color="black")
            ax.set(xlabel="x, м", ylabel="t, с", title=f"{name}, {unit} · " + ("опорная" if column == 0 else "сравниваемая" if column == 1 else "разность"))
            fig.colorbar(im, ax=ax, shrink=0.8, format="%.1e")
    exclusion = (f"\nНа рисунке скрыты |x−xₛ|<{exclude_radius_m:g} м; общие шкалы пары по оставшимся приёмникам за всё окно"
                 if exclude_radius_m > 0 else "")
    fig.suptitle(f"{value.label}\nминус {reference.label}\nВертикальная линия — источник; у разности отдельная шкала{exclusion}", fontsize=11)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def far_surface_report(value_path: Path, reference_path: Path, output: Path) -> dict[str, object]:
    """Дополнительная полная сейсмограмма с видимым исключением радиуса 20 м."""
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    value, reference = load_run(value_path), load_run(reference_path)
    compatible(value, reference, "p")
    output.mkdir(parents=True)
    images = image_directory(output)
    destination = images / "seismograms_full_far20.png"
    if destination.exists():
        raise FileExistsError(destination)
    render_surface_pair(value, reference, destination, exclude_radius_m=20.0)
    record: dict[str, object] = {
        "status": "supplemental_far20_plot_complete", "value": value.slug, "reference": reference.slug,
        "time_window_s": [0.0, 2.0], "hidden_source_radius_m": 20.0,
        "hidden_condition": "abs(x-source_x)<20m", "raw_outputs_modified": False,
        "receiver_mask_boundary_tolerance_m": RECEIVER_COORDINATE_TOLERANCE_M,
        "scales": "One constant physical-unit scale per component shared by the order pair, computed from all times and retained receivers; difference has its own scale; no per-trace/time normalization",
        "image": str(destination.relative_to(PROJECT_ROOT)),
        "inputs": [{"path": str(run.path), "metadata_sha256": sha256(run.path / "metadata.json"),
                    "surface_sha256": sha256(run.path / "surface.npz")} for run in (value, reference)],
        "limitation": "Дополнительный рисунок для удалённых приёмников; полные all-receiver данные и их отличия у источника сохраняют самостоятельное значение.",
    }
    (output / "report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def render_surface(run: RunData, destination: Path, begin_s: float = 0.0) -> None:
    """Четыре реальные компоненты одного завершённого опыта без мнимой пары."""
    selected = run.time_s >= begin_s
    data = run.surface[selected]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for component, (name, unit) in enumerate(zip(COMPONENTS, UNITS, strict=True)):
        ax = axes.ravel()[component]
        limit = float(np.max(np.abs(data[:, :, component]))) or 1.0
        im = ax.imshow(data[:, :, component], extent=(0, 700, run.end_s, begin_s),
                       aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
        ax.axvline(run.source_x_m, linewidth=0.7, linestyle=":", color="black")
        ax.set(xlabel="x, м", ylabel="t, с", title=f"{name}, {unit}")
        fig.colorbar(im, ax=ax, shrink=0.85, format="%.1e")
    fig.suptitle(f"{run.label}\nОдин завершённый опыт; вертикальная линия — источник", fontsize=11)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def single_report(run_path: Path, output: Path, allow_short: bool = False) -> dict[str, object]:
    """Сейсмограммы одного опыта; этот отчёт не заявляет проверки сходимости."""
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    run = load_run(run_path, allow_short)
    output.mkdir(parents=True)
    images = image_directory(output)
    outputs: list[str] = []
    for begin in (0.0, 1.0):
        if begin >= run.end_s:
            continue
        destination = images / f"seismograms_{'full' if begin == 0 else 'late'}.png"
        if destination.exists():
            raise FileExistsError(destination)
        render_surface(run, destination, begin)
        outputs.append(str(destination.relative_to(PROJECT_ROOT)))
    record: dict[str, object] = {
        "status": "single_run_plot_complete", "short_run": allow_short,
        "run": run.slug, "input": str(run.path), "metadata_sha256": sha256(run.path / "metadata.json"),
        "surface_sha256": sha256(run.path / "surface.npz"), "images": outputs,
        "surface_shape": run.surface.shape, "component_order": COMPONENTS,
        "max_abs_by_component": np.max(np.abs(run.surface), axis=(0, 1)).tolist(),
        "limitation": "Один завершённый опыт; p/dt-сходимость и геометрический эффект не оценивались.",
    }
    (output / "report.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def pairs(runs: Sequence[RunData]) -> list[tuple[str, RunData, RunData]]:
    """Найти пары p3/p4, p3/p5, p4/p5, геометрий и соседних шагов."""
    result: list[tuple[str, RunData, RunData]] = []
    identities = [(r.model_id, r.kind, r.order, r.source_x_m, r.substeps, r.end_s) for r in runs]
    if len(set(identities)) != len(identities):
        raise ValueError("Дублирующиеся идентичности запусков")
    for left in runs:
        for right in runs:
            if (left.model_id, left.source_x_m, left.end_s) != (right.model_id, right.source_x_m, right.end_s):
                continue
            if left.kind == right.kind and left.order < right.order and left.substeps == right.substeps:
                result.append(("p", left, right))
            if left.kind == "curved" and right.kind == "raster" and left.order == right.order and left.substeps == right.substeps:
                result.append(("geometry", left, right))
            if left.kind == right.kind and left.order == right.order and right.substeps == 2 * left.substeps:
                result.append(("dt", left, right))
    return result


def compare(run_paths: Sequence[Path], output: Path, allow_short: bool = False) -> dict[str, object]:
    """Проверить, сопоставить и сохранить новый отчёт; старые продукты защищены."""
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    runs = sorted((load_run(path, allow_short) for path in run_paths), key=lambda item: item.slug)
    comparisons = pairs(runs)
    if not comparisons:
        raise ValueError("Нет согласованных пар порядков, raster/curved либо dt/dt/2")
    # Complete validation before creating any permanent artifacts.
    records = [metrics(value, reference, kind) for kind, value, reference in comparisons]
    output.mkdir(parents=True)
    images = image_directory(output)
    image_paths: list[str] = []
    for kind, value, reference in comparisons:
        slug = f"{kind}_{value.slug}_minus_{reference.slug}"
        for begin in (0.0, 1.0):
            if begin >= value.end_s:
                continue
            destination = images / f"{slug}_{'full' if begin == 0 else 'late'}.png"
            if destination.exists():
                raise FileExistsError(destination)
            render_surface_pair(value, reference, destination, begin)
            image_paths.append(str(destination.relative_to(PROJECT_ROOT)))
        if kind == "p":
            destination = images / f"{slug}_full_far20.png"
            if destination.exists():
                raise FileExistsError(destination)
            render_surface_pair(value, reference, destination, exclude_radius_m=20.0)
            image_paths.append(str(destination.relative_to(PROJECT_ROOT)))
    coverage = [r.slug for r in runs]
    observed_sources = sorted({r.source_x_m for r in runs})
    distinct_cases = {(r.model_id, r.kind, r.order, r.source_x_m) for r in runs}
    expected_orders = (3, 4, 5) if any(r.order == 3 for r in runs) else (4, 5)
    report: dict[str, object] = {
        "chapter": "1.3", "status": "postprocessing_complete", "short_run": allow_short,
        "completed_run_count": len(runs), "observed_source_x_m": observed_sources,
        "expected_main_run_count": 2 * 2 * len(expected_orders) * len(observed_sources),
        "distinct_main_case_count": len(distinct_cases),
        "additional_time_step_run_count": len(runs) - len(distinct_cases),
        "coverage_definition": "Two models, raster/curved, p3/p4/p5 if p3 is present, otherwise p4/p5; only observed source positions; additional time steps counted separately",
        "coverage": coverage, "comparisons": records, "images": image_paths,
        "inputs": [{"path": str(r.path), "metadata_sha256": sha256(r.path / "metadata.json"),
                    "surface_sha256": sha256(r.path / "surface.npz")} for r in runs],
        "limitations": ["Нормы рассчитаны в общих физических приёмниках; поля на разных сетках напрямую не вычитались.",
                        "Прохождение p-критерия требует также временного контроля и проверки отражений; этот отчёт их не подменяет.",
                        "Результаты исследовательские и не являются независимой валидацией решателя."],
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-short", action="store_true", help="Только отдельные smoke-запуски, явно помеченные как короткие")
    parser.add_argument("--single", action="store_true", help="Сейсмограммы одного завершённого опыта без заявления сходимости")
    parser.add_argument("--far20-only", action="store_true", help="Новый дополнительный рисунок с маской 20 м; передать меньший порядок, затем опорный больший")
    args = parser.parse_args(argv)
    if args.far20_only:
        if args.single or len(args.runs) != 2:
            raise ValueError("--far20-only требует два порядка и несовместим с --single")
        far_surface_report(args.runs[0], args.runs[1], args.out)
        print(f"Дополнительная сейсмограмма far20: {args.out / 'report.json'}")
        return
    if args.single:
        if len(args.runs) != 1:
            raise ValueError("--single требует ровно один каталог")
        single_report(args.runs[0], args.out, args.allow_short)
        print(f"Сейсмограммы одного опыта: {args.out / 'report.json'}")
        return
    report = compare(args.runs, args.out, args.allow_short)
    print(f"Сопоставлено запусков: {report['completed_run_count']}; отчёт: {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
