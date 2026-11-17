"""Локализация разности порядков у точечного источника по завершённым записям."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.dev_1_3_point_source.compare import COMPONENTS, UNITS, FloatArray, RunData, compatible, load_run, sha256
from scripts.project_paths import PROJECT_ROOT


def localization(value: FloatArray, reference: FloatArray, coordinates: FloatArray,
                 source_x_m: float) -> dict[str, object]:
    """Доли квадрата дискретной ошибки, источник и все остальные приёмники."""
    if value.shape != reference.shape or value.ndim != 3 or value.shape[2] != 4 or coordinates.shape != (value.shape[1], 2):
        raise ValueError("Нужны согласованные массивы (time,receiver,4) и координаты")
    if not all(np.isfinite(array).all() for array in (value, reference, coordinates)):
        raise ValueError("Все данные должны быть конечными")
    source_ids = np.flatnonzero((np.abs(coordinates[:, 0] - source_x_m) <= 1e-9) & (np.abs(coordinates[:, 1]) <= 1e-9))
    if source_ids.size != 1:
        raise ValueError("Нужен ровно один поверхностный приёмник в узле источника")
    source_index = int(source_ids[0])
    difference = value - reference
    squared_by_receiver = np.sum(difference**2, axis=0)
    squared_reference = np.sum(reference**2, axis=0)
    total_squared = squared_by_receiver.sum(axis=0)
    full_reference_norm = np.sqrt(squared_reference.sum(axis=0))
    threshold = np.maximum(full_reference_norm * 1e-12, np.finfo(float).tiny)
    outside = np.arange(len(coordinates)) != source_index
    components: dict[str, object] = {}
    for index, (name, unit) in enumerate(zip(COMPONENTS, UNITS, strict=True)):
        source_error = float(np.sqrt(squared_by_receiver[source_index, index]))
        source_ref = float(np.sqrt(squared_reference[source_index, index]))
        outside_error = float(np.sqrt(squared_by_receiver[outside, index].sum()))
        outside_ref = float(np.sqrt(squared_reference[outside, index].sum()))
        components[name] = {
            "unit": unit,
            "source_share_of_squared_difference": float(squared_by_receiver[source_index, index] / total_squared[index]) if total_squared[index] > 0 else None,
            "source_only_relative_l2": source_error / source_ref if source_ref > threshold[index] else None,
            "source_only_difference_l2": source_error, "source_only_reference_l2": source_ref,
            "excluding_only_source_relative_l2": outside_error / outside_ref if outside_ref > threshold[index] else None,
            "excluding_only_source_difference_l2": outside_error,
            "excluding_only_source_reference_l2": outside_ref,
            "all_receiver_difference_l2": float(np.sqrt(total_squared[index])),
        }
    per_receiver: list[dict[str, object]] = []
    for receiver, coordinate in enumerate(coordinates):
        norms = np.sqrt(squared_by_receiver[receiver])
        refs = np.sqrt(squared_reference[receiver])
        per_receiver.append({
            "x_m": float(coordinate[0]), "is_source": receiver == source_index,
            "difference_l2": norms.tolist(), "reference_l2": refs.tolist(),
            "relative_l2": [float(norms[k] / refs[k]) if refs[k] > threshold[k] else None for k in range(4)],
            "share_of_squared_difference": [float(squared_by_receiver[receiver, k] / total_squared[k]) if total_squared[k] > 0 else None for k in range(4)],
        })
    return {"component_order": COMPONENTS, "source_receiver_index": source_index,
            "components": components, "per_receiver": per_receiver,
            "normalization": "squared differences summed over common time samples; receiver fractions sum to one for each nonzero-error component"}


def source_force_comparison(value: RunData, reference: RunData) -> dict[str, object]:
    """Проверить заданную силу, включая исходный нулевой отсчёт, на общем шаге."""
    compatible(value, reference, "p")
    force_records: list[dict[str, object]] = []
    forces: list[FloatArray] = []
    times: list[FloatArray] = []
    for run in (value, reference):
        with np.load(run.path / "source_history.npz", allow_pickle=False) as data:
            force = np.asarray(data["force_y_n_per_m"], dtype=np.float64)
            time = np.asarray(data["time_s"], dtype=np.float64)
        expected_count = 2000 * run.substeps + 1
        if (run.end_s != 2.0 or time.shape != (expected_count,) or force.shape != time.shape
                or not np.isfinite(force).all() or not np.isfinite(time).all()
                or not np.allclose(time, np.linspace(0, 2.0, expected_count), rtol=0, atol=1e-12)):
            raise ValueError("Нужна полная внутренняя история силы")
        force_records.append({"order": run.order, "minimum_force_y_n_per_m": float(force.min()),
                              "maximum_abs_force_n_per_m": float(np.max(np.abs(force))),
                              "initial_time_s": float(time[0]), "initial_force_y_n_per_m": float(force[0]),
                              "initial_force_is_zero": bool(force[0] == 0.0),
                              "main_peak_time_s": float(time[np.argmin(force)]),
                              "source_history_sha256": sha256(run.path / "source_history.npz")})
        forces.append(force)
        times.append(time)
    if not np.allclose(times[0], times[1], atol=1e-12, rtol=0):
        raise ValueError("Истории силы относятся к разным внутренним временам")
    return {"runs": force_records, "identical_history": bool(np.array_equal(forces[0], forces[1])),
            "maximum_abs_difference_n_per_m": float(np.max(np.abs(forces[0] - forces[1]))),
            "comparison_includes_initial_sample": True}


def diagnose(value_path: Path, reference_path: Path, output: Path) -> dict[str, object]:
    """Сохранить новое свидетельство меньшего порядка относительно большего."""
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    value, reference = load_run(value_path), load_run(reference_path)
    force_report = source_force_comparison(value, reference)
    report: dict[str, object] = {
        "status": "completed_source_localization", "model_id": value.model_id, "kind": value.kind,
        "source_x_m": value.source_x_m, "time_window_s": [0.0, 2.0],
        "value": str(value.path), "reference": str(reference.path),
        "value_order": value.order, "reference_order": reference.order,
        "surface": localization(value.surface, reference.surface, value.coordinates_m, value.source_x_m),
        "source_force": force_report,
        "inputs": [{"path": str(run.path), "metadata_sha256": sha256(run.path / "metadata.json"),
                    "surface_sha256": sha256(run.path / "surface.npz")} for run in (value, reference)],
        "interpretation_limit": "Эмпирическая локализация наблюдаемой ошибки и проверка одинаковой силы; это не доказательство теоретической причины ошибки или сходимости точечного значения.",
    }
    output.mkdir(parents=True)
    with (output / "point_source_diagnostic.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    value_group = parser.add_mutually_exclusive_group(required=True)
    value_group.add_argument("--value", type=Path, help="Завершённый запуск меньшего порядка")
    value_group.add_argument("--p4", type=Path, help="Совместимый прежний аргумент для p4")
    reference_group = parser.add_mutually_exclusive_group(required=True)
    reference_group.add_argument("--reference", type=Path, help="Завершённый запуск большего порядка")
    reference_group.add_argument("--p5", type=Path, help="Совместимый прежний аргумент для p5")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.p4 is not None and load_run(args.p4).order != 4:
        raise ValueError("Прежний --p4 разрешён только для порядка 4; используйте --value")
    if args.p5 is not None and load_run(args.p5).order != 5:
        raise ValueError("Прежний --p5 разрешён только для порядка 5; используйте --reference")
    report = diagnose(args.value or args.p4, args.reference or args.p5, args.out)
    print(f"Локализация: {report['status']}; {args.out / 'point_source_diagnostic.json'}")


if __name__ == "__main__":
    main()
