"""Проверить узлы поверхности и масштабы длин волн, не строя расчётные сетки."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from scripts.project_paths import DATA_12, PROJECT_ROOT

FloatArray = NDArray[np.float64]
DOMAIN_WIDTH_M = 700.0
FREQUENCY_HZ = 30.0
NODE_TOLERANCE_M = 1e-9
OUTPUT = PROJECT_ROOT / "data/dev_1_3_point_source/preflight.json"


@dataclass(frozen=True)
class SurfaceLayout:
    """Прямая поверхность с аффинной параметризацией равных элементов."""

    order: int
    element_size_m: float

    def nodes(self) -> FloatArray:
        """Вычислить GLL-узлы из корней производной полинома Лежандра."""
        if self.order < 2 or self.element_size_m <= 0:
            raise ValueError("Нужны p >= 2 и положительная длина элемента")
        count = round(DOMAIN_WIDTH_M / self.element_size_m)
        if abs(count * self.element_size_m - DOMAIN_WIDTH_M) > NODE_TOLERANCE_M:
            raise ValueError("Размер элемента должен делить ширину поверхности")
        internal = np.polynomial.Legendre.basis(self.order).deriv().roots()
        reference = np.asarray(np.r_[-1.0, internal, 1.0], dtype=np.float64)
        local = (reference + 1.0) * (self.element_size_m / 2.0)
        return np.unique((np.arange(count)[:, None] * self.element_size_m + local).ravel())


def point_match(points: FloatArray, nodes: FloatArray) -> dict[str, object]:
    """Сохранить ближайший узел и фактическое расстояние для каждой точки."""
    distances = np.abs(points[:, None] - nodes[None, :])
    nearest_ids = np.argmin(distances, axis=1)
    minimum = distances[np.arange(len(points)), nearest_ids]
    present = minimum <= NODE_TOLERANCE_M
    return {
        "requested_x_m": points.tolist(),
        "nearest_node_x_m": nodes[nearest_ids].tolist(),
        "distance_m": minimum.tolist(),
        "is_node_within_tolerance": present.tolist(),
        "present_count": int(np.count_nonzero(present)),
        "missing_x_m": points[~present].tolist(),
        "maximum_distance_m": float(minimum.max()),
        "all_present": bool(np.all(present)),
    }


def surface_record(layout: SurfaceLayout) -> dict[str, object]:
    """Проверить пять источников и 141 приёмник независимо от решателя."""
    nodes = layout.nodes()
    return {
        "order": layout.order,
        "boundary_element_size_m": layout.element_size_m,
        "boundary_elements": round(DOMAIN_WIDTH_M / layout.element_size_m),
        "surface_nodes": len(nodes),
        "source_match": point_match(np.linspace(0.0, DOMAIN_WIDTH_M, 5), nodes),
        "receiver_match": point_match(np.arange(0.0, DOMAIN_WIDTH_M + 1.0, 5.0), nodes),
    }


def material_record(path: Path, expected_id: int) -> dict[str, object]:
    """Прочесть только контрольное поле и зафиксировать его происхождение."""
    record: dict[str, object] = {
        "model_id": expected_id,
        "dataset": "ECVB",
        "path": str(path.relative_to(PROJECT_ROOT)),
        "exists": path.is_file(),
    }
    if not path.is_file():
        record["status"] = "missing_local_input"
        return record
    record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path, allow_pickle=False) as data:
        model_id = int(data["model_id"])
        if model_id != expected_id:
            raise ValueError(f"Неверный номер модели в {path}: {model_id}")
        for key in ("vp", "vs"):
            values = np.asarray(data[key], dtype=np.float64)
            if values.shape != (70, 70) or not np.all(np.isfinite(values)) or np.any(values <= 0):
                raise ValueError(f"Неверное поле {key} в {path}")
            low, high = float(values.min()), float(values.max())
            record[f"{key}_range_m_s"] = [low, high]
            record[f"{key}_wavelength_at_30hz_m"] = [low / FREQUENCY_HZ, high / FREQUENCY_HZ]
            record[f"layer175_over_{key}_wavelength_at_30hz"] = [175.0 * FREQUENCY_HZ / high, 175.0 * FREQUENCY_HZ / low]
    record["status"] = "read_only_material_check_complete"
    return record


def build_record() -> dict[str, object]:
    """Сформировать свидетельство только выполненных лёгких проверок."""
    return {
        "chapter": "1.3",
        "status": "preflight_complete_dynamic_checks_pending",
        "solver_executed": False,
        "mesh_generated": False,
        "frequency_hz": FREQUENCY_HZ,
        "physical_domain_size_m": [700.0, 700.0],
        "damping_layer_width_m": 175.0,
        "surface_parameterization": "straight_affine_uniform_elements_standard_GLL",
        "node_tolerance_m": NODE_TOLERANCE_M,
        "accepted_decisions": {
            "boundary_element_size_m": 5.0,
            "boundary_elements": 140,
            "solution_orders": [4, 5],
            "source_x_m": [0.0, 175.0, 350.0, 525.0, 700.0],
            "receiver_x_step_m": 5.0,
            "source_normalization": "fixed_resultant_ricker_force",
            "source_peak_force_n_per_m": 1.0,
            "source_work_policy": "measure_only_do_not_equalize",
            "historical_10m_layout_is_comparison_only": True,
        },
        "layouts": [surface_record(SurfaceLayout(order, size)) for size in (10.0, 5.0) for order in (4, 5)],
        "materials": [
            material_record(DATA_12 / "pilot_geometry/selected_model.npz", 12404),
            material_record(DATA_12 / "model65/geometry/model65.npz", 65),
        ],
        "sampling": {
            "seismogram_count": 2001,
            "seismogram_interval_s": 0.001,
            "receiver_count": 141,
            "requested_components": ["ux", "uy", "vx", "vy"],
            "test_field_frame_count": 201,
            "test_field_interval_s": 0.01,
            "includes_t0_and_t2": True,
        },
        "resolved_conflicts": [
            "Пользователь выбрал полное измельчение: прежние 70 верхних элементов по 10 м заменяются 140 элементами по 5 м.",
            "На верхней разбивке 5 м все пять источников и 141 приёмник являются вершинами элементов при обоих порядках p4/p5.",
            "Вариант 70×10 м сохранён только для объяснения отсутствующих серединных GLL-узлов p5.",
            "По последнему уточнению принята одинаковая результирующая сила Рикера F0=1 Н/м; работу источника нужно измерять, но не выравнивать калибровкой.",
        ],
        "pending_checks": [
            "Историческая curved-сетка 00065 не имеет заданной равномерной верхней разбивки; её нельзя просто раздробить и получить требуемый верх 5 м, потребуется отдельная подготовка сетки.",
            "Постоянная результирующая сила устраняет искусственное масштабирование нагрузки сеткой, но не гарантирует одинаковую механическую мощность или работу; измерение работы в динамике ещё не выполнено.",
            "Длина волны при 30 Гц не ограничивает весь спектр Рикера и не доказывает достаточность слоя 175 м; нужны контроль отражения и выбор коэффициента демпфирования.",
        ],
        "limitations": [
            "Выполнены аналитическая проверка GLL на прямой аффинной границе и чтение двух сохранённых полей; расчётные сетки 1.3 не построены.",
            "Геометрия, сборка нагрузки, ABC, демпфирование, устойчивость, сходимость и запись энергии пока не проверены в динамическом расчёте 1.3.",
            "Пять источников подразумевают отдельные эксперименты только после фиксации соответствующей постановки; этот файл не запускает их.",
        ],
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Записать новый JSON; существующее свидетельство защищено по умолчанию."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--overwrite", action="store_true", help="Явно разрешить замену выходного JSON")
    args = parser.parse_args(argv)
    output = args.out.resolve()
    output.relative_to((PROJECT_ROOT / "data").resolve())
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Свидетельство уже существует: {output}; выберите новый --out")
    record = build_record()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Проверка без решателя сохранена: {output}")


if __name__ == "__main__":
    main()
