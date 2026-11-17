"""Физические соглашения главы 1.3 без импорта нативного решателя."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Размеры в метрах; y направлен вверх, свободная поверхность y=0."""

    size_m: float = 700.0
    padding_m: float = 175.0
    frequency_hz: float = 30.0
    center_s: float = 0.05
    end_s: float = 2.0
    output_interval_s: float = 0.001
    field_interval_s: float = 0.01
    source_x_m: tuple[float, ...] = (0.0, 175.0, 350.0, 525.0, 700.0)
    density_kg_m3: float = 1000.0
    force_amplitude_n_per_m: float = 1.0

    def __post_init__(self) -> None:
        for value in (self.size_m, self.padding_m, self.frequency_hz, self.end_s,
                      self.output_interval_s, self.field_interval_s, self.density_kg_m3,
                      self.force_amplitude_n_per_m):
            if not isfinite(value) or value <= 0.0:
                raise ValueError("Размеры, частота, интервалы и плотность должны быть положительными")
        if not isfinite(self.center_s) or not 0.0 < self.center_s < self.end_s:
            raise ValueError("Центр импульса должен находиться внутри временного интервала")
        if not self.source_x_m or any(not isfinite(x) or not 0.0 <= x <= self.size_m for x in self.source_x_m):
            raise ValueError("Источники должны лежать на верхней грани физической области")
        if len(set(self.source_x_m)) != len(self.source_x_m):
            raise ValueError("Положения источников должны быть различны")


@dataclass(frozen=True, slots=True)
class NodalPointForce:
    """Нагрузка одного глобального узла; значения уже являются узловой силой."""

    node_id: int
    force_x: float
    force_y: float


def inclusive_times(end_s: float, interval_s: float) -> FloatArray:
    """Построить равномерную запись с обоими концами, без округления длительности."""
    if not isfinite(end_s) or not isfinite(interval_s) or end_s <= 0.0 or interval_s <= 0.0:
        raise ValueError("Длительность и интервал должны быть конечными и положительными")
    intervals = round(end_s / interval_s)
    if intervals < 1 or not np.isclose(intervals * interval_s, end_s, rtol=0.0, atol=1e-12):
        raise ValueError("Длительность должна быть целым числом интервалов записи")
    return np.linspace(0.0, end_s, intervals + 1, dtype=np.float64)


def ricker(times_s: FloatArray, spec: ExperimentSpec = ExperimentSpec()) -> FloatArray:
    """Рикер с единичным положительным пиком; амплитуда силы задаётся отдельно."""
    if not np.isfinite(times_s).all():
        raise ValueError("Время должно быть конечным")
    phase = np.pi * spec.frequency_hz * (times_s - spec.center_s)
    return np.asarray((1.0 - 2.0 * phase**2) * np.exp(-phase**2), dtype=np.float64)


def _coordinates(nodes_xy_m: FloatArray) -> None:
    if nodes_xy_m.ndim != 2 or nodes_xy_m.shape[1] != 2 or not np.isfinite(nodes_xy_m).all():
        raise ValueError("Ожидаются конечные координаты формы (n, 2)")


def nodal_point_force(nodes_xy_m: FloatArray, source_x_m: float, amplitude: float,
                      spec: ExperimentSpec = ExperimentSpec()) -> NodalPointForce:
    """Найти единственный узел (x,0) и задать силу вниз без масштабирования массой.

    Допуск 1e-9 м учитывает лишь округление координат: выбора ближайшего узла
    или интерполяционного распределения здесь нет. В двумерной постановке
    амплитуда соответствует силе на единицу толщины.
    """
    _coordinates(nodes_xy_m)
    if not isfinite(source_x_m) or not 0.0 <= source_x_m <= spec.size_m:
        raise ValueError("Источник должен находиться на верхе физической области")
    if not isfinite(amplitude) or amplitude <= 0.0:
        raise ValueError("Амплитуда должна быть конечной и положительной")
    ids = np.flatnonzero(np.all(np.abs(nodes_xy_m - [source_x_m, 0.0]) <= 1e-9, axis=1))
    if ids.size != 1:
        raise ValueError(f"Нужен единственный точный узел источника, найдено {ids.size}")
    return NodalPointForce(node_id=int(ids[0]), force_x=0.0, force_y=-amplitude)


def damping_alpha(nodes_xy_m: FloatArray, alpha_max_s_inv: float,
                  spec: ExperimentSpec = ExperimentSpec()) -> FloatArray:
    """Квадратичный массовый коэффициент α [1/с]; в нижних углах вклады суммируются.

    Используется соглашение C=αM, а не C=2αM. В углах α достигает 2α_max.
    Верх боковых продолжений демпфируется объёмно, но остаётся свободной гранью.
    """
    _coordinates(nodes_xy_m)
    if not isfinite(alpha_max_s_inv) or alpha_max_s_inv < 0.0:
        raise ValueError("Коэффициент демпфирования должен быть конечным и неотрицательным")
    x, y = nodes_xy_m[:, 0], nodes_xy_m[:, 1]
    tolerance = 1e-9
    if np.any(x < -spec.padding_m - tolerance) or np.any(x > spec.size_m + spec.padding_m + tolerance) or np.any(y > tolerance) or np.any(y < -spec.size_m - spec.padding_m - tolerance):
        raise ValueError("Узел находится вне расчётной области с демпфирующим слоем")
    x = np.clip(x, -spec.padding_m, spec.size_m + spec.padding_m)
    y = np.clip(y, -spec.size_m - spec.padding_m, 0.0)
    sx = np.maximum.reduce((-x, x - spec.size_m, np.zeros_like(x))) / spec.padding_m
    sy = np.maximum(-y - spec.size_m, 0.0) / spec.padding_m
    return np.asarray(alpha_max_s_inv * (sx**2 + sy**2), dtype=np.float64)


def material_coordinates(nodes_xy_m: FloatArray,
                         spec: ExperimentSpec = ExperimentSpec()) -> FloatArray:
    """Координаты выборки материала при плоскопараллельном продолжении.

    По бокам берётся граничный профиль, снизу — нижний профиль; в углах —
    соответствующее угловое значение. Выбор материала на интерфейсе должен
    оставаться односторонним для конкретного элемента, а не усредняться.
    """
    _coordinates(nodes_xy_m)
    damping_alpha(nodes_xy_m, 0.0, spec)
    return np.column_stack((np.clip(nodes_xy_m[:, 0], 0.0, spec.size_m),
                            np.clip(nodes_xy_m[:, 1], -spec.size_m, 0.0)))


def source_work(times_s: FloatArray, force_xy: FloatArray, velocity_xy: FloatArray) -> float:
    """Работа ∫F·v dt по внутренним шагам, включая отрицательную мощность.

    Сила и скорость должны относиться к одному источнику, физическому времени
    и соглашению о толщине. Это диагностическая квадратура; её точность следует
    проверять уменьшением внутреннего шага. Записи через 1 мс недостаточно
    считать автоматически точным энергетическим балансом интегратора.
    """
    if times_s.ndim != 1 or times_s.size < 2 or not np.isfinite(times_s).all() or np.any(np.diff(times_s) <= 0.0):
        raise ValueError("Времена должны быть конечны и строго возрастать")
    expected = (times_s.size, 2)
    if force_xy.shape != expected or velocity_xy.shape != expected:
        raise ValueError("Сила и скорость должны иметь форму (число времён, 2)")
    if not np.isfinite(force_xy).all() or not np.isfinite(velocity_xy).all():
        raise ValueError("Сила и скорость должны быть конечными")
    power = np.sum(force_xy * velocity_xy, axis=1)
    return float(np.trapezoid(power, times_s))
