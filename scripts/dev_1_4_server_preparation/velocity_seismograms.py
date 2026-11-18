"""Сравнение двух сохранённых Vx/Vy70 записей: слой 175 м и ABC на основном домене."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from scripts.project_paths import PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.figure import Figure
import numpy as np
from numpy.typing import NDArray

from scripts.dev_1_4_server_preparation.compare_benchmark import load_run
from scripts.dev_1_4_server_preparation.compare_smoke import file_evidence, metrics
from scripts.dev_1_4_server_preparation.no_padding_compare import compare
from scripts.reporting import mapping, number

FloatArray = NDArray[np.float64]
DATA_ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation"
IMAGE_ROOT = PROJECT_ROOT / "img/dev_1_4_server_preparation"
PADDED = DATA_ROOT / "benchmark_3060_v1/EFVB_00000/x350"
CORE = DATA_ROOT / "no_padding_3060_v1/x350"
COMPONENTS = ("Vx", "Vy")
COLORS = ("#245a9f", "#d65f16", "#683ca4")


def save_figure(figure: Figure, directory: Path, name: str) -> list[dict[str, object]]:
    """Сохранить растровый просмотр и масштабируемый оригинал без перезаписи."""
    result: list[dict[str, object]] = []
    for suffix in ("png", "svg"):
        path = directory / f"{name}.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        figure.savefig(path, dpi=160, facecolor="white")
        result.append(file_evidence(path))
    plt.close(figure)
    return result


def seismograms(times: FloatArray, x: FloatArray, padded: FloatArray, core: FloatArray,
                output: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Одна общая симметричная логарифмическая шкала для шести физических массивов."""
    difference = core - padded
    maximum = float(max(np.max(np.abs(padded)), np.max(np.abs(core)), np.max(np.abs(difference)))) * 1e9
    linear_threshold = .03
    norm = SymLogNorm(linthresh=linear_threshold, linscale=1., vmin=-maximum, vmax=maximum, base=10)
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 11.), sharex=True, sharey=True, layout="constrained")
    names = ("Со слоем 175 м", "Без слоя, ABC на границе 700×700 м", "Разность: без слоя − со слоем")
    mesh = None
    for component, row in enumerate(axes):
        relative = 100 * number(metrics(core[:, :, component], padded[:, :, component])["relative_l2"])
        for column, (axis, array, name) in enumerate(zip(row, (padded, core, difference), names, strict=True)):
            mesh = axis.pcolormesh(x, times, array[:, :, component] * 1e9, cmap="RdBu_r", norm=norm,
                                   shading="nearest", rasterized=True)
            axis.set(xlim=(0., 700.), ylim=(2., 0.), xticks=[0, 175, 350, 525, 700],
                     yticks=np.arange(0., 2.01, .25))
            axis.set_title(f"{COMPONENTS[component]} · {name}", fontsize=11, pad=9)
            if column == 0:
                axis.set_ylabel("Время, с")
            if component == 1:
                axis.set_xlabel("Координата приёмника x, м")
            if column == 2:
                axis.text(.97, .97, f"Отн. L2: {relative:.2f}%", transform=axis.transAxes,
                          ha="right", va="top", fontsize=11,
                          bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": .95})
    if mesh is None:
        raise AssertionError("No seismogram panels")
    colorbar = figure.colorbar(mesh, ax=axes.ravel().tolist(), shrink=.84, pad=.02,
                               ticks=[-10., -1., -.1, -.03, 0., .03, .1, 1., 10.])
    colorbar.ax.set_yticklabels(["−10", "−1", "−0,1", "−0,03", "0", "0,03", "0,1", "1", "10"])
    colorbar.set_label("Скорость / разность скорости, нм/с", labelpad=12)
    figure.suptitle("EFVB 00000 · центральный источник x=350 м · p4 / K96\n"
                   "Все 70 приёмников, 2001 отсчёт · единая симметричная логарифмическая шкала\n"
                   "Линейная зона |v| ≤ 0,03 нм/с; нормировка по трассам и времени не применяется",
                   fontsize=14)
    return save_figure(figure, output, "seismograms"), {
        "units": "nm/s (1e-9 m/s)", "normalization": "one shared SymLogNorm for all six panels",
        "limits_nm_s": [-maximum, maximum], "linear_threshold_nm_s": linear_threshold,
        "no_clipping": True, "no_trace_or_time_normalization": True,
        "difference_sign": "core_minus_padded", "rows": ["Vx", "Vy"],
        "columns": ["padded", "core", "core_minus_padded"], "sample_rendering": "nearest cell, no smoothing"}


def traces(times: FloatArray, x: FloatArray, padded: FloatArray, core: FloatArray,
           output: Path) -> tuple[list[dict[str, object]], list[float]]:
    """Полное окно трёх реальных приёмников; линейная ось в физических единицах."""
    selected = [5., 175., 345.]
    figure, axes = plt.subplots(3, 2, figsize=(15., 10.), sharex=True, layout="constrained")
    for row, coordinate in enumerate(selected):
        indices = np.flatnonzero(x == coordinate)
        if len(indices) != 1:
            raise ValueError("Requested trace is not an exact receiver")
        node = int(indices[0])
        for component in range(2):
            axis = axes[row, component]
            a, b = padded[:, node, component] * 1e9, core[:, node, component] * 1e9
            axis.plot(times, a, color=COLORS[0], lw=1.3, label="Со слоем 175 м")
            axis.plot(times, b, color=COLORS[1], lw=1.1, ls="--", label="Без слоя")
            axis.plot(times, b - a, color=COLORS[2], lw=.8, alpha=.9, label="Разность")
            axis.axhline(0., color="#666666", lw=.4)
            axis.set_title(f"{COMPONENTS[component]} · x={coordinate:g} м", fontsize=12)
            axis.set(ylabel="Скорость, нм/с", xlim=(0., 2.))
            axis.grid(alpha=.18)
            if row == 2:
                axis.set_xlabel("Время, с")
    axes[0, 0].legend(loc="upper right", framealpha=.95, fontsize=10)
    figure.suptitle("Трассы у края, на четверти профиля и рядом с источником\n"
                   "0–2 с, линейные шкалы в нм/с · амплитуды не нормированы; пределы оси Y указаны в каждой панели",
                   fontsize=14)
    return save_figure(figure, output, "traces"), selected


def difference_over_time(times: FloatArray, padded: FloatArray, core: FloatArray,
                         output: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Абсолютный RMS по приёмникам и накопленная доля квадрата полной разности."""
    difference = core - padded
    rms = [np.sqrt(np.mean(array**2, axis=1)) for array in (padded, core, difference)]
    figure, axes = plt.subplots(3, 1, figsize=(13.5, 10.5), sharex=True, layout="constrained")
    for component in range(2):
        axis = axes[component]
        for values, label, color, style in zip(rms, ("Со слоем 175 м", "Без слоя", "Разность"),
                                               COLORS, ("-", "--", "-"), strict=True):
            y = values[:, component] * 1e9
            axis.plot(times, np.where(y > 0, y, np.nan), color=color, label=label, ls=style, lw=1.4)
        axis.set(yscale="log", ylim=(1e-5, 3.), ylabel=f"RMS {COMPONENTS[component]}, нм/с", xlim=(0., 2.))
        axis.grid(alpha=.2, which="both")
        axis.set_title(f"{COMPONENTS[component]} · RMS по 70 приёмникам в каждый момент времени", fontsize=12)
    axes[0].legend(loc="upper right", ncols=3)
    squared_by_time = np.sum(difference**2, axis=(1, 2))
    total = float(np.sum(squared_by_time))
    if total <= 0:
        raise ValueError("The diagnostic expects a nonzero difference")
    cumulative = np.cumsum(squared_by_time) / total
    axes[2].plot(times, cumulative * 100, color=COLORS[2], lw=2)
    axes[2].set(xlabel="Время, с", ylabel="Накопленная доля, %", ylim=(0., 102.), yticks=[0, 25, 50, 75, 100])
    axes[2].grid(alpha=.2)
    axes[2].set_title("Доля накопленной суммы (ΔVx² + ΔVy²) от её значения за полные 0–2 с", fontsize=12)
    quantiles: dict[str, float] = {}
    for fraction in (.1, .5, .9):
        index = int(np.searchsorted(cumulative, fraction))
        time = float(times[index])
        quantiles[str(fraction)] = time
        axes[2].plot(time, fraction * 100, "o", color=COLORS[2], ms=5)
        axes[2].annotate(f"{fraction * 100:g}%: {time:.3f} с", (time, fraction * 100),
                         xytext=(8, 8), textcoords="offset points", fontsize=10)
    figure.suptitle("Когда меняется запись после удаления слоя\n"
                   "Абсолютная разность без деления на почти нулевой сигнал; нули на логарифмической оси не показаны",
                   fontsize=14)
    peak_index = int(np.argmax(np.sqrt(np.mean(difference**2, axis=(1, 2)))))
    return save_figure(figure, output, "difference_over_time"), {
        "rms_definition": "sqrt(mean over 70 receivers of component squared), at each recorded time",
        "rms_plot_lower_limit_nm_s": 1e-5,
        "cumulative_definition": "sum(deltaVx**2+deltaVy**2) through time / sum over all 2001 times and 70 receivers",
        "cumulative_fraction_times_s": quantiles, "maximum_difference_rms_time_s": float(times[peak_index]),
        "maximum_difference_rms_m_s": float(np.sqrt(np.mean(difference[peak_index]**2)))}


def build(images: Path, record_path: Path) -> dict[str, object]:
    images, record_path = images.resolve(), record_path.resolve()
    images.relative_to(IMAGE_ROOT)
    record_path.relative_to(DATA_ROOT)
    if images.exists() or record_path.exists():
        raise FileExistsError("Use a fresh visualization version")
    comparison = compare(CORE, PADDED)
    padded, core = load_run(PADDED).surface, load_run(CORE).surface
    if not np.array_equal(padded.times, core.times) or not np.array_equal(padded.coordinates, core.coordinates):
        raise ValueError("Mismatched recording grids")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 12,
                         "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    images.mkdir(parents=True, exist_ok=False)
    times, x = padded.times, padded.coordinates[:, 0]
    artifacts, scale = seismograms(times, x, padded.velocity, core.velocity, images)
    trace_files, trace_positions = traces(times, x, padded.velocity, core.velocity, images)
    time_files, time_summary = difference_over_time(times, padded.velocity, core.velocity, images)
    artifacts.extend(trace_files + time_files)
    difference = core.velocity - padded.velocity
    full_difference_sum = float(np.sum(difference**2))
    windows: list[dict[str, object]] = []
    for lower, upper in ((0., .5), (.5, 1.), (1., 1.5), (1.5, 2.)):
        selected = (times >= lower) & ((times < upper) if upper < 2. else (times <= upper))
        window_metrics = metrics(core.velocity[selected], padded.velocity[selected])
        windows.append({"time_interval_s": [lower, upper], "right_endpoint_included": upper == 2.,
                        "sample_count": int(np.count_nonzero(selected)), "metrics": window_metrics,
                        "fraction_of_total_squared_difference": float(np.sum(difference[selected]**2)) / full_difference_sum})
    report: dict[str, object] = {"schema": "chapter14.velocity_seismograms.v1", "status": "completed",
        "utc": datetime.now(timezone.utc).isoformat(), "source_runs": {"padded": comparison["reference"], "core": comparison["core"]},
        "full_window_metrics": comparison["metrics"], "component_metrics": comparison["components"],
        "shape": list(core.velocity.shape), "time_extent_s": [0., 2.], "receiver_count": 70,
        "scale": scale, "trace_receiver_x_m": trace_positions, "time_diagnostics": time_summary,
        "windows": windows,
        "relative_l2_definition": "norm(core-padded) / norm(padded), over all selected time samples, receivers and components; window metrics use their own padded window denominator",
        "interpretation": "Changed boundary conditions; differences are not numerical error against an exact solution or ABC accuracy validation",
        "plots": artifacts, "code": file_evidence(Path(__file__)),
        "comparison_evidence": file_evidence(DATA_ROOT / "no_padding_3060_v1/comparison.json")}
    if not np.isclose(sum(number(window["fraction_of_total_squared_difference"]) for window in windows), 1., atol=1e-14):
        raise AssertionError("Disjoint windows do not partition the full difference")
    if sum(int(number(window["sample_count"])) for window in windows) != 2001:
        raise AssertionError("Disjoint windows do not cover all samples")
    record_path.parent.mkdir(parents=True, exist_ok=True)
    with record_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=IMAGE_ROOT / "velocity_comparison_v1")
    parser.add_argument("--record", type=Path, default=DATA_ROOT / "velocity_visualization_v1/seismograms.json")
    arguments = parser.parse_args()
    report = build(arguments.images, arguments.record)
    print(json.dumps({"status": report["status"], "metrics": report["full_window_metrics"],
                      "time_diagnostics": report["time_diagnostics"], "images": str(arguments.images)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
