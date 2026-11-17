"""Потоковая анимация рассчитанных полей скорости и полной плотности энергии."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Sequence

from scripts.project_paths import PROJECT_ROOT, image_directory
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.colors import LogNorm, Normalize
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from scripts.dev_1_3_point_source.compare import RunData, load_run, sha256

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class FieldSeries:
    """Пути к кадрам и постоянная карта отображения, без истории полей в RAM."""

    run: RunData
    frame_paths: tuple[Path, ...]
    node_count: int
    pixel_node_ids: IntArray
    pixel_shape: tuple[int, int]
    maximum_pixel_distance_m: float


@dataclass(frozen=True, slots=True)
class Frame:
    """Один физический кадр; после записи изображения освобождается."""

    time_s: float
    speed_m_s: FloatArray
    energy_density_j_m3: FloatArray


def prepare_series(run: RunData, pixel_width: int) -> FieldSeries:
    """Проверить геометрию и построить nearest-node карту регулярных пикселей."""
    geometry_path = run.path / "fields/geometry.npz"
    with np.load(geometry_path, allow_pickle=False) as data:
        coordinates = np.asarray(data["coordinates_m"], dtype=np.float64)
        cells = np.asarray(data["cells"], dtype=np.int64)
    if coordinates.ndim != 2 or coordinates.shape[1] != 2 or not np.isfinite(coordinates).all():
        raise ValueError(f"Неверные координаты поля: {run.path}")
    if cells.ndim != 2 or cells.shape[1] != (run.order + 1)**2 or cells.min() < 0 or cells.max() >= len(coordinates):
        raise ValueError(f"Неверная связность поля: {run.path}")
    if not np.allclose(coordinates.min(axis=0), [-175, -875], rtol=0, atol=1e-7) or not np.allclose(coordinates.max(axis=0), [875, 0], rtol=0, atol=1e-7):
        raise ValueError("Анимация ожидает расширенную область 1050×875 м")
    if pixel_width < 140:
        raise ValueError("Для полного домена нужно не менее 140 пикселей по X")
    pixel_height = round(pixel_width * 875 / 1050)
    x = -175 + (np.arange(pixel_width) + 0.5) * 1050 / pixel_width
    y = -875 + (np.arange(pixel_height) + 0.5) * 875 / pixel_height
    xx, yy = np.meshgrid(x, y)
    distances, ids = cKDTree(coordinates).query(np.column_stack((xx.ravel(), yy.ravel())), workers=1)
    frames = tuple(sorted((run.path / "fields").glob("frame_*.npz")))
    expected_count = round(run.end_s / 0.01) + 1
    expected_names = tuple(f"frame_{index:04d}.npz" for index in range(expected_count))
    if tuple(path.name for path in frames) != expected_names:
        raise ValueError(f"Ожидались все {expected_count} кадров поля без пропусков: {run.path}")
    return FieldSeries(run, frames, len(coordinates), np.asarray(ids, dtype=np.int64),
                       (pixel_height, pixel_width), float(np.max(distances)))


def read_frame(series: FieldSeries, index: int) -> Frame:
    """Прочесть только скорость и две плотности энергии одного кадра."""
    with np.load(series.frame_paths[index], allow_pickle=False) as data:
        time_s = float(data["time_s"])
        velocity = np.asarray(data["velocity_m_s"], dtype=np.float64)
        kinetic = np.asarray(data["kinetic_energy_density_j_m3"], dtype=np.float64)
        elastic = np.asarray(data["elastic_energy_density_j_m3"], dtype=np.float64)
    if not np.isclose(time_s, index * 0.01, rtol=0, atol=1e-12):
        raise ValueError(f"Неверное время кадра {series.frame_paths[index]}")
    if velocity.shape != (series.node_count, 2) or kinetic.shape != (series.node_count,) or elastic.shape != kinetic.shape:
        raise ValueError(f"Поле не содержит все узлы: {series.frame_paths[index]}")
    if not all(np.isfinite(value).all() for value in (velocity, kinetic, elastic)):
        raise ValueError(f"Неконечные значения: {series.frame_paths[index]}")
    for component in (kinetic, elastic):
        tolerance = max(float(np.max(np.abs(component))) * 1e-10, np.finfo(float).tiny)
        if np.min(component) < -tolerance:
            raise ValueError(f"Существенно отрицательная плотность энергии: {series.frame_paths[index]}")
    speed = np.hypot(velocity[:, 0], velocity[:, 1])
    return Frame(time_s, speed, np.maximum(kinetic + elastic, 0.0))


def scan_scales(series: Sequence[FieldSeries]) -> tuple[float, float]:
    """Найти глобальные максимумы по всем узлам/кадрам всех сравниваемых сеток."""
    speed_max = 0.0
    energy_max = 0.0
    for item in series:
        for index in range(len(item.frame_paths)):
            frame = read_frame(item, index)
            speed_max = max(speed_max, float(np.max(frame.speed_m_s)))
            energy_max = max(energy_max, float(np.max(frame.energy_density_j_m3)))
        print(f"Шкалы проверены: {item.run.slug}", flush=True)
    return speed_max, energy_max


def animate(run_paths: Sequence[Path], output: Path, pixel_width: int = 630,
            fps: int = 25, allow_short: bool = False, log_scale: bool = False) -> dict[str, object]:
    """Записать оба ролика в одном проходе по кадрам после определения общих шкал."""
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("Для MP4 требуется доступный ffmpeg")
    if not 1 <= len(run_paths) <= 4 or not 1 <= fps <= 120:
        raise ValueError("Нужны 1–4 сравниваемых серии и 1–120 кадров/с")
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    runs = [load_run(path, allow_short) for path in run_paths]
    if len({run.path for run in runs}) != len(runs):
        raise ValueError("Один запуск передан несколько раз")
    if len({(run.model_id, run.source_x_m, run.end_s) for run in runs}) != 1:
        raise ValueError("Общие шкалы предназначены для одной модели, одного источника и времени")
    series = [prepare_series(run, pixel_width) for run in runs]
    speed_max, energy_max = scan_scales(series)
    if log_scale and (speed_max <= 0 or energy_max <= 0):
        raise ValueError("Логарифмическая анимация требует ненулевых скорости и энергии хотя бы в одном кадре")
    output.mkdir(parents=True)
    images_dir = image_directory(output)
    for filename in ("speed.mp4", "energy.mp4", "speed_final.png", "energy_final.png"):
        if (images_dir / filename).exists():
            raise FileExistsError(images_dir / filename)
    figures: list[plt.Figure] = []
    artists: list[list[AxesImage]] = []
    time_labels: list[Text] = []
    for name, unit, maximum, floor_fraction in (("Модуль скорости частиц", "м/с", speed_max, 1e-6),
                                               ("Полная плотность энергии", "Дж/м³", energy_max, 1e-12)):
        fig, axes = plt.subplots(1, len(series), figsize=(max(7.0, 5.0 * len(series)), 5.5), squeeze=False, layout="constrained")
        row_artists: list[AxesImage] = []
        norm = LogNorm(vmin=maximum * floor_fraction, vmax=maximum) if log_scale else Normalize(vmin=0, vmax=maximum or 1.0)
        cmap = matplotlib.colormaps["inferno"].copy()
        cmap.set_bad("black")
        for ax, item in zip(axes[0], series, strict=True):
            artist = ax.imshow(np.zeros(item.pixel_shape), extent=(-175, 875, -875, 0), origin="lower",
                               norm=norm, cmap=cmap, interpolation="nearest")
            ax.add_patch(Rectangle((0, -700), 700, 700, fill=False, edgecolor="cyan", linewidth=0.7, linestyle="--"))
            ax.plot(item.run.source_x_m, 0, marker="v", color="cyan", markersize=5, clip_on=False)
            ax.set(xlabel="x, м", ylabel="y, м", title=f"{item.run.kind}, p={item.run.order}",
                   xlim=(-175, 875), ylim=(-875, 0))
            row_artists.append(artist)
        fig.colorbar(row_artists[-1], ax=axes.ravel().tolist(), label=unit, shrink=0.85, format="%.1e")
        title = fig.suptitle(f"{name} · ECVB {runs[0].model_id:05d} · xₛ={runs[0].source_x_m:g} м · t=0,00 с", fontsize=11)
        scale_note = (f"Общая постоянная логарифмическая шкала\nНижний предел {maximum * floor_fraction:.2e} {unit}; нули чёрные"
                      if log_scale else "Общая постоянная линейная шкала")
        fig.supxlabel(scale_note + "\nПунктир — основная область; выборка ближайшего узла", fontsize=8)
        figures.append(fig)
        artists.append(row_artists)
        time_labels.append(title)
    filenames = ("speed.mp4", "energy.mp4")
    writer_objects = [FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"],
                                  metadata={"title": title}) for title in ("Particle velocity", "Total energy density")]
    try:
        with ExitStack() as stack:
            for writer, fig, filename in zip(writer_objects, figures, filenames, strict=True):
                stack.enter_context(writer.saving(fig, str(images_dir / filename), dpi=100))
            for index in range(len(series[0].frame_paths)):
                for column, item in enumerate(series):
                    frame = read_frame(item, index)
                    for row, values in enumerate((frame.speed_m_s, frame.energy_density_j_m3)):
                        artists[row][column].set_data(values[item.pixel_node_ids].reshape(item.pixel_shape))
                for label, name in zip(time_labels, ("Модуль скорости частиц", "Полная плотность энергии"), strict=True):
                    label.set_text(f"{name} · ECVB {runs[0].model_id:05d} · xₛ={runs[0].source_x_m:g} м · t={index * 0.01:.2f} с")
                for writer in writer_objects:
                    writer.grab_frame()
                if index % 20 == 0:
                    print(f"Анимация: кадр {index + 1}/{len(series[0].frame_paths)}", flush=True)
            for fig, filename in zip(figures, ("speed_final.png", "energy_final.png"), strict=True):
                fig.savefig(images_dir / filename, dpi=140)
    finally:
        for fig in figures:
            plt.close(fig)
    record: dict[str, object] = {
        "status": "animation_complete", "short_run": allow_short, "model_id": runs[0].model_id,
        "source_x_m": runs[0].source_x_m, "frame_count": len(series[0].frame_paths),
        "physical_interval_s": 0.01, "end_s": runs[0].end_s, "playback_fps": fps,
        "common_limits": {"speed_m_s": [0.0, speed_max], "total_energy_density_j_m3": [0.0, energy_max]},
        "color_scale": "logarithmic" if log_scale else "linear",
        "log_color_limits": ({"speed_m_s": [speed_max * 1e-6, speed_max],
                              "total_energy_density_j_m3": [energy_max * 1e-12, energy_max]} if log_scale else None),
        "log_zero_policy": "Zeros are masked and drawn black; positive values below the fixed lower limit use the lowest colormap color" if log_scale else None,
        "zero_field_display_limit": "1 only if every actual value is zero; recorded maxima remain zero",
        "rendering": "nearest_global_node_on_common_regular_pixel_centres_no_temporal_rescaling",
        "pixel_shape": series[0].pixel_shape,
        "inputs": [{"path": str(item.run.path), "metadata_sha256": sha256(item.run.path / "metadata.json"),
                    "geometry_sha256": sha256(item.run.path / "fields/geometry.npz"),
                    "node_count": item.node_count, "max_pixel_to_node_distance_m": item.maximum_pixel_distance_m}
                   for item in series],
        "outputs": [str((images_dir / name).relative_to(PROJECT_ROOT)) for name in (*filenames, "speed_final.png", "energy_final.png")],
        "limits": ["Цветная картинка является пространственной выборкой, а не всеми исходными значениями; все узловые поля сохранены отдельно.",
                   "Плотность энергии у материальных интерфейсов является узловым восстановлением решателя.",
                   "Одна постоянная шкала может скрывать слабую позднюю коду; временного усиления в этих роликах нет."],
    }
    (output / "animation.json").write_text(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pixel-width", type=int, default=630)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--log-scale", action="store_true", help="Постоянная общая LogNorm: speed 1e-6…1 и energy 1e-12…1 от максимума, без усиления во времени")
    args = parser.parse_args(argv)
    result = animate(args.runs, args.out, args.pixel_width, args.fps, args.allow_short, args.log_scale)
    print(f"Записаны два ролика по {result['frame_count']} кадров; {args.out / 'animation.json'}")


if __name__ == "__main__":
    main()
