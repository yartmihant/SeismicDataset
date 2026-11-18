"""Полные анимации Vx/Vy и модуля скорости: слой 175 м против ABC на основной области."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import cast

from scripts.project_paths import PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from matplotlib.text import Text
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree  # type: ignore[import-untyped]

from scripts.reporting import mapping, number, read_json, sequence

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
FULL_EXTENT = (-175., 875., -875., 0.)
CORE_EXTENT = (0., 700., -700., 0.)
PREVIEW_INDICES = (25, 60, 120, 200)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Fields:
    directory: Path
    metadata: dict[str, object]
    coordinates: FloatArray
    times: FloatArray
    velocity: FloatArray
    files: dict[str, object]


@dataclass(frozen=True)
class Pixels:
    indices: IntArray
    shape: tuple[int, int]
    extent: tuple[float, float, float, float]
    maximum_distance_m: float


@dataclass
class Movie:
    name: str
    figure: plt.Figure
    artists: list[AxesImage]
    time_label: Text
    title: str


def load_fields(directory: Path, padded: bool) -> Fields:
    metadata = read_json(directory / "metadata.json")
    if metadata["status"] != "completed" or metadata.get("field_components") != ["Vx", "Vy"]:
        raise ValueError("Need a completed two-component velocity field run")
    counts = mapping(metadata["recorded_counts"])
    if counts.get("fields") != 201 or counts.get("surface") != 2001 or metadata.get("field_cadence_s") != .01:
        raise ValueError("Need the full 0–2 s field/surface recording")
    spec = mapping(metadata["run_spec"])
    for key, expected in {"dataset": "EFVB", "model_id": 0, "source_x_m": 350., "substeps": 96, "order": 4, "t_end_s": 2.}.items():
        if spec.get(key) != expected:
            raise ValueError(f"Unexpected field run parameter {key}")
    files = mapping(metadata["field_files"])
    evidence: dict[str, object] = {}
    for name in ("nodes.npy", "field_time_s.npy", "velocity_m_s.npy"):
        path = directory / name
        actual = sha256(path)
        recorded = mapping(files[name])
        if actual != recorded["sha256"] or path.stat().st_size != recorded["bytes"]:
            raise ValueError(f"Field evidence mismatch: {path}")
        evidence[name] = {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}
    coordinates = cast(FloatArray, np.load(directory / "nodes.npy", mmap_mode="r", allow_pickle=False))
    times = cast(FloatArray, np.load(directory / "field_time_s.npy", allow_pickle=False))
    velocity = cast(FloatArray, np.load(directory / "velocity_m_s.npy", mmap_mode="r", allow_pickle=False))
    count = 589541 if padded else 314721
    if coordinates.shape != (count, 2) or velocity.shape != (201, count, 2) or times.shape != (201,):
        raise ValueError("Unexpected field array shape")
    if any(value.dtype != np.float64 for value in (coordinates, times, velocity)):
        raise ValueError("Full fields must retain float64")
    if not np.isfinite(coordinates).all() or not np.allclose(times, np.arange(201) * .01, rtol=0, atol=1e-12):
        raise ValueError("Invalid field coordinates or sampling times")
    extent = FULL_EXTENT if padded else CORE_EXTENT
    if not np.allclose(coordinates.min(axis=0), [extent[0], extent[2]], atol=1e-9, rtol=0) or not np.allclose(coordinates.max(axis=0), [extent[1], extent[3]], atol=1e-9, rtol=0):
        raise ValueError("Unexpected physical domain bounds")
    if np.count_nonzero(velocity[0]) != 0:
        raise ValueError("Initial field is not zero")
    return Fields(directory, metadata, coordinates, times, velocity, evidence)


def pixel_map(coordinates: FloatArray, extent: tuple[float, float, float, float], spacing: float) -> Pixels:
    xmin, xmax, ymin, ymax = extent
    width, height = round((xmax - xmin) / spacing), round((ymax - ymin) / spacing)
    x = xmin + (np.arange(width) + .5) * (xmax - xmin) / width
    y = ymin + (np.arange(height) + .5) * (ymax - ymin) / height
    xx, yy = np.meshgrid(x, y)
    distance, indices = cKDTree(coordinates).query(np.column_stack((xx.ravel(), yy.ravel())), workers=1)
    return Pixels(np.asarray(indices, dtype=np.int64), (height, width), extent, float(np.max(distance)))


def match_core_nodes(padded: Fields, core: Fields) -> tuple[IntArray, float]:
    distances, ids = cKDTree(padded.coordinates).query(core.coordinates, workers=1)
    maximum = float(np.max(distances))
    ids = np.asarray(ids, dtype=np.int64)
    if maximum > 1e-9 or len(np.unique(ids)) != len(core.coordinates):
        raise ValueError("Core field nodes do not correspond one-to-one to the original padded mesh")
    return ids, maximum


def scan(padded: Fields, core: Fields, matched: IntArray) -> dict[str, object]:
    speed_max = component_max = difference_max = 0.
    sums_delta = sums_reference = 0.
    per_frame: list[dict[str, float]] = []
    for index in range(201):
        padded_v = np.asarray(padded.velocity[index])
        core_v = np.asarray(core.velocity[index])
        if not np.isfinite(padded_v).all() or not np.isfinite(core_v).all():
            raise ValueError(f"Non-finite velocity field at frame {index}")
        delta = core_v - padded_v[matched]
        speed_max = max(speed_max, float(np.max(np.hypot(padded_v[:, 0], padded_v[:, 1]))),
                        float(np.max(np.hypot(core_v[:, 0], core_v[:, 1]))))
        component_max = max(component_max, float(np.max(np.abs(padded_v))), float(np.max(np.abs(core_v))))
        difference_max = max(difference_max, float(np.max(np.hypot(delta[:, 0], delta[:, 1]))))
        delta_sq, ref_sq = float(np.sum(delta * delta)), float(np.sum(padded_v[matched] ** 2))
        sums_delta += delta_sq
        sums_reference += ref_sq
        per_frame.append({"time_s": float(core.times[index]), "rms_vector_difference_m_s": float(np.sqrt(delta_sq / len(core.coordinates))),
                          "rms_reference_vector_m_s": float(np.sqrt(ref_sq / len(core.coordinates)))})
    if speed_max <= 0 or component_max <= 0:
        raise ValueError("Cannot animate an entirely zero experiment")
    return {"speed_max_m_s": speed_max, "component_abs_max_m_s": component_max,
            "vector_difference_max_m_s": difference_max,
            "core_all_nodes_all_frames_unweighted_relative_l2": float(np.sqrt(sums_delta / sums_reference)),
            "field_metric_note": "Unweighted samples over common SEM nodes and 201 frames, not a spatial quadrature/energy norm or accuracy estimate",
            "per_frame": per_frame}


def decorate(ax: plt.Axes, title: str, extent: tuple[float, float, float, float], core_box: bool) -> None:
    ax.set_facecolor("#e4e7eb")
    ax.set(xlabel="x, м", ylabel="y, м", title=title, xlim=extent[:2], ylim=extent[2:])
    ax.set_aspect("equal")
    ax.tick_params(labelsize=9)
    ax.plot(350., 0., marker="v", color="#16b9e5", markersize=6, clip_on=False, zorder=5)
    if core_box:
        ax.add_patch(Rectangle((0, -700), 700, 700, fill=False, edgecolor="#16b9e5", linewidth=.9, linestyle="--"))


def make_movies(padded_pixels: Pixels, core_pixels: Pixels, speed_norm: Normalize,
                component_norm: Normalize) -> list[Movie]:
    sequential = matplotlib.colormaps["inferno"].copy()
    sequential.set_bad("black")
    sequential.set_under("black")
    signed = matplotlib.colormaps["RdBu_r"]
    movies: list[Movie] = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), layout="constrained")
    artists = []
    for ax, title, pixels in zip(axes, ("Слой 175 м + внешняя ABC", "Без слоя: ABC на границе 700×700 м"), (padded_pixels, core_pixels), strict=True):
        artist = ax.imshow(np.zeros(pixels.shape), extent=pixels.extent, origin="lower", norm=speed_norm,
                           cmap=sequential, interpolation="nearest")
        decorate(ax, title, FULL_EXTENT, True)
        artists.append(artist)
    fig.colorbar(artists[0], ax=list(axes), label="|v|, м/с · общая постоянная log-шкала", shrink=.85)
    title = "Модуль скорости |v| · EFVB 00000 · p4 · источник x=350 м"
    label = fig.suptitle(title + " · t=0,00 с", fontsize=12)
    fig.supxlabel("Пунктир — основная область; серое — вне расчётного домена; нули — чёрные\n201 кадр через 0,01 с · ближайший узел · без усиления во времени", fontsize=9)
    movies.append(Movie("speed_full", fig, artists, label, title))
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 10.4), layout="constrained")
    artists = []
    for row, component in enumerate(("Vx", "Vy")):
        for column, (title, pixels) in enumerate(zip(("Слой 175 м", "Без слоя"), (padded_pixels, core_pixels), strict=True)):
            artist = axes[row, column].imshow(np.zeros(pixels.shape), extent=pixels.extent, origin="lower",
                norm=component_norm, cmap=signed, interpolation="nearest")
            decorate(axes[row, column], f"{title} · {component}", FULL_EXTENT, True)
            artists.append(artist)
    fig.colorbar(artists[0], ax=list(axes.ravel()), label="Vx / Vy, м/с · общая постоянная symlog-шкала", shrink=.85)
    title = "Компоненты скорости со знаком · EFVB 00000 · p4"
    label = fig.suptitle(title + " · t=0,00 с", fontsize=12)
    fig.supxlabel("Красное — положительная компонента; синее — отрицательная; нуль — белый\nСерое — вне домена · x вправо, y вверх · общие неизменные пределы для Vx/Vy", fontsize=9)
    movies.append(Movie("components_full", fig, artists, label, title))
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.8), layout="constrained")
    artists = []
    for ax, title in zip(axes, ("Со слоем · |v|", "Без слоя · |v|", "Разность · |vбез − vслой|"), strict=True):
        artist = ax.imshow(np.zeros(core_pixels.shape), extent=CORE_EXTENT, origin="lower", norm=speed_norm,
                           cmap=sequential, interpolation="nearest")
        decorate(ax, title, CORE_EXTENT, False)
        artists.append(artist)
    fig.colorbar(artists[0], ax=list(axes), label="м/с · одна постоянная log-шкала", shrink=.85)
    title = "Основная область: скорость и модуль разности векторов · EFVB 00000"
    label = fig.suptitle(title + " · t=0,00 с", fontsize=12)
    fig.supxlabel("Разность вычислена в совпадающих SEM-узлах до отображения: √[(ΔVx)²+(ΔVy)²]\nСравнение разных граничных постановок; не оценка ошибки относительно точного решения", fontsize=9)
    movies.append(Movie("speed_difference", fig, artists, label, title))
    return movies


def mapped(values: FloatArray, pixels: Pixels) -> FloatArray:
    return values[pixels.indices].reshape(pixels.shape)


def update_movies(movies: list[Movie], padded: Fields, core: Fields, index: int,
                  matched: IntArray, padded_pixels: Pixels, core_pixels: Pixels) -> None:
    p, c = np.asarray(padded.velocity[index]), np.asarray(core.velocity[index])
    p_speed, c_speed = np.hypot(p[:, 0], p[:, 1]), np.hypot(c[:, 0], c[:, 1])
    delta = c - p[matched]
    delta_speed = np.hypot(delta[:, 0], delta[:, 1])
    arrays = ([mapped(p_speed, padded_pixels), mapped(c_speed, core_pixels)],
              [mapped(p[:, 0], padded_pixels), mapped(c[:, 0], core_pixels), mapped(p[:, 1], padded_pixels), mapped(c[:, 1], core_pixels)],
              [mapped(p_speed[matched], core_pixels), mapped(c_speed, core_pixels), mapped(delta_speed, core_pixels)])
    for movie, frames in zip(movies, arrays, strict=True):
        for artist, values in zip(movie.artists, frames, strict=True):
            artist.set_data(values)
        movie.time_label.set_text(f"{movie.title} · t={core.times[index]:.2f} с")


def montage(padded: Fields, core: Fields, matched: IntArray, pixels: Pixels, norm: Normalize, output: Path) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), layout="constrained")
    cmap = matplotlib.colormaps["inferno"].copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    for block, index in enumerate(PREVIEW_INDICES):
        p, c = np.asarray(padded.velocity[index])[matched], np.asarray(core.velocity[index])
        delta = c - p
        arrays = [np.hypot(value[:, 0], value[:, 1]) for value in (p, c, delta)]
        for column, (name, value) in enumerate(zip(("Со слоем", "Без слоя", "|Δv|"), arrays, strict=True)):
            ax = axes[block // 2, 3 * (block % 2) + column]
            artist = ax.imshow(mapped(value, pixels), extent=CORE_EXTENT, origin="lower", norm=norm, cmap=cmap, interpolation="nearest")
            decorate(ax, f"{name} · {core.times[index]:.2f} с", CORE_EXTENT, False)
    fig.colorbar(artist, ax=list(axes.ravel()), label="м/с · общая постоянная log-шкала", shrink=.8)
    fig.suptitle("EFVB 00000 · эволюция поля и разности граничных постановок", fontsize=14)
    fig.supxlabel("Разность — модуль разности векторов в одинаковых узлах; источник x=350 м, p4/K96", fontsize=10)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def video_evidence(path: Path) -> dict[str, object]:
    process = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
        "stream=width,height,nb_read_frames,r_frame_rate,duration,pix_fmt", "-of", "json", str(path)],
        check=True, capture_output=True, text=True)
    stream = mapping(sequence(mapping(json.loads(process.stdout))["streams"])[0])
    if stream.get("nb_read_frames") != "201" or stream.get("r_frame_rate") != "25/1" or stream.get("pix_fmt") != "yuv420p":
        raise ValueError(f"Invalid encoded video: {path}")
    return {"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size, "ffprobe": stream}


def animate(padded_path: Path, core_path: Path, output: Path, report_path: Path, pixel_width: int = 630) -> dict[str, object]:
    output, report_path = output.resolve(), report_path.resolve()
    output.relative_to(PROJECT_ROOT / "img/dev_1_4_server_preparation")
    report_path.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if report_path.exists() or pixel_width < 210 or pixel_width % 6 != 0:
        raise ValueError("Need a new report path and pixel width >=210 divisible by 6")
    filenames = [f"{name}.mp4" for name in ("speed_full", "components_full", "speed_difference")]
    previews = [f"{name}_t{index:03d}.png" for name in ("speed_full", "components_full", "speed_difference") for index in PREVIEW_INDICES]
    all_outputs = filenames + previews + ["speed_difference_montage.png"]
    if any((output / name).exists() for name in all_outputs):
        raise FileExistsError("Animation outputs already exist; select a new output directory")
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    padded, core = load_fields(padded_path, True), load_fields(core_path, False)
    if padded.metadata["source_input_sha256"] != core.metadata["source_input_sha256"] or mapping(padded.metadata["native"])["sha256"] != mapping(core.metadata["native"])["sha256"]:
        raise ValueError("Source force/native differ between the field runs")
    if not np.array_equal(padded.times, core.times):
        raise ValueError("Field times differ")
    matched, maximum_match_distance = match_core_nodes(padded, core)
    scales = scan(padded, core, matched)
    print(json.dumps({"stage": "scales_checked", "speed_max_m_s": scales["speed_max_m_s"], "difference_max_m_s": scales["vector_difference_max_m_s"]}), flush=True)
    spacing = 1050 / pixel_width
    padded_pixels = pixel_map(padded.coordinates, FULL_EXTENT, spacing)
    core_pixels = pixel_map(core.coordinates, CORE_EXTENT, spacing)
    speed_max = max(number(scales["speed_max_m_s"]), number(scales["vector_difference_max_m_s"]))
    component_max = number(scales["component_abs_max_m_s"])
    speed_norm = LogNorm(vmin=speed_max * 1e-5, vmax=speed_max)
    component_norm = SymLogNorm(linthresh=component_max * 1e-3, vmin=-component_max, vmax=component_max, base=10)
    output.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    movies = make_movies(padded_pixels, core_pixels, speed_norm, component_norm)
    writers_ = [FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-threads", "2"],
        metadata={"title": movie.title}) for movie in movies]
    try:
        with ExitStack() as stack:
            for movie, writer in zip(movies, writers_, strict=True):
                stack.enter_context(writer.saving(movie.figure, str(output / f"{movie.name}.mp4"), dpi=100))
            for index in range(201):
                update_movies(movies, padded, core, index, matched, padded_pixels, core_pixels)
                for movie, writer in zip(movies, writers_, strict=True):
                    writer.grab_frame()
                    if index in PREVIEW_INDICES:
                        movie.figure.savefig(output / f"{movie.name}_t{index:03d}.png", dpi=140)
                if index % 20 == 0:
                    print(json.dumps({"stage": "rendered", "frame": index + 1, "frames": 201}), flush=True)
    finally:
        for movie in movies:
            plt.close(movie.figure)
    montage(padded, core, matched, core_pixels, speed_norm, output / "speed_difference_montage.png")
    record: dict[str, object] = {"status": "completed", "utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": 201, "physical_end_s": 2., "physical_cadence_s": .01, "playback_fps": 25,
        "playback_duration_s": 8.04, "playback_slowdown": 4., "components": ["Vx", "Vy"],
        "color_scales": {"speed_and_vector_difference_log_limits_m_s": [speed_max * 1e-5, speed_max],
            "components_symlog_limits_m_s": [-component_max, component_max], "components_linear_threshold_m_s": component_max * 1e-3,
            "policy": "One fixed physical scale across both variants, every frame and the vector-difference view; no temporal or per-panel rescaling. Log zeros/below floor black; outside core grey. Signed zero white."},
        "node_matching": {"method": "unique nearest padded SEM node for every core SEM node", "count": len(matched),
            "maximum_coordinate_distance_m": maximum_match_distance, "tolerance_m": 1e-9,
            "difference": "hypot(Vx_core - Vx_padded, Vy_core - Vy_padded), computed at matched SEM nodes before pixel selection"},
        "rendering": {"method": "nearest SEM node at regular pixel centres; no temporal interpolation",
            "nominal_pixel_spacing_m": spacing, "padded_pixel_shape": padded_pixels.shape, "core_pixel_shape": core_pixels.shape,
            "padded_max_pixel_to_node_distance_m": padded_pixels.maximum_distance_m,
            "core_max_pixel_to_node_distance_m": core_pixels.maximum_distance_m,
            "outside_core": "No data: grey axes background, not zero velocity"},
        "field_diagnostics": scales, "interpretation": "Different boundary-condition experiments, not an exact-solution error or independent solver validation",
        "inputs": [{"directory": str(item.directory), "metadata_sha256": sha256(item.directory / "metadata.json"), "fields": item.files} for item in (padded, core)],
        "videos": [video_evidence(output / filename) for filename in filenames],
        "previews": [{"path": str(output / name), "sha256": sha256(output / name)} for name in previews + ["speed_difference_montage.png"]],
        "script_sha256": sha256(Path(__file__))}
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = PROJECT_ROOT / "data/dev_1_4_server_preparation"
    parser.add_argument("--padded", type=Path, default=root / "velocity_fields_3060_v1/padded")
    parser.add_argument("--core", type=Path, default=root / "velocity_fields_3060_v1/core")
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / "img/dev_1_4_server_preparation/velocity_comparison_v1")
    parser.add_argument("--report", type=Path, default=root / "velocity_visualization_v1/animations.json")
    parser.add_argument("--pixel-width", type=int, default=630)
    args = parser.parse_args()
    result = animate(args.padded, args.core, args.out, args.report, args.pixel_width)
    print(json.dumps({"status": result["status"], "report": str(args.report)}), flush=True)


if __name__ == "__main__":
    main()
