"""Сейсмограммы и полные поля однородной среды: материал верхнего слоя EFVB0."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import cast

from scripts.project_paths import PROJECT_ROOT
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, writers
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
import numpy as np

from scripts.dev_1_4_server_preparation.compare_smoke import Surface, file_evidence, metrics
from scripts.dev_1_4_server_preparation.velocity_seismograms import COLORS, COMPONENTS, save_figure
from scripts.dev_1_4_server_preparation.velocity_animate import (
    CORE_EXTENT, FULL_EXTENT, Fields, FloatArray, IntArray, Pixels, decorate,
    load_fields, make_movies, mapped, match_core_nodes, pixel_map, scan, sha256,
    update_movies, video_evidence,
)
from scripts.reporting import mapping, number, read_json

DATA_ROOT = PROJECT_ROOT / "data/dev_1_4_server_preparation"
IMAGE_ROOT = PROJECT_ROOT / "img/dev_1_4_server_preparation"
RUN_ROOT = DATA_ROOT / "homogeneous_3060_v1"
MATERIAL = {"vp_m_s": 4087.0, "vs_m_s": 2431.62036, "density_kg_m3": 1000.0, "homogeneous": True}
MATERIAL_LABEL = "Однородная среда · Vp=4087, Vs=2431,62 м/с"
PREVIEW_INDICES = (10, 25, 60, 200)


def material_metadata(directory: Path, padded: bool) -> dict[str, object]:
    metadata = read_json(directory / "metadata.json")
    if metadata.get("status") != "completed" or (directory / "metadata.failed.json").exists():
        raise ValueError("Need a completed homogeneous field run")
    if metadata.get("material_model") != "homogeneous_reference_toplayer" or metadata.get("is_original_dataset_model") is not False or metadata.get("physical_model") != MATERIAL:
        raise ValueError("The actual material override is not the requested homogeneous medium")
    variant = "padded" if padded else "core"
    expected_physics = [11000, 32000, 30001, 20200] if padded else [11000, 32000, 20200]
    if metadata.get("variant") != variant or metadata.get("physics_ids") != expected_physics or metadata.get("mass_damping_disabled") != (not padded):
        raise ValueError("Homogeneous domain variant/physics differs")
    if metadata.get("boundary_mode") != ("outer_abc" if padded else "core_abc"):
        raise ValueError("Unexpected ABC location")
    spec = mapping(metadata["run_spec"])
    for key, expected in {"t_end_s": 2., "substeps": 96, "order": 4, "source_x_m": 350.}.items():
        if spec.get(key) != expected:
            raise ValueError(f"Unexpected source/time parameter {key}")
    runtime = mapping(metadata["runtime"])
    if runtime["result_code"] != runtime["time_end_code"] or runtime["steps_count"] != 192000 or not np.isclose(number(runtime["current_time_s"]), 2., atol=2e-10, rtol=0):
        raise ValueError("Homogeneous run did not complete 2 seconds")
    mesh_record = mapping(metadata["mesh_evidence"])
    mesh = Path(str(mesh_record["path"]))
    if sha256(mesh) != mesh_record["sha256"]:
        raise ValueError("Uniform material mesh hash differs")
    with np.load(mesh, allow_pickle=False) as archive:
        if not np.all(archive["vp"] == MATERIAL["vp_m_s"]) or not np.all(archive["vs"] == MATERIAL["vs_m_s"]) or not np.all(archive["regions"] == 0):
            raise ValueError("The mesh does not actually contain one uniform material")
    return metadata


def load_surface(directory: Path, metadata: dict[str, object]) -> Surface:
    path = directory / "surface.npz"
    if sha256(path) != mapping(metadata["surface_file"])["sha256"]:
        raise ValueError("Surface SHA mismatch")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != {"time_s", "coordinates_m", "node_ids", "velocity_m_s"}:
            raise ValueError("Unexpected surface schema")
        times = cast(FloatArray, np.asarray(archive["time_s"]))
        coordinates = cast(FloatArray, np.asarray(archive["coordinates_m"]))
        node_ids = cast(IntArray, np.asarray(archive["node_ids"]))
        velocity = cast(FloatArray, np.asarray(archive["velocity_m_s"]))
    if times.shape != (2001,) or coordinates.shape != (70, 2) or node_ids.shape != (70,) or velocity.shape != (2001, 70, 2):
        raise ValueError("Unexpected 70-receiver shape")
    if any(array.dtype != np.float64 or not np.isfinite(array).all() for array in (times, coordinates, velocity)) or node_ids.dtype != np.int64:
        raise ValueError("Invalid surface dtype or nonfinite value")
    if not np.array_equal(times, np.linspace(0., 2., 2001)) or not np.allclose(coordinates[:, 0], np.arange(5., 700., 10.), atol=1e-9, rtol=0) or not np.allclose(coordinates[:, 1], 0., atol=1e-9, rtol=0):
        raise ValueError("Unexpected receiver/time grid")
    if np.any(velocity[0] != 0) or len(np.unique(node_ids)) != 70 or np.any(node_ids < 0):
        raise ValueError("Nonzero initial surface or invalid node IDs")
    return Surface(times, coordinates, node_ids, velocity)


def validate_pair(padded: dict[str, object], core: dict[str, object]) -> None:
    for key in ("source_input_sha256", "initial_state", "physical_model", "dt_s", "steps_count"):
        if padded[key] != core[key]:
            raise ValueError(f"Unexpected difference between homogeneous runs: {key}")
    if mapping(padded["native"])["sha256"] != mapping(core["native"])["sha256"]:
        raise ValueError("Different native binaries")


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
    figure.suptitle(MATERIAL_LABEL + " · p4 / K96\n"
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
    figure.suptitle(MATERIAL_LABEL + "\nТрассы у края, на четверти профиля и рядом с источником\n"
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
        axis.set(yscale="log", ylim=(1e-5, max(float(np.max(value[:, component])) * 1e9 for value in rms) * 1.3), ylabel=f"RMS {COMPONENTS[component]}, нм/с", xlim=(0., 2.))
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
    figure.suptitle(MATERIAL_LABEL + "\nКогда меняется запись после удаления слоя\n"
                   "Абсолютная разность без деления на почти нулевой сигнал; нули на логарифмической оси не показаны",
                   fontsize=14)
    peak_index = int(np.argmax(np.sqrt(np.mean(difference**2, axis=(1, 2)))))
    return save_figure(figure, output, "difference_over_time"), {
        "rms_definition": "sqrt(mean over 70 receivers of component squared), at each recorded time",
        "rms_plot_lower_limit_nm_s": 1e-5,
        "cumulative_definition": "sum(deltaVx**2+deltaVy**2) through time / sum over all 2001 times and 70 receivers",
        "cumulative_fraction_times_s": quantiles, "maximum_difference_rms_time_s": float(times[peak_index]),
        "maximum_difference_rms_m_s": float(np.sqrt(np.mean(difference[peak_index]**2)))}


def write_json(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def build_seismograms(padded_path: Path, core_path: Path, images: Path, report_path: Path) -> dict[str, object]:
    """Read only the two new homogeneous surfaces; do not compare with layered EFVB."""
    if report_path.exists():
        raise FileExistsError(report_path)
    padded_metadata, core_metadata = material_metadata(padded_path, True), material_metadata(core_path, False)
    validate_pair(padded_metadata, core_metadata)
    padded, core = load_surface(padded_path, padded_metadata), load_surface(core_path, core_metadata)
    if not np.array_equal(padded.times, core.times) or not np.array_equal(padded.coordinates, core.coordinates):
        raise ValueError("Homogeneous recording grids differ")
    images.mkdir(parents=True, exist_ok=True)
    times, x = padded.times, padded.coordinates[:, 0]
    artifacts, scale = seismograms(times, x, padded.velocity, core.velocity, images)
    trace_files, trace_positions = traces(times, x, padded.velocity, core.velocity, images)
    time_files, time_summary = difference_over_time(times, padded.velocity, core.velocity, images)
    difference = core.velocity - padded.velocity
    total = float(np.sum(difference**2))
    windows: list[dict[str, object]] = []
    for lower, upper in ((0., .5), (.5, 1.), (1., 1.5), (1.5, 2.)):
        selected = (times >= lower) & ((times < upper) if upper < 2. else (times <= upper))
        windows.append({"time_interval_s": [lower, upper], "right_endpoint_included": upper == 2.,
            "sample_count": int(np.count_nonzero(selected)), "metrics": metrics(core.velocity[selected], padded.velocity[selected]),
            "fraction_of_total_squared_difference": float(np.sum(difference[selected]**2)) / total})
    if sum(int(number(window["sample_count"])) for window in windows) != 2001 or not np.isclose(sum(number(window["fraction_of_total_squared_difference"]) for window in windows), 1., atol=1e-14):
        raise AssertionError("Time windows do not partition all surface data")
    report: dict[str, object] = {"schema": "chapter14.homogeneous_seismograms.v1", "status": "completed",
        "utc": datetime.now(timezone.utc).isoformat(), "material_model": "homogeneous_reference_toplayer",
        "physical_model": MATERIAL, "is_original_dataset_model": False,
        "source_x_m": 350., "shape": [2001, 70, 2], "time_extent_s": [0., 2.],
        "full_window_metrics": metrics(core.velocity, padded.velocity),
        "component_metrics": {name: metrics(core.velocity[:, :, axis], padded.velocity[:, :, axis]) for axis, name in enumerate(COMPONENTS)},
        "relative_l2_definition": "norm(core-padded)/norm(padded), over selected samples, receivers and components; window metrics use their own padded-window denominator",
        "scale": scale, "trace_receiver_x_m": trace_positions, "time_diagnostics": time_summary, "windows": windows,
        "source_force_and_native_equal": True, "material_values_verified_in_actual_mesh": True,
        "source_runs": {name: {"metadata": file_evidence(path / "metadata.json"), "surface": file_evidence(path / "surface.npz")}
                        for name, path in (("padded", padded_path), ("core", core_path))},
        "plots": artifacts + trace_files + time_files,
        "interpretation": "One uniform material, different boundary treatments. Differences are not exact-solution errors, and the stored density/velocities are not those of the original layered dataset model.",
        "code": file_evidence(Path(__file__)), "shared_plot_helpers": file_evidence(Path(__file__).with_name("velocity_seismograms.py"))}
    write_json(report_path, report)
    return report


def homogeneous_montage(padded: Fields, core: Fields, matched: IntArray, pixels: Pixels,
                        norm: Normalize, output: Path) -> None:
    """Early intact fronts and subsequent boundary interactions on a common scale."""
    fig, axes = plt.subplots(2, 6, figsize=(20, 8), layout="constrained")
    cmap = matplotlib.colormaps["inferno"].copy()
    cmap.set_bad("black")
    cmap.set_under("black")
    for block, index in enumerate(PREVIEW_INDICES):
        p, c = np.asarray(padded.velocity[index])[matched], np.asarray(core.velocity[index])
        arrays = [np.hypot(value[:, 0], value[:, 1]) for value in (p, c, c - p)]
        for column, (name, value) in enumerate(zip(("Со слоем", "Без слоя", "|Δv|"), arrays, strict=True)):
            ax = axes[block // 2, 3 * (block % 2) + column]
            artist = ax.imshow(mapped(value, pixels), extent=CORE_EXTENT, origin="lower", norm=norm,
                               cmap=cmap, interpolation="nearest")
            decorate(ax, f"{name} · {core.times[index]:.2f} с", CORE_EXTENT, False)
    fig.colorbar(artist, ax=list(axes.ravel()), label="м/с · общая постоянная log-шкала", shrink=.8)
    fig.suptitle(MATERIAL_LABEL + " · эволюция поля и разности", fontsize=14)
    fig.supxlabel("Разность — модуль разности векторов в одинаковых узлах · центральный источник x=350 м · p4/K96", fontsize=10)
    fig.savefig(output, dpi=140)
    plt.close(fig)


def build_animations(padded_path: Path, core_path: Path, output: Path,
                     report_path: Path, pixel_width: int = 630) -> dict[str, object]:
    if report_path.exists() or pixel_width < 210 or pixel_width % 6 != 0:
        raise ValueError("Need a new report and pixel width >=210 divisible by 6")
    names = ("speed_full", "components_full", "speed_difference")
    filenames = [f"{name}.mp4" for name in names]
    previews = [f"{name}_t{index:03d}.png" for name in names for index in PREVIEW_INDICES]
    all_outputs = filenames + previews + ["speed_difference_montage.png"]
    if any((output / name).exists() for name in all_outputs):
        raise FileExistsError("Animation output already exists")
    if not writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is required")
    padded_metadata, core_metadata = material_metadata(padded_path, True), material_metadata(core_path, False)
    validate_pair(padded_metadata, core_metadata)
    padded, core = load_fields(padded_path, True), load_fields(core_path, False)
    if not np.array_equal(padded.times, core.times):
        raise ValueError("Field time axes differ")
    matched, match_distance = match_core_nodes(padded, core)
    scales = scan(padded, core, matched)
    print(json.dumps({"stage": "scales_checked", "speed_max_m_s": scales["speed_max_m_s"],
                      "difference_max_m_s": scales["vector_difference_max_m_s"]}), flush=True)
    spacing = 1050 / pixel_width
    padded_pixels, core_pixels = pixel_map(padded.coordinates, FULL_EXTENT, spacing), pixel_map(core.coordinates, CORE_EXTENT, spacing)
    speed_max = max(number(scales["speed_max_m_s"]), number(scales["vector_difference_max_m_s"]))
    component_max = number(scales["component_abs_max_m_s"])
    speed_norm = LogNorm(vmin=speed_max * 1e-5, vmax=speed_max)
    component_norm = SymLogNorm(linthresh=component_max * 1e-3, vmin=-component_max, vmax=component_max, base=10)
    output.mkdir(parents=True, exist_ok=True)
    movies = make_movies(padded_pixels, core_pixels, speed_norm, component_norm)
    titles = ("Однородная среда · модуль скорости |v|", "Однородная среда · компоненты Vx/Vy со знаком",
              "Однородная среда · скорость и модуль разности векторов")
    for movie, title in zip(movies, titles, strict=True):
        movie.title = title + "\nVp=4087 м/с · Vs=2431,62 м/с · источник x=350 м · p4"
        movie.time_label.set_text(movie.title + " · t=0,00 с")
    writer_objects = [FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-threads", "2"],
        metadata={"title": movie.title}) for movie in movies]
    try:
        with ExitStack() as stack:
            for movie, writer in zip(movies, writer_objects, strict=True):
                stack.enter_context(writer.saving(movie.figure, str(output / f"{movie.name}.mp4"), dpi=100))
            for index in range(201):
                update_movies(movies, padded, core, index, matched, padded_pixels, core_pixels)
                for movie, writer in zip(movies, writer_objects, strict=True):
                    writer.grab_frame()
                    if index in PREVIEW_INDICES:
                        movie.figure.savefig(output / f"{movie.name}_t{index:03d}.png", dpi=140)
                if index % 20 == 0:
                    print(json.dumps({"stage": "rendered", "frame": index + 1, "frames": 201}), flush=True)
    finally:
        for movie in movies:
            plt.close(movie.figure)
    homogeneous_montage(padded, core, matched, core_pixels, speed_norm, output / "speed_difference_montage.png")
    record: dict[str, object] = {"schema": "chapter14.homogeneous_animations.v1", "status": "completed",
        "utc": datetime.now(timezone.utc).isoformat(), "material_model": "homogeneous_reference_toplayer",
        "physical_model": MATERIAL, "is_original_dataset_model": False,
        "frame_count": 201, "physical_end_s": 2., "physical_cadence_s": .01,
        "playback_fps": 25, "playback_duration_s": 8.04, "playback_slowdown": 4.,
        "preview_physical_times_s": [index * .01 for index in PREVIEW_INDICES],
        "color_scales": {"speed_and_vector_difference_log_limits_m_s": [speed_max * 1e-5, speed_max],
            "components_symlog_limits_m_s": [-component_max, component_max], "components_linear_threshold_m_s": component_max * 1e-3,
            "policy": "Fixed physical limits across both variants and all frames; log zeros/below floor black, outside domain grey; signed zero white; no temporal rescaling"},
        "node_matching": {"count": len(matched), "maximum_coordinate_distance_m": match_distance, "tolerance_m": 1e-9,
            "difference": "hypot(Vx_core-Vx_padded,Vy_core-Vy_padded), computed at one-to-one matched SEM nodes before pixel selection"},
        "rendering": {"method": "nearest SEM node at regular pixel centres; no temporal interpolation",
            "nominal_pixel_spacing_m": spacing, "padded_pixel_shape": padded_pixels.shape, "core_pixel_shape": core_pixels.shape,
            "padded_max_pixel_to_node_distance_m": padded_pixels.maximum_distance_m,
            "core_max_pixel_to_node_distance_m": core_pixels.maximum_distance_m},
        "field_diagnostics": scales, "interpretation": "Homogeneous elastic material with a free surface and two boundary treatments; no independent solver/ABC accuracy claim",
        "inputs": [{"directory": str(item.directory), "metadata_sha256": sha256(item.directory / "metadata.json"), "fields": item.files} for item in (padded, core)],
        "videos": [video_evidence(output / filename) for filename in filenames],
        "previews": [{"path": str(output / name), "sha256": sha256(output / name)} for name in previews + ["speed_difference_montage.png"]],
        "code": file_evidence(Path(__file__)), "shared_field_helpers": file_evidence(Path(__file__).with_name("velocity_animate.py"))}
    write_json(report_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--padded", type=Path, default=RUN_ROOT / "padded")
    parser.add_argument("--core", type=Path, default=RUN_ROOT / "core")
    parser.add_argument("--images", type=Path, default=IMAGE_ROOT / "homogeneous_comparison_v1")
    parser.add_argument("--reports", type=Path, default=DATA_ROOT / "homogeneous_visualization_v1")
    parser.add_argument("--mode", choices=("all", "seismograms", "animations"), default="all")
    parser.add_argument("--pixel-width", type=int, default=630)
    args = parser.parse_args()
    images, reports = args.images.resolve(), args.reports.resolve()
    images.relative_to(IMAGE_ROOT)
    reports.relative_to(DATA_ROOT)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    if args.mode in ("all", "seismograms"):
        report = build_seismograms(args.padded, args.core, images, reports / "seismograms.json")
        print(json.dumps({"stage": "seismograms_complete", "metrics": report["full_window_metrics"]}), flush=True)
    if args.mode in ("all", "animations"):
        report = build_animations(args.padded, args.core, images, reports / "animations.json", args.pixel_width)
        print(json.dumps({"stage": "animations_complete", "frames": report["frame_count"]}), flush=True)


if __name__ == "__main__":
    main()

