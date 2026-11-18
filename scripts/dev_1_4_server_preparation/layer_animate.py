"""Полные поля и разности однородного опыта с толщинами слоя 0/175/350 м."""

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
from matplotlib.animation import FFMpegWriter
from matplotlib.colors import LogNorm
import numpy as np

from scripts.dev_1_4_server_preparation.homogeneous_visualize import MATERIAL, load_surface
from scripts.dev_1_4_server_preparation.velocity_animate import (
    CORE_EXTENT, Fields, FloatArray, IntArray, Movie, Pixels, decorate,
    mapped, match_core_nodes, pixel_map, sha256, video_evidence,
)
from scripts.reporting import mapping, read_json

DATA = PROJECT_ROOT / "data/dev_1_4_server_preparation"
IMAGES = PROJECT_ROOT / "img/dev_1_4_server_preparation/damping_layer_comparison_v1"
REPORTS = DATA / "damping_layer_analysis_v1"
DIRECTORIES = (
    DATA / "homogeneous_3060_v1/core",
    DATA / "homogeneous_3060_v1/padded",
    DATA / "double_layer_3060_v1/padded350",
    DATA / "double_layer_3060_v1/held175_profile",
)
LABELS = ("Без слоя", "Слой 175 м", "Слой 350 м · плавный", "350 м · прежние 175 м + плато")
PAIRS = ((1, 0), (2, 1), (3, 1), (2, 3))
DIFFERENCES = (
    "Δv: 175 м − 0 м",
    "Δv: 350 м плавный − 175 м",
    "Δv: 350 м с плато − 175 м · перенос ABC",
    "Δv: 350 м плавный − 350 м с плато · профиль",
)
PREVIEWS = (20, 35, 50, 65, 80, 200)


def load(directory: Path) -> Fields:
    """Validate saved arrays without assuming a particular enlarged node count."""
    metadata = read_json(directory / "metadata.json")
    if metadata.get("status") != "completed" or metadata.get("physical_model") != MATERIAL:
        raise ValueError("Need a completed homogeneous run")
    if read_json(directory / "validation.json")["status"] != "passed":
        raise ValueError("Missing successful run validation")
    spec = mapping(metadata["run_spec"])
    for key, value in {"order": 4, "source_x_m": 350., "substeps": 96, "t_end_s": 2.}.items():
        if spec[key] != value:
            raise ValueError(f"Wrong parameter: {key}")
    evidence = mapping(metadata["field_files"])
    for filename in ("nodes.npy", "field_time_s.npy", "velocity_m_s.npy"):
        record = mapping(evidence[filename])
        path = directory / filename
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError(f"Field hash/size differs: {filename}")
    nodes = cast(FloatArray, np.load(directory / "nodes.npy", mmap_mode="r", allow_pickle=False))
    times = cast(FloatArray, np.load(directory / "field_time_s.npy", allow_pickle=False))
    velocity = cast(FloatArray, np.load(directory / "velocity_m_s.npy", mmap_mode="r", allow_pickle=False))
    if nodes.shape[1:] != (2,) or velocity.shape != (201, len(nodes), 2) or times.shape != (201,):
        raise ValueError("Wrong full-field shapes")
    if any(array.dtype != np.float64 for array in (nodes, times, velocity)):
        raise ValueError("Expected float64")
    if not np.isfinite(nodes).all() or not np.allclose(times, np.arange(201) * .01, rtol=0, atol=1e-12):
        raise ValueError("Invalid coordinates or physical times")
    surface = load_surface(directory, metadata)
    if not np.array_equal(velocity[:, surface.node_ids], surface.velocity[::10]):
        raise ValueError("All-node and surface observations disagree")
    return Fields(directory, metadata, nodes, times, velocity, evidence)


def extent(fields: Fields) -> tuple[float, float, float, float]:
    lo, hi = fields.coordinates.min(axis=0), fields.coordinates.max(axis=0)
    return float(lo[0]), float(hi[0]), float(lo[1]), float(hi[1])


def common_values(fields: list[Fields], ids: list[IntArray], frame: int) -> list[FloatArray]:
    return [np.asarray(item.velocity[frame, indices]) for item, indices in zip(fields, ids, strict=True)]


def difference_values(common: list[FloatArray]) -> list[FloatArray]:
    return [common[left] - common[right] for left, right in PAIRS]


def figures(pixels: list[Pixels], core_pixels: Pixels, speed_max: float, delta_max: float) -> list[Movie]:
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_under("black")
    result: list[Movie] = []
    for name, labels, maps, maximum in (
        ("speed_domains", LABELS, pixels, speed_max),
        ("difference_fields", DIFFERENCES, [core_pixels] * 4, delta_max),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
        artists = []
        for ax, label, grid in zip(axes.ravel(), labels, maps, strict=True):
            artist = ax.imshow(np.zeros(grid.shape), extent=grid.extent, origin="lower",
                norm=LogNorm(maximum * 1e-5, maximum), cmap=cmap, interpolation="nearest")
            view = (-350., 1050., -1050., 0.) if name == "speed_domains" else CORE_EXTENT
            decorate(ax, label, view, name == "speed_domains")
            artists.append(artist)
        title = "Однородная среда · модуль скорости" if name == "speed_domains" else "Однородная среда · модуль разности векторов скорости"
        time_label = fig.suptitle(title + " · t=0,00 с", fontsize=13)
        fig.colorbar(artists[0], ax=list(axes.ravel()), label="м/с · общая постоянная log-шкала", shrink=.8)
        fig.supxlabel("201 кадр · шаг 0,01 с · замедление ×4 · чёрное: ниже порога шкалы\nРазности в совпадающих узлах; сравнение постановок, а не ошибка относительно точного решения", fontsize=9)
        result.append(Movie(name, fig, artists, time_label, title))
    return result


def animate(output: Path, report: Path) -> dict[str, object]:
    output.resolve().relative_to(PROJECT_ROOT / "img/dev_1_4_server_preparation")
    report.resolve().relative_to(DATA)
    names = ("speed_domains", "difference_fields")
    outputs = [f"{name}.mp4" for name in names] + [f"{name}_t{i:03d}.png" for name in names for i in PREVIEWS]
    if report.exists() or any((output / filename).exists() for filename in outputs):
        raise FileExistsError("Select fresh animation outputs")
    fields = [load(directory) for directory in DIRECTORIES]
    for item in fields[1:]:
        for key in ("source_input_sha256", "physical_model", "dt_s", "steps_count"):
            if item.metadata[key] != fields[0].metadata[key]:
                raise ValueError(f"Runs disagree on {key}")
        if mapping(item.metadata["native"])["sha256"] != mapping(fields[0].metadata["native"])["sha256"]:
            raise ValueError("Different solver native binaries")
        if not np.array_equal(item.times, fields[0].times):
            raise ValueError("Different full-field times")
    matched = [match_core_nodes(item, fields[0]) for item in fields]
    ids = [item[0] for item in matched]
    maxima = np.zeros(4)
    speed_max = 0.
    for index in range(201):
        for item in fields:
            values = np.asarray(item.velocity[index])
            if not np.isfinite(values).all():
                raise ValueError("Non-finite full-field velocity")
            speed_max = max(speed_max, float(np.max(np.hypot(values[:, 0], values[:, 1]))))
        for j, delta in enumerate(difference_values(common_values(fields, ids, index))):
            maxima[j] = max(maxima[j], float(np.max(np.hypot(delta[:, 0], delta[:, 1]))))
    delta_max = float(np.max(maxima))
    if min(speed_max, delta_max) <= 0:
        raise ValueError("Empty speed or difference signal")
    pixels = [pixel_map(item.coordinates, extent(item), 2.5) for item in fields]
    core_pixels = pixels[0]
    output.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    movies = figures(pixels, core_pixels, speed_max, delta_max)
    writers = [FFMpegWriter(fps=25, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "19", "-threads", "2"]) for _ in movies]
    try:
        with ExitStack() as stack:
            for movie, writer in zip(movies, writers, strict=True):
                stack.enter_context(writer.saving(movie.figure, str(output / f"{movie.name}.mp4"), dpi=100))
            for index in range(201):
                for artist, item, grid in zip(movies[0].artists, fields, pixels, strict=True):
                    v = np.asarray(item.velocity[index])
                    artist.set_data(mapped(np.hypot(v[:, 0], v[:, 1]), grid))
                for artist, delta in zip(movies[1].artists, difference_values(common_values(fields, ids, index)), strict=True):
                    artist.set_data(mapped(np.hypot(delta[:, 0], delta[:, 1]), core_pixels))
                for movie, writer in zip(movies, writers, strict=True):
                    movie.time_label.set_text(f"{movie.title} · t={fields[0].times[index]:.2f} с")
                    writer.grab_frame()
                    if index in PREVIEWS:
                        movie.figure.savefig(output / f"{movie.name}_t{index:03d}.png", dpi=120)
                if index % 20 == 0:
                    print(json.dumps({"stage": "render", "frame": index + 1, "total": 201}), flush=True)
    finally:
        for movie in movies:
            plt.close(movie.figure)
    record: dict[str, object] = {"status": "completed", "utc": datetime.now(timezone.utc).isoformat(),
        "frame_count": 201, "physical_end_s": 2., "physical_cadence_s": .01,
        "color_scales": {"speed_log_m_s": [speed_max * 1e-5, speed_max], "difference_log_m_s": [delta_max * 1e-5, delta_max], "fixed_all_frames_and_panels": True},
        "difference_labels": DIFFERENCES, "difference_maxima_m_s": maxima.tolist(),
        "core_node_count": len(ids[0]), "node_matching_max_distance_m": [item[1] for item in matched],
        "pixel_spacing_m": 2.5, "pixel_max_distance_m": [item.maximum_distance_m for item in pixels],
        "inputs": [{"directory": str(item.directory.relative_to(PROJECT_ROOT)), "metadata_sha256": sha256(item.directory / "metadata.json"), "files": item.files} for item in fields],
        "videos": [video_evidence(output / f"{name}.mp4") for name in names],
        "previews": [{"path": str((output / filename).relative_to(PROJECT_ROOT)), "sha256": sha256(output / filename)} for filename in outputs if filename.endswith(".png")],
        "code_sha256": sha256(Path(__file__))}
    with report.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=IMAGES)
    parser.add_argument("--report", type=Path, default=REPORTS / "animations.json")
    args = parser.parse_args()
    animate(args.out, args.report)


if __name__ == "__main__":
    main()
