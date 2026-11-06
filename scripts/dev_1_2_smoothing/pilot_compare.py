"""Compare completed raster and curved seismic-pilot runs; never runs a solver."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path, PureWindowsPath
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
import sys
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
COMPONENTS = ("Ux", "Uy")
ORDERS = (2, 3, 4)
OPTIONAL_ORDER = 5
KINDS = ("raster", "curved")
FAR_RADIUS_M = 20.0
TIME_WINDOWS_S = ((0.0, 0.2), (0.2, 0.5), (0.5, 1.0), (1.0, 2.0))
EXPECTED_SOLVER_ID = "lin_elast_2d_tenzor_newmark_kim_ceed"
EXPECTED_CEED_RESOURCE = "/cpu/self/gen"
EXPECTED_PROVENANCE_FILES = ("native_pyd", "ceed_dll", "pilot_run.py", "pilot_mesh.py", "selected_model.npz")
PHYSICAL_SOURCE_FIELDS = ("kind", "ricker_center_s", "ricker_hz", "sigma_m", "support_m", "total_force_N_per_unit_thickness", "x_m")


@dataclass(frozen=True, slots=True)
class RunData:
    label: str
    path: Path
    kind: str
    order: int
    substeps: int
    time_s: FloatArray
    surface: FloatArray
    x_m: FloatArray
    field_time_s: FloatArray
    field: FloatArray
    field_x_m: FloatArray
    field_y_m: FloatArray
    metadata: dict[str, object]


def _array(data: np.lib.npyio.NpzFile, key: str, dimensions: int, archive: Path) -> FloatArray:
    if key not in data:
        raise ValueError(f"Missing {key!r} in {archive}")
    value = np.asarray(data[key], dtype=np.float64)
    if value.ndim != dimensions or not np.all(np.isfinite(value)):
        raise ValueError(f"Invalid {key!r} in {archive}")
    return value


def load_run(path: Path, expected_kind: str, expected_order: int, expected_substeps: int) -> RunData:
    metadata_path = path / "metadata.json"
    surface_path = path / "surface.npz"
    fields_path = path / "fields.npz"
    for needed in (metadata_path, surface_path, fields_path):
        if not needed.is_file():
            raise FileNotFoundError(f"Expected completed pilot output {needed}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata must be an object: {metadata_path}")
    pilot = metadata.get("pilot")
    if not isinstance(pilot, dict):
        raise ValueError(f"metadata has no pilot record: {metadata_path}")
    kind = pilot.get("kind")
    order = pilot.get("order")
    substeps = pilot.get("substeps")
    if not isinstance(kind, str) or type(order) is not int or type(substeps) is not int:
        raise ValueError(f"metadata has invalid pilot identity: {metadata_path}")
    if (kind, order, substeps) != (expected_kind, expected_order, expected_substeps):
        raise ValueError(f"{path} metadata is {(kind, order, substeps)}, expected {(expected_kind, expected_order, expected_substeps)}")
    if pilot.get("t_end") != 2.0 or pilot.get("output_count") != 2001 or pilot.get("steps_count") != 2000 * expected_substeps:
        raise ValueError(f"metadata has incompatible pilot time discretization: {metadata_path}")
    runtime = metadata.get("runtime")
    solver_intent = metadata.get("solver_intent")
    if not isinstance(runtime, dict) or runtime.get("result_code") != 0:
        raise ValueError(f"metadata has no successful runtime result: {metadata_path}")
    if not isinstance(solver_intent, dict) or solver_intent.get("id") != EXPECTED_SOLVER_ID or solver_intent.get("ceed_resource") != EXPECTED_CEED_RESOURCE:
        raise ValueError(f"metadata has incompatible CEED CPU-gen solver intent: {metadata_path}")
    with np.load(surface_path, allow_pickle=False) as source:
        time_s = _array(source, "time_s", 1, surface_path)
        surface = _array(source, "displacement_ux_uy_m", 3, surface_path)
        x_m = _array(source, "x_m", 1, surface_path)
    with np.load(fields_path, allow_pickle=False) as source:
        field_time_s = _array(source, "time_s", 1, fields_path)
        field = _array(source, "displacement_ux_uy_m", 4, fields_path)
        field_x_m = _array(source, "x_m", 1, fields_path)
        field_y_m = _array(source, "y_m", 1, fields_path)
    if surface.shape != (time_s.size, 2, x_m.size):
        raise ValueError(f"surface shape is incompatible with coordinates in {surface_path}")
    if field.shape != (field_time_s.size, 2, field_y_m.size, field_x_m.size):
        raise ValueError(f"field shape is incompatible with coordinates in {fields_path}")
    if np.any(np.diff(time_s) <= 0.0) or np.any(np.diff(field_time_s) <= 0.0):
        raise ValueError(f"time coordinate must be strictly increasing in {path}")
    expected_time = np.linspace(0.0, 2.0, 2001, dtype=np.float64)
    expected_x = np.linspace(0.0, 700.0, 71, dtype=np.float64)
    expected_field_time = np.asarray((0.2, 0.5, 1.0, 2.0), dtype=np.float64)
    if time_s.shape != expected_time.shape or not np.allclose(time_s, expected_time, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"surface time coordinate must be exactly 0,...,2 s / 2001 samples: {surface_path}")
    if x_m.shape != expected_x.shape or not np.allclose(x_m, expected_x, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"surface receiver x coordinate must be 0,...,700 m / 71 samples: {surface_path}")
    if field_time_s.shape != expected_field_time.shape or not np.allclose(field_time_s, expected_field_time, rtol=0.0, atol=1.0e-12):
        raise ValueError(f"field snapshots must be at 0.2, 0.5, 1, 2 s: {fields_path}")
    return RunData(f"{kind}_p{order}_s{substeps}", path, kind, order, substeps, time_s, surface, x_m, field_time_s, field, field_x_m, field_y_m, metadata)


def _same_coordinates(left: RunData, right: RunData) -> None:
    for name, a, b in (("surface time", left.time_s, right.time_s), ("surface x", left.x_m, right.x_m), ("field time", left.field_time_s, right.field_time_s), ("field x", left.field_x_m, right.field_x_m), ("field y", left.field_y_m, right.field_y_m)):
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=1.0e-12):
            raise ValueError(f"{left.label} and {right.label} have incompatible {name}")


def _relative_l2(value: FloatArray, reference: FloatArray, mask: NDArray[np.bool_] | None = None) -> list[float]:
    values = value if mask is None else value[..., mask]
    refs = reference if mask is None else reference[..., mask]
    answer: list[float] = []
    for component in range(2):
        numerator = float(np.linalg.norm(values[:, component] - refs[:, component]))
        denominator = float(np.linalg.norm(refs[:, component]))
        answer.append(numerator / denominator if denominator > 0.0 else float("nan"))
    return answer


def _component_l2(value: FloatArray, mask: NDArray[np.bool_] | None = None) -> list[float]:
    values = value if mask is None else value[..., mask]
    return [float(np.linalg.norm(values[:, component])) for component in range(2)]


def _time_window_metrics(value: FloatArray, reference: FloatArray, time_s: FloatArray, far: NDArray[np.bool_]) -> list[dict[str, object]]:
    full_reference_l2 = _component_l2(reference)
    result: list[dict[str, object]] = []
    for index, (start_s, end_s) in enumerate(TIME_WINDOWS_S):
        mask = (time_s >= start_s) & ((time_s <= end_s) if index == len(TIME_WINDOWS_S) - 1 else (time_s < end_s))
        if not np.any(mask):
            raise ValueError(f"No surface samples in requested time window [{start_s}, {end_s}]")
        current = value[mask]
        current_reference = reference[mask]
        current_difference = current - current_reference
        reference_l2 = _component_l2(current_reference)
        result.append({
            "time_window_s": [start_s, end_s],
            "relative_l2": _relative_l2(current, current_reference),
            "difference_l2_m": _component_l2(current_difference),
            "reference_l2_m": reference_l2,
            "reference_l2_fraction_of_full": [reference_l2[component] / full_reference_l2[component] if full_reference_l2[component] > 0.0 else float("nan") for component in range(2)],
            "far_relative_l2": _relative_l2(current, current_reference, far),
            "far_difference_l2_m": _component_l2(current_difference, far),
            "far_reference_l2_m": _component_l2(current_reference, far),
        })
    return result


def _surface_array_metrics(value: FloatArray, reference: FloatArray, time_s: FloatArray, x_m: FloatArray) -> dict[str, object]:
    far = np.abs(x_m - 350.0) >= FAR_RADIUS_M
    difference = value - reference
    return {
        "all_surface": _relative_l2(value, reference),
        "far_surface": _relative_l2(value, reference, far),
        "max_abs_difference_m": np.max(np.abs(difference), axis=(0, 2)).tolist(),
        "reference_max_abs_m": np.max(np.abs(reference), axis=(0, 2)).tolist(),
        "difference_l2_m": _component_l2(difference),
        "value_l2_m": _component_l2(value),
        "reference_l2_m": _component_l2(reference),
        "far_difference_l2_m": _component_l2(difference, far),
        "far_value_l2_m": _component_l2(value, far),
        "far_reference_l2_m": _component_l2(reference, far),
        "time_windows": _time_window_metrics(value, reference, time_s, far),
    }


def _surface_metrics(value: RunData, reference: RunData) -> dict[str, object]:
    _same_coordinates(value, reference)
    return _surface_array_metrics(value.surface, reference.surface, value.time_s, value.x_m)


def _field_array_metrics(value: FloatArray, reference: FloatArray, time_s: FloatArray) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for index, time in enumerate(time_s):
        current = value[index:index + 1]
        current_reference = reference[index:index + 1]
        difference = current - current_reference
        result.append({
            "time_s": float(time),
            "relative_l2": _relative_l2(current, current_reference),
            "max_abs_difference_m": np.max(np.abs(difference), axis=(0, 2, 3)).tolist(),
            "difference_l2_m": _component_l2(difference),
            "value_l2_m": _component_l2(current),
            "reference_l2_m": _component_l2(current_reference),
        })
    return result


def _field_metrics(value: RunData, reference: RunData) -> list[dict[str, object]]:
    _same_coordinates(value, reference)
    return _field_array_metrics(value.field, reference.field, value.field_time_s)


def _metric_record(name: str, value: RunData, reference: RunData) -> dict[str, object]:
    return {"name": name, "value": value.label, "reference": reference.label, "surface": _surface_metrics(value, reference), "fields": _field_metrics(value, reference)}


def _geometry_difference_record(name: str, value_curved: RunData, value_raster: RunData, reference_curved: RunData, reference_raster: RunData) -> dict[str, object]:
    """Compare two geometry-difference fields G = curved - raster."""
    for left, right in ((value_curved, value_raster), (reference_curved, reference_raster), (value_curved, reference_curved)):
        _same_coordinates(left, right)
    value_surface = value_curved.surface - value_raster.surface
    reference_surface = reference_curved.surface - reference_raster.surface
    value_field = value_curved.field - value_raster.field
    reference_field = reference_curved.field - reference_raster.field
    return {
        "name": name,
        "value": f"{value_curved.label} - {value_raster.label}",
        "reference": f"{reference_curved.label} - {reference_raster.label}",
        "surface": _surface_array_metrics(value_surface, reference_surface, value_curved.time_s, value_curved.x_m),
        "fields": _field_array_metrics(value_field, reference_field, value_curved.field_time_s),
    }


def _numeric_pair(values: object) -> list[float]:
    if not isinstance(values, list) or len(values) != 2 or not all(isinstance(item, (int, float)) for item in values):
        raise AssertionError("component metric layout")
    return [float(item) for item in values]


def _surface_l2_series(record: dict[str, object], all_key: str, far_key: str) -> dict[str, object]:
    surface = record.get("surface")
    if not isinstance(surface, dict):
        raise AssertionError("surface metric layout")
    windows = surface.get("time_windows")
    if not isinstance(windows, list):
        raise AssertionError("time-window metric layout")
    series_windows: list[dict[str, object]] = []
    window_all_key = "relative_l2" if all_key == "all_surface" else all_key
    window_far_key = "far_relative_l2" if far_key == "far_surface" else far_key
    for window in windows:
        if not isinstance(window, dict):
            raise AssertionError("time-window metric record")
        time_window = window.get("time_window_s")
        if not isinstance(time_window, list) or len(time_window) != 2 or not all(isinstance(item, (int, float)) for item in time_window):
            raise AssertionError("time-window coordinate")
        series_windows.append({
            "time_window_s": [float(item) for item in time_window],
            "all_surface": _numeric_pair(window.get(window_all_key)),
            "far_surface": _numeric_pair(window.get(window_far_key)),
        })
    return {
        "all_surface": _numeric_pair(surface.get(all_key)),
        "far_surface": _numeric_pair(surface.get(far_key)),
        "time_windows": series_windows,
    }


def _sum_l2_series(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
    left_windows = left.get("time_windows")
    right_windows = right.get("time_windows")
    if not isinstance(left_windows, list) or not isinstance(right_windows, list) or len(left_windows) != len(right_windows):
        raise AssertionError("incompatible time-window series")

    def summed_pair(left_value: object, right_value: object) -> list[float]:
        first = _numeric_pair(left_value)
        second = _numeric_pair(right_value)
        return [first[component] + second[component] for component in range(2)]

    windows: list[dict[str, object]] = []
    for left_window, right_window in zip(left_windows, right_windows, strict=True):
        if not isinstance(left_window, dict) or not isinstance(right_window, dict) or left_window.get("time_window_s") != right_window.get("time_window_s"):
            raise AssertionError("incompatible time-window coordinates")
        windows.append({
            "time_window_s": left_window["time_window_s"],
            "all_surface": summed_pair(left_window.get("all_surface"), right_window.get("all_surface")),
            "far_surface": summed_pair(left_window.get("far_surface"), right_window.get("far_surface")),
        })
    return {
        "all_surface": summed_pair(left.get("all_surface"), right.get("all_surface")),
        "far_surface": summed_pair(left.get("far_surface"), right.get("far_surface")),
        "time_windows": windows,
    }


def _ratio_l2_series(effect: dict[str, object], error: dict[str, object]) -> dict[str, object]:
    effect_windows = effect.get("time_windows")
    error_windows = error.get("time_windows")
    if not isinstance(effect_windows, list) or not isinstance(error_windows, list) or len(effect_windows) != len(error_windows):
        raise AssertionError("incompatible time-window series")

    def ratio_pair(effect_value: object, error_value: object) -> list[float]:
        numerator = _numeric_pair(effect_value)
        denominator = _numeric_pair(error_value)
        return [numerator[component] / denominator[component] if denominator[component] > 0.0
                else (float("inf") if numerator[component] > 0.0 else 0.0) for component in range(2)]

    windows: list[dict[str, object]] = []
    for effect_window, error_window in zip(effect_windows, error_windows, strict=True):
        if not isinstance(effect_window, dict) or not isinstance(error_window, dict) or effect_window.get("time_window_s") != error_window.get("time_window_s"):
            raise AssertionError("incompatible time-window coordinates")
        windows.append({
            "time_window_s": effect_window["time_window_s"],
            "all_surface": ratio_pair(effect_window.get("all_surface"), error_window.get("all_surface")),
            "far_surface": ratio_pair(effect_window.get("far_surface"), error_window.get("far_surface")),
        })
    return {
        "all_surface": ratio_pair(effect.get("all_surface"), error.get("all_surface")),
        "far_surface": ratio_pair(effect.get("far_surface"), error.get("far_surface")),
        "time_windows": windows,
    }


def _less_than_one_percent(record: dict[str, object]) -> dict[str, object]:
    relative = _surface_l2_series(record, "all_surface", "far_surface")

    def flagged(values: object) -> list[bool]:
        return [item < 0.01 for item in _numeric_pair(values)]

    windows = relative["time_windows"]
    if not isinstance(windows, list):
        raise AssertionError("time-window metric layout")
    return {
        "all_surface": flagged(relative["all_surface"]),
        "far_surface": flagged(relative["far_surface"]),
        "time_windows": [{"time_window_s": window["time_window_s"], "all_surface": flagged(window["all_surface"]), "far_surface": flagged(window["far_surface"])} for window in windows if isinstance(window, dict)],
    }


def _protocol_attribution(geometry_p5: dict[str, object], geometry_self_p5: dict[str, object], raster_spatial: dict[str, object], curved_spatial: dict[str, object], raster_temporal: dict[str, object], curved_temporal: dict[str, object]) -> dict[str, object]:
    """Apply the unmodified five-times diagnostic without common-mode cancellation."""
    effect = _surface_l2_series(geometry_p5, "difference_l2_m", "far_difference_l2_m")
    geometry_self = _surface_l2_series(geometry_self_p5, "difference_l2_m", "far_difference_l2_m")
    spatial_proxy = _sum_l2_series(
        _surface_l2_series(raster_spatial, "difference_l2_m", "far_difference_l2_m"),
        _surface_l2_series(curved_spatial, "difference_l2_m", "far_difference_l2_m"),
    )
    temporal_proxy = _sum_l2_series(
        _surface_l2_series(raster_temporal, "difference_l2_m", "far_difference_l2_m"),
        _surface_l2_series(curved_temporal, "difference_l2_m", "far_difference_l2_m"),
    )
    spatial_ratio = _ratio_l2_series(effect, spatial_proxy)
    temporal_ratio = _ratio_l2_series(effect, temporal_proxy)
    return {
        "protocol_criterion": "PROTOCOL.md: a geometry difference no more than five times the numerical-error estimate cannot be confidently attributed to smoothing.",
        "primary_numerical_error_proxy": "conservative spatial sum ||raster p5 s36 - p4 s36|| + ||curved p5 s36 - p4 s36||; no common-mode cancellation is assumed",
        "spatial_numerical_error_l2_m": spatial_proxy,
        "geometry_effect_l2_m": effect,
        "effect_to_spatial_error_ratio": spatial_ratio,
        "componentwise_meets_five_times": {
            "all_surface": [item > 5.0 for item in _numeric_pair(spatial_ratio["all_surface"])],
            "far_surface": [item > 5.0 for item in _numeric_pair(spatial_ratio["far_surface"])],
        },
        "temporal_error_proxy": "separate p4 transfer: ||raster p4 s36 - p4 s18|| + ||curved p4 s36 - p4 s18||; it is not combined with the spatial proxy and does not establish p5 temporal convergence",
        "temporal_numerical_error_l2_m": temporal_proxy,
        "effect_to_temporal_error_ratio": temporal_ratio,
        "geometry_difference_self_convergence_l2_m": geometry_self,
        "geometry_difference_self_convergence_note": "Useful sensitivity diagnostic only; it is not the numerical-error basis of the original five-times gate because common-mode cancellation is possible.",
        "interpretation": "This is the protocol's numerical-attribution diagnostic, not proof of a physical causal mechanism.",
    }


def calculate_metrics(runs: dict[str, RunData]) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for kind in KINDS:
        records.append(_metric_record(f"{kind}: p2 to p3", runs[f"{kind}_p2"], runs[f"{kind}_p3"]))
        records.append(_metric_record(f"{kind}: p3 to p4", runs[f"{kind}_p3"], runs[f"{kind}_p4"]))
        records.append(_metric_record(f"{kind}: p4 s18 to s36 temporal", runs[f"{kind}_p4"], runs[f"{kind}_p4_s36"]))
    records.append(_metric_record("curved minus raster at p4", runs["curved_p4"], runs["raster_p4"]))
    records.append(_metric_record("curved minus raster at p4 s36", runs["curved_p4_s36"], runs["raster_p4_s36"]))
    records.append(_geometry_difference_record("geometry difference self-convergence: p3 s18 to p4 s18", runs["curved_p3"], runs["raster_p3"], runs["curved_p4"], runs["raster_p4"]))
    has_p5 = all(f"{kind}_p{OPTIONAL_ORDER}" in runs for kind in KINDS)
    result: dict[str, object] = {
        "units": "relative L2; components are [Ux, Uy] in metres before normalization",
        "far_field": "surface receivers satisfying |x - 350 m| >= 20 m",
        "records": records,
    }
    if has_p5:
        spatial_records: dict[str, dict[str, object]] = {}
        for kind in KINDS:
            spatial_record = _metric_record(f"{kind}: p4 s36 to p5 s36", runs[f"{kind}_p4_s36"], runs[f"{kind}_p{OPTIONAL_ORDER}"])
            records.append(spatial_record)
            spatial_records[kind] = spatial_record
        geometry_p5 = _metric_record("curved minus raster at p5 s36", runs["curved_p5"], runs["raster_p5"])
        geometry_self_p5 = _geometry_difference_record("geometry difference self-convergence: p4 s36 to p5 s36", runs["curved_p4_s36"], runs["raster_p4_s36"], runs["curved_p5"], runs["raster_p5"])
        records.extend((geometry_p5, geometry_self_p5))
        temporal_records = {kind: next(record for record in records if record["name"] == f"{kind}: p4 s18 to s36 temporal") for kind in KINDS}
        result["convergence_flags"] = {
            "threshold": "relative L2 < 1% componentwise; time refinement has been checked only at p4",
            "spatial_p4_s36_to_p5_s36": {kind: _less_than_one_percent(spatial_records[kind]) for kind in KINDS},
            "temporal_p4_s18_to_s36": {kind: _less_than_one_percent(temporal_records[kind]) for kind in KINDS},
        }
        result["protocol_attribution"] = _protocol_attribution(geometry_p5, geometry_self_p5, spatial_records["raster"], spatial_records["curved"], temporal_records["raster"], temporal_records["curved"])
    return result


def _import_plotting() -> None:
    cache = PROJECT_ROOT / ".matplotlib"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib

    matplotlib.use("Agg")


def _pair_stem(raster: RunData, curved: RunData) -> str:
    if raster.order == curved.order:
        return f"p{raster.order}"
    return f"{raster.label}_vs_{curved.label}"


def _cell_edges(coordinates: FloatArray) -> tuple[float, float]:
    if coordinates.size < 2:
        raise ValueError("Need at least two coordinate samples for image extent")
    first_spacing = float(coordinates[1] - coordinates[0])
    last_spacing = float(coordinates[-1] - coordinates[-2])
    return float(coordinates[0] - 0.5 * first_spacing), float(coordinates[-1] + 0.5 * last_spacing)


def _save_seismograms(output: Path, raster: RunData, curved: RunData, subtitle: str | None = None, time_window_s: tuple[float, float] | None = None) -> Path:
    output = image_directory(output)
    _same_coordinates(raster, curved)
    _import_plotting()
    import matplotlib.pyplot as plt
    time_mask = np.ones(raster.time_s.shape, dtype=np.bool_)
    name_prefix = "seismograms"
    if time_window_s is not None:
        start_s, end_s = time_window_s
        time_mask = (raster.time_s >= start_s) & (raster.time_s <= end_s)
        if not np.any(time_mask):
            raise ValueError(f"No samples for seismogram window [{start_s}, {end_s}]")
        name_prefix = "seismograms_late"
    raster_surface = raster.surface[time_mask]
    curved_surface = curved.surface[time_mask]
    time_s = raster.time_s[time_mask]
    difference = curved_surface - raster_surface
    amplitude = float(np.max(np.abs(np.stack((raster_surface, curved_surface)))))
    difference_amplitude = float(np.max(np.abs(difference)))
    figure, axes = plt.subplots(2, 3, figsize=(14, 6.5), constrained_layout=True, sharex=True, sharey="row")
    caption = f"p={raster.order}, dt={1000/raster.substeps:.2f} мкс; исходная и сглаженная — общая шкала, разность — отдельная"
    if subtitle is not None:
        caption = f"{caption}\n{subtitle}"
    figure.suptitle(caption)
    x_edges = _cell_edges(raster.x_m)
    time_edges = _cell_edges(time_s)
    extent = (x_edges[0], x_edges[1], time_edges[1], time_edges[0])
    for component, name in enumerate(COMPONENTS):
        entries = ((raster_surface[:, component], "исходная"), (curved_surface[:, component], "сглаженная"), (difference[:, component], "разность"))
        for column, (values, title) in enumerate(entries):
            limit = difference_amplitude if column == 2 else amplitude
            image = axes[component, column].imshow(values, aspect="auto", cmap="seismic", vmin=-limit, vmax=limit, extent=extent)
            axes[component, column].set_title(f"{name}: {title}")
            axes[component, column].set_xlabel("x, м")
            axes[component, column].set_xlim(float(raster.x_m[0]), float(raster.x_m[-1]))
            axes[component, column].set_ylim(float(time_s[-1]), float(time_s[0]))
            if column == 0:
                axes[component, column].set_ylabel("время, с")
            unit = "разность, м (отдельная шкала)" if column == 2 else "смещение, м (общая шкала)"
            figure.colorbar(image, ax=axes[component, column], label=unit)
    target = output / f"{name_prefix}_{_pair_stem(raster, curved)}.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def _save_snapshots(output: Path, raster: RunData, curved: RunData) -> Path:
    output = image_directory(output)
    _same_coordinates(raster, curved)
    _import_plotting()
    import matplotlib.pyplot as plt
    difference = curved.field - raster.field
    amplitude = float(np.max(np.abs(np.stack((raster.field, curved.field)))))
    difference_amplitude = float(np.max(np.abs(difference)))
    rows = 2 * raster.field_time_s.size
    figure, axes = plt.subplots(rows, 3, figsize=(12, 3.0 * rows), constrained_layout=True)
    figure.suptitle(f"p={raster.order}; общий масштаб во всех снимках для исходной и сглаженной среды; отдельная шкала разности")
    x_edges = _cell_edges(raster.field_x_m)
    y_edges = _cell_edges(raster.field_y_m)
    extent = (x_edges[0], x_edges[1], y_edges[1], y_edges[0])
    for time_index, time_s in enumerate(raster.field_time_s):
        for component, name in enumerate(COMPONENTS):
            row = 2 * time_index + component
            entries = ((raster.field[time_index, component], "исходная"), (curved.field[time_index, component], "сглаженная"), (difference[time_index, component], "разность"))
            for column, (values, title) in enumerate(entries):
                limit = difference_amplitude if column == 2 else amplitude
                image = axes[row, column].imshow(values, extent=extent, origin="upper", cmap="seismic", vmin=-limit, vmax=limit, aspect="equal")
                axes[row, column].set_title(f"t={time_s:g} с; {name}: {title}")
                axes[row, column].set_xlabel("x, м")
                if column == 0:
                    axes[row, column].set_ylabel("y, м")
                unit = "разность, м" if column == 2 else "смещение, м"
                figure.colorbar(image, ax=axes[row, column], label=unit)
    target = output / f"snapshots_{_pair_stem(raster, curved)}.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def _save_late_snapshots(output: Path, raster: RunData, curved: RunData) -> Path:
    output = image_directory(output)
    _same_coordinates(raster, curved)
    _import_plotting()
    import matplotlib.pyplot as plt
    late_indices = [index for index, time_s in enumerate(raster.field_time_s) if time_s >= 1.0]
    if not late_indices:
        raise ValueError("No snapshots in late interval [1, 2] s")
    figure, axes = plt.subplots(2 * len(late_indices), 3, figsize=(12, 3.0 * 2 * len(late_indices)), constrained_layout=True)
    figure.suptitle(f"p={raster.order}; t=1, 2 с: общая шкала исходной и сглаженной среды внутри каждого времени; отдельная шкала разности")
    x_edges = _cell_edges(raster.field_x_m)
    y_edges = _cell_edges(raster.field_y_m)
    extent = (x_edges[0], x_edges[1], y_edges[1], y_edges[0])
    for late_row, time_index in enumerate(late_indices):
        time_s = raster.field_time_s[time_index]
        difference = curved.field[time_index] - raster.field[time_index]
        amplitude = float(np.max(np.abs(np.stack((raster.field[time_index], curved.field[time_index])))))
        difference_amplitude = float(np.max(np.abs(difference)))
        for component, name in enumerate(COMPONENTS):
            row = 2 * late_row + component
            entries = ((raster.field[time_index, component], "исходная"), (curved.field[time_index, component], "сглаженная"), (difference[component], "разность"))
            for column, (values, title) in enumerate(entries):
                limit = difference_amplitude if column == 2 else amplitude
                image = axes[row, column].imshow(values, extent=extent, origin="upper", cmap="seismic", vmin=-limit, vmax=limit, aspect="equal")
                axes[row, column].set_title(f"t={time_s:g} с; {name}: {title}")
                axes[row, column].set_xlabel("x, м")
                if column == 0:
                    axes[row, column].set_ylabel("y, м")
                unit = "разность, м" if column == 2 else "смещение, м"
                figure.colorbar(image, ax=axes[row, column], label=unit)
    target = output / f"snapshots_late_{_pair_stem(raster, curved)}.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def _save_geometry_difference(output: Path, raster: RunData, curved: RunData) -> Path:
    output = image_directory(output)
    """Save the signed p-order geometry difference on both output representations."""
    _same_coordinates(raster, curved)
    _import_plotting()
    import matplotlib.pyplot as plt
    surface_difference = curved.surface - raster.surface
    field_difference = curved.field - raster.field
    amplitude = float(np.max(np.abs(np.concatenate((surface_difference.ravel(), field_difference.ravel())))))
    columns = 1 + raster.field_time_s.size
    figure, axes = plt.subplots(2, columns, figsize=(4.2 * columns, 6.0), constrained_layout=True)
    figure.suptitle(f"p={raster.order}; разность: сглаженная − исходная; отдельная шкала")
    surface_x_edges = _cell_edges(raster.x_m)
    surface_time_edges = _cell_edges(raster.time_s)
    field_x_edges = _cell_edges(raster.field_x_m)
    field_y_edges = _cell_edges(raster.field_y_m)
    surface_extent = (surface_x_edges[0], surface_x_edges[1], surface_time_edges[1], surface_time_edges[0])
    field_extent = (field_x_edges[0], field_x_edges[1], field_y_edges[1], field_y_edges[0])
    for component, name in enumerate(COMPONENTS):
        image = axes[component, 0].imshow(surface_difference[:, component], aspect="auto", cmap="seismic", vmin=-amplitude, vmax=amplitude, extent=surface_extent)
        axes[component, 0].set_title(f"{name}: поверхность")
        axes[component, 0].set_xlabel("x, м")
        axes[component, 0].set_ylabel("время, с")
        axes[component, 0].set_xlim(float(raster.x_m[0]), float(raster.x_m[-1]))
        axes[component, 0].set_ylim(float(raster.time_s[-1]), float(raster.time_s[0]))
        figure.colorbar(image, ax=axes[component, 0], label="разность, м (отдельная шкала)")
        for time_index, time_s in enumerate(raster.field_time_s, start=1):
            image = axes[component, time_index].imshow(field_difference[time_index - 1, component], extent=field_extent, origin="upper", cmap="seismic", vmin=-amplitude, vmax=amplitude, aspect="equal")
            axes[component, time_index].set_title(f"{name}: t={time_s:g} с")
            axes[component, time_index].set_xlabel("x, м")
            axes[component, time_index].set_ylabel("y, м")
            figure.colorbar(image, ax=axes[component, time_index], label="разность, м (отдельная шкала)")
    target = output / f"geometry_difference_{_pair_stem(raster, curved)}.png"
    figure.savefig(target, dpi=160)
    plt.close(figure)
    return target


def render_pair(output: Path, raster: RunData, curved: RunData, *, provisional: bool = False) -> dict[str, str]:
    """Render a same-order raw pair; provisional mode makes no convergence claim."""
    output.mkdir(parents=True, exist_ok=True)
    subtitle = "сходимость ещё не установлена" if provisional else None
    seismogram = _save_seismograms(output, raster, curved, subtitle)
    late_seismogram = _save_seismograms(output, raster, curved, subtitle, (1.0, 2.0))
    snapshot = _save_snapshots(output, raster, curved)
    late_snapshot = _save_late_snapshots(output, raster, curved)
    return {"seismogram": str(seismogram), "late_seismogram": str(late_seismogram), "snapshots": str(snapshot), "late_snapshots": str(late_snapshot)}


def make_provisional_preview(raster: RunData, curved: RunData, output: Path) -> dict[str, object]:
    if raster.order != curved.order:
        raise ValueError("A provisional raw pair requires equal element orders")
    images = render_pair(output, raster, curved, provisional=True)
    document: dict[str, object] = {
        "status": "provisional_raw_pair_not_convergence_result",
        "subtitle": f"p={raster.order}; сходимость ещё не установлена",
        "provenance": {"raster": str(raster.path), "curved": str(curved.path)},
        "images": images,
    }
    (output / "status.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def _save_convergence(output: Path, metrics: dict[str, object]) -> None:
    output = image_directory(output)
    _import_plotting()
    import matplotlib.pyplot as plt
    records = metrics["records"]
    if not isinstance(records, list):
        raise AssertionError("metrics record layout")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for kind in KINDS:
        selected = [record for record in records if isinstance(record, dict) and str(record["name"]).startswith(kind) and "to p" in str(record["name"])]
        labels = [str(record["name"]).removeprefix(f"{kind}: ") for record in selected]
        for component, component_name in enumerate(COMPONENTS):
            values = [float(record["surface"]["all_surface"][component]) for record in selected]
            axes[component].semilogy(labels, values, marker="o", label=kind)
            axes[component].set_title(f"{component_name} surface convergence")
            axes[component].set_ylabel("relative L2")
            axes[component].grid(True, which="both")
            axes[component].legend()
    figure.savefig(output / "convergence_p.png", dpi=160)
    plt.close(figure)


def _provenance_hashes(run: RunData) -> dict[str, str]:
    provenance = run.metadata.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError(f"{run.label}: metadata has no provenance object")
    files = provenance.get("files")
    if not isinstance(files, list):
        raise ValueError(f"{run.label}: metadata has no provenance file list")
    hashes: dict[str, str] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError(f"{run.label}: malformed provenance file entry")
        path = entry.get("path")
        digest = entry.get("sha256")
        if not isinstance(path, str) or not isinstance(digest, str):
            raise ValueError(f"{run.label}: malformed provenance path or hash")
        # Metadata from Windows must also remain readable on a POSIX host.
        filename = PureWindowsPath(path).name.lower()
        key: str | None = None
        if filename.startswith("_native") and filename.endswith(".pyd"):
            key = "native_pyd"
        elif filename == "ceed.dll":
            key = "ceed_dll"
        elif filename in EXPECTED_PROVENANCE_FILES:
            key = filename
        if key is not None:
            if key in hashes:
                raise ValueError(f"{run.label}: duplicate provenance file {key}")
            hashes[key] = digest
    if tuple(sorted(hashes)) != tuple(sorted(EXPECTED_PROVENANCE_FILES)):
        raise ValueError(f"{run.label}: provenance does not contain the required native, CEED, pilot-script, and selected-model hashes")
    return hashes


def _validate_series_consistency(runs: dict[str, RunData]) -> dict[str, object]:
    if not runs:
        raise ValueError("no pilot runs provided")
    first = next(iter(runs.values()))
    reference_hashes = _provenance_hashes(first)
    source = first.metadata.get("source")
    material = first.metadata.get("material")
    solver_intent = first.metadata.get("solver_intent")
    if not isinstance(source, dict) or not isinstance(material, dict) or not isinstance(solver_intent, dict):
        raise ValueError(f"{first.label}: metadata lacks source, material, or solver intent")
    reference_source = {key: source.get(key) for key in PHYSICAL_SOURCE_FIELDS}
    if any(value is None for value in reference_source.values()):
        raise ValueError(f"{first.label}: source lacks a physical comparison field")
    for run in runs.values():
        run_source = run.metadata.get("source")
        if not isinstance(run_source, dict) or {key: run_source.get(key) for key in PHYSICAL_SOURCE_FIELDS} != reference_source:
            raise ValueError(f"{run.label}: physical source differs from the pilot comparison series")
        if run.metadata.get("material") != material or run.metadata.get("solver_intent") != solver_intent:
            raise ValueError(f"{run.label}: material or solver intent differs from the pilot comparison series")
        if _provenance_hashes(run) != reference_hashes:
            raise ValueError(f"{run.label}: provenance hashes differ from the pilot comparison series")
    return {"physical_source": reference_source, "source_per_run_fields_not_compared": ["node_count", "consistent_integral_before_normalization_m"], "material": material, "solver_intent": solver_intent, "required_file_sha256": reference_hashes}


def make_report(runs: dict[str, RunData], output: Path) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    series_consistency = _validate_series_consistency(runs)
    metrics = calculate_metrics(runs)
    provenance = {label: {"path": str(run.path), "kind": run.kind, "order": run.order, "substeps": run.substeps} for label, run in runs.items()}
    has_p5 = all(f"{kind}_p{OPTIONAL_ORDER}" in runs for kind in KINDS)
    highest_completed_order = OPTIONAL_ORDER if has_p5 else 4
    final_status = "highest completed common order; convergence is assessed only by the separate spatial and temporal flags, never merely by file presence"
    document: dict[str, object] = {
        "status": "computed_from_completed_solver_outputs",
        "provenance": provenance,
        "series_consistency": series_consistency,
        "metrics": metrics,
        "final_order_selection": {"order": highest_completed_order, "status": final_status},
    }
    (output / "metrics.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    render_pair(output, runs["raster_p4_s36"], runs["curved_p4_s36"])
    if has_p5:
        render_pair(output, runs["raster_p5"], runs["curved_p5"])
        _save_geometry_difference(output, runs["raster_p5"], runs["curved_p5"])
    _save_convergence(output, metrics)
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for kind in KINDS:
        for order in ORDERS:
            parser.add_argument(f"--{kind}-p{order}", type=Path, required=True, metavar="DIR")
        parser.add_argument(f"--{kind}-p4-s36", type=Path, required=True, metavar="DIR")
        parser.add_argument(f"--{kind}-p{OPTIONAL_ORDER}", type=Path, metavar="DIR")
    parser.add_argument("--out", type=Path, required=True, metavar="EMPTY_DIR")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    p5_paths = [getattr(args, f"{kind}_p{OPTIONAL_ORDER}") for kind in KINDS]
    if any(path is None for path in p5_paths) and any(path is not None for path in p5_paths):
        raise ValueError("--raster-p5 and --curved-p5 must be supplied together")
    runs: dict[str, RunData] = {}
    for kind in KINDS:
        for order in ORDERS:
            name = f"{kind}_p{order}"
            runs[name] = load_run(getattr(args, f"{kind}_p{order}"), kind, order, 18)
        name = f"{kind}_p4_s36"
        runs[name] = load_run(getattr(args, f"{kind}_p4_s36"), kind, 4, 36)
        if getattr(args, f"{kind}_p{OPTIONAL_ORDER}") is not None:
            name = f"{kind}_p{OPTIONAL_ORDER}"
            runs[name] = load_run(getattr(args, f"{kind}_p{OPTIONAL_ORDER}"), kind, OPTIONAL_ORDER, 36)
    report = make_report(runs, args.out)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
