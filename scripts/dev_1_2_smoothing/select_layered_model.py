"""Select a topologically simple curved ECVB model for a SEM pilot."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
from typing import List, Tuple

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import label  # type: ignore[import-untyped]


ROOT = DATA_12
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class Candidate:
    model_id: int
    layer_count: int
    minimum_thickness_m: float
    minimum_thickness_cells: int
    interface_amplitude_m: float
    interface_slope_max: float
    curvature_score: float
    score: float


def column_runs(labels: IntArray, column: int) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    values = labels[:, column]
    changes = np.flatnonzero(np.diff(values) != 0) + 1
    starts = np.r_[0, changes]
    ends = np.r_[changes, 70]
    return tuple(int(values[s]) for s in starts), tuple(int(e-s) for s, e in zip(starts, ends))


def describe(vp: FloatArray, vs: FloatArray, model_id: int) -> Tuple[Candidate, IntArray, FloatArray]:
    materials, inverse = np.unique(np.stack((vp.ravel(), vs.ravel()), axis=1), axis=0, return_inverse=True)
    labels = inverse.reshape((70, 70)).astype(np.int64)
    sequence, thickness = column_runs(labels, 0)
    if not sequence or min(thickness) < 4:
        raise ValueError("Not a sufficiently thick layered candidate")
    for column in range(1, 70):
        other_sequence, other_thickness = column_runs(labels, column)
        if other_sequence != sequence or min(other_thickness) < 4:
            raise ValueError("Topology differs among columns")
    # Interface positions are at cell edges, depth positive downwards.
    boundaries = np.empty((len(sequence) - 1, 70), dtype=np.float64)
    for column in range(70):
        _, widths = column_runs(labels, column)
        boundaries[:, column] = np.cumsum(widths[:-1]) * 10.0
    amplitude = float(np.max(np.ptp(boundaries, axis=1))) if boundaries.size else 0.0
    slope = float(np.max(np.abs(np.diff(boundaries, axis=1) / 10.0))) if boundaries.size else 0.0
    curvature = float(np.mean(np.abs(np.diff(boundaries, n=2, axis=1)))) if boundaries.shape[1] > 2 else 0.0
    minimum = int(min(min(column_runs(labels, c)[1]) for c in range(70)))
    # Prefer 4--8 layers, clear curvature, but avoid near-vertical interfaces.
    score = 5.0 * amplitude + 300.0 * curvature - 180.0 * max(0.0, slope - 1.0) - 25.0 * abs(len(sequence) - 6)
    candidate = Candidate(model_id, len(sequence), minimum * 10.0, minimum, amplitude, slope, curvature, score)
    return candidate, labels, np.column_stack((materials[:, 0], materials[:, 1]))


def soft_describe(vp: FloatArray, vs: FloatArray, model_id: int) -> Tuple[Candidate, IntArray, FloatArray]:
    """Rank visually layered maps while rejecting compact inclusions and pinches."""
    materials, inverse = np.unique(np.stack((vp.ravel(), vs.ravel()), axis=1), axis=0, return_inverse=True)
    labels = inverse.reshape((70, 70)).astype(np.int64)
    all_runs = [column_runs(labels, column) for column in range(70)]
    min_widths = np.asarray([min(widths) for _, widths in all_runs], dtype=np.float64)
    sequences = {sequence for sequence, _ in all_runs}
    component_areas: List[int] = []
    for value in range(len(materials)):
        components, count = label(labels == value)
        if count:
            component_areas.extend(np.bincount(components.ravel())[1:].tolist())
    small = sum(area < 25 for area in component_areas)
    # Reject detached inclusions under 25 cells and models dominated by pinched layers.
    if small or float(np.quantile(min_widths, 0.15)) < 4.0 or len(sequences) > 3:
        raise ValueError("Not a clean layered candidate")
    # Median run boundaries are a stable reference when a layer locally reaches an outer edge.
    layered = sorted(all_runs, key=lambda item: len(item[0]))
    reference = layered[len(layered)//2][0]
    if len(reference) < 4 or len(reference) > 8:
        raise ValueError("Unsuitable layer count")
    if any(len(sequence) != len(reference) for sequence, _ in all_runs):
        raise ValueError("Layer count differs among columns")
    positions: List[List[float]] = []
    for sequence, widths in all_runs:
        positions.append((np.cumsum(widths[:-1]) * 10.0).tolist())
    boundaries = np.asarray(positions, dtype=np.float64).T
    amplitude = float(np.max(np.ptp(boundaries, axis=1))) if boundaries.size else 0.0
    slope = float(np.max(np.abs(np.diff(boundaries, axis=1) / 10.0))) if boundaries.shape[1] > 1 else 0.0
    curvature = float(np.mean(np.abs(np.diff(boundaries, n=2, axis=1)))) if boundaries.shape[1] > 2 else 0.0
    score = 1_000.0 * float(np.quantile(min_widths, 0.15)) + 8.0 * amplitude - 250.0 * max(0.0, slope - 1.2) - 80.0 * len(sequences)
    return Candidate(model_id, len(reference), float(np.quantile(min_widths, 0.15))*10.0, int(min(min_widths)), amplitude, slope, curvature, score), labels, np.column_stack((materials[:, 0], materials[:, 1]))


def exact_sequence(vp: FloatArray, vs: FloatArray) -> Tuple[bool, int, int, float, float, float]:
    """Fast exact test: every column has equal ordered VP/VS run sequence."""
    pair = np.ascontiguousarray(np.stack((vp, vs), axis=-1)).view("V16").reshape((70, 70))
    reference: NDArray[np.void] | None = None
    positions: List[FloatArray] = []
    minimum = 70
    for column in range(70):
        values = pair[:, column]
        starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
        sequence = values[starts]
        if reference is None:
            reference = sequence
        elif len(sequence) != len(reference) or not np.array_equal(sequence, reference):
            return False, 0, 0, 0.0, 0.0, 0.0
        minimum = min(minimum, int(np.min(np.diff(np.r_[starts, 70]))))
        positions.append(np.cumsum(np.diff(np.r_[starts, 70])[:-1]) * 10.0)
    boundaries = np.asarray(positions, dtype=np.float64).T
    amplitude = float(np.max(np.ptp(boundaries, axis=1))) if boundaries.size else 0.0
    slope = float(np.max(np.abs(np.diff(boundaries, axis=1) / 10.0))) if boundaries.shape[1] > 1 else 0.0
    curvature = float(np.mean(np.abs(np.diff(boundaries, n=2, axis=1)))) if boundaries.shape[1] > 2 else 0.0
    assert reference is not None
    return True, len(reference), minimum, amplitude, slope, curvature


def select() -> Tuple[List[Candidate], List[int]]:
    candidates: List[Candidate] = []
    counts = [0, 0, 0, 0]
    for path in sorted((BINARY).glob("ECVB_*.npz")):
        with np.load(path, allow_pickle=False) as block:
            ids = block["ids"]
            vp_all = block["vp"]
            vs_all = block["vs"]
            for index, model_id in enumerate(ids):
                valid, layers, minimum, amplitude, slope, curvature = exact_sequence(vp_all[index], vs_all[index])
                if not valid:
                    continue
                for threshold in range(1, 5):
                    counts[threshold-1] += int(minimum >= threshold)
                # Prefer broad layers, smooth but visually non-flat interfaces, and slopes < 1.
                score = 1_000_000.0 * minimum + 10.0 * amplitude + 400.0 * curvature - 2_000.0 * max(0.0, slope - 1.0) - 50.0 * abs(layers - 6)
                candidate = Candidate(int(model_id), layers, minimum * 10.0, minimum, amplitude, slope, curvature, score)
                candidates.append(candidate)
    if not candidates:
        raise RuntimeError("No strictly layered ECVB model found")
    return sorted(candidates, key=lambda item: item.score, reverse=True), counts


def render(labels: IntArray, boundaries: FloatArray, materials: FloatArray, target: Path) -> None:
    values = materials[labels]
    vp = values[..., 0]
    vs = values[..., 1]
    manifest = json.loads((BINARY / "manifest.json").read_text(encoding="utf-8"))
    bounds = manifest["bounds"]
    red = (vp-bounds["vp_min"]) / (bounds["vp_max"]-bounds["vp_min"])
    green = (vs-bounds["vs_min"]) / (bounds["vs_max"]-bounds["vs_min"])
    image = Image.fromarray(np.rint(np.clip(np.stack((red, green, np.zeros_like(red)), axis=-1), 0, 1) * 255).astype(np.uint8)).resize((700, 700), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(image)
    for boundary in boundaries:
        draw.line([(float(x*10), float(y)) for x, y in enumerate(boundary)], fill="white", width=2)
    image.save(image_directory(target.parent) / target.name)


def main() -> None:
    candidates, threshold_counts = select()
    output = ROOT / "pilot_geometry"
    output.mkdir(exist_ok=True)
    selected = candidates[0]
    start = selected.model_id // 1000 * 1000
    with np.load(BINARY / f"ECVB_{start:05d}_{start+999:05d}.npz", allow_pickle=False) as block:
        index = selected.model_id - start
        vp = np.asarray(block["vp"][index], dtype=np.float64)
        vs = np.asarray(block["vs"][index], dtype=np.float64)
    materials, inverse = np.unique(np.stack((vp.ravel(), vs.ravel()), axis=1), axis=0, return_inverse=True)
    labels = inverse.reshape((70, 70)).astype(np.int64)
    boundaries = np.empty((selected.layer_count - 1, 70), dtype=np.float64)
    for column in range(70):
        _, widths = column_runs(labels, column)
        boundaries[:, column] = np.cumsum(widths[:-1]) * 10.0
    np.savez_compressed(output / "selected_model.npz", model_id=np.int64(selected.model_id), vp=vp, vs=vs, material_labels=labels, materials_vp_vs=materials, interface_depth_m=boundaries, x_m=np.arange(70, dtype=np.float64)*10.0 + 5.0)
    result = {
        "selection_rule": "Same ordered exact (VP,VS) material sequence in every column; no material is removed; rank candidates first by minimum layer thickness, then curvature with a slope penalty.",
        "exact_topology_candidate_counts_by_minimum_thickness_cells": {str(i + 1): count for i, count in enumerate(threshold_counts)},
        "selected": asdict(selected),
        "top_candidates": [asdict(item) for item in candidates[:12]],
        "coordinate_convention": "x increases right; interface_depth_m is depth below top surface and increases downwards; source arrays retain first row at top.",
        "limits": "Selection proves only raster topology. It does not prove spline validity, mesh element Jacobians, or wave-simulation convergence.",
    }
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    render(labels, boundaries, materials, output / "selected_model.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
