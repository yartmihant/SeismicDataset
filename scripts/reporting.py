"""Typed readers for the compact, already computed research results."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Tuple, cast


@dataclass(frozen=True)
class DatasetSummary:
    name: str
    models: int
    vp: Tuple[float, float]
    vs: Tuple[float, float]
    materials: Tuple[int, int]
    counts: Dict[int, int]


@dataclass(frozen=True)
class ComparisonSummary:
    label: str
    full: Tuple[float, float]
    late: Tuple[float, float]
    final_field: Tuple[float, float]


def mapping(value: object) -> Dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError("Expected a JSON object with string keys")
    return cast(Dict[str, object], value)


def sequence(value: object) -> List[object]:
    if not isinstance(value, list):
        raise ValueError("Expected a JSON list")
    return cast(List[object], value)


def number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("Expected a JSON number")
    return float(value)


def pair(value: object) -> Tuple[float, float]:
    values = sequence(value)
    if len(values) != 2:
        raise ValueError("Expected Ux/Uy pair")
    return number(values[0]), number(values[1])


def read_json(path: Path) -> Dict[str, object]:
    return mapping(json.loads(path.read_text(encoding="utf-8")))


def dataset_summaries(path: Path) -> List[DatasetSummary]:
    groups = mapping(read_json(path)["groups"])
    result: List[DatasetSummary] = []
    for name in ("EFVB", "ECVB"):
        record = mapping(groups[name])
        counts = {int(key): int(number(value)) for key, value in mapping(record["material_count_histogram"]).items()}
        models = int(number(record["models"]))
        if sum(counts.values()) != models:
            raise ValueError(f"Histogram does not cover all models: {name}")
        result.append(DatasetSummary(name, models,
            (number(record["vp_min"]), number(record["vp_max"])),
            (number(record["vs_min"]), number(record["vs_max"])),
            (int(number(record["materials_min"])), int(number(record["materials_max"]))), counts))
    return result


def comparison_summary(path: Path, label: str) -> ComparisonSummary:
    metrics = mapping(read_json(path)["metrics"])
    surface = mapping(metrics["surface"])
    windows = [mapping(value) for value in sequence(surface["time_windows"])]
    late = next(record for record in windows if pair(record["time_window_s"]) == (1.0, 2.0))
    fields = [mapping(value) for value in sequence(metrics["fields"])]
    final = next(record for record in fields if number(record["time_s"]) == 2.0)
    return ComparisonSummary(label, pair(surface["all_surface"]), pair(late["relative_l2"]), pair(final["relative_l2"]))
