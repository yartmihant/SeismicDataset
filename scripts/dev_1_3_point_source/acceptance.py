"""Приёмка центральной серии по завершённым p4/p5 и временным контролям."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.dev_1_3_point_source.compare import COMPONENTS, UNITS, RECEIVER_COORDINATE_TOLERANCE_M, RunData, compatible, load_run, metrics, receiver_mask, sha256
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, sequence


@dataclass(frozen=True, slots=True)
class ComponentAcceptance:
    """Отдельные критерии нельзя заменять сравнением относительных процентов."""

    p_relative_l2: float | None
    dt_relative_l2: float | None
    p_difference_l2: float
    dt_difference_l2: float | None
    p_within_one_percent: bool | None
    dt_within_point_one_percent: bool | None
    dt_below_one_fifth_p_absolute: bool | None
    dt_to_p_absolute_ratio: float | None
    accepted: bool | None


def judge_component(p_metric: dict[str, object], dt_metric: dict[str, object] | None) -> ComponentAcceptance:
    """Проверить 1%, 0,1% и абсолютный запас 1/5; неизвестное не считается успехом."""
    p_raw = p_metric["relative_l2"]
    p_relative = None if p_raw is None else number(p_raw)
    p_abs = number(p_metric["difference_discrete_l2"])
    p_pass = None if p_relative is None else p_relative <= 0.01
    if dt_metric is None:
        return ComponentAcceptance(p_relative, None, p_abs, None, p_pass, None, None, None, None)
    dt_raw = dt_metric["relative_l2"]
    dt_relative = None if dt_raw is None else number(dt_raw)
    dt_abs = number(dt_metric["difference_discrete_l2"])
    dt_pass = None if dt_relative is None else dt_relative <= 0.001
    absolute_pass = dt_abs <= p_abs / 5.0
    ratio = dt_abs / p_abs if p_abs > 0 else None
    accepted = None if p_pass is None or dt_pass is None else p_pass and dt_pass and absolute_pass
    return ComponentAcceptance(p_relative, dt_relative, p_abs, dt_abs, p_pass, dt_pass, absolute_pass, ratio, accepted)


def window(record: dict[str, object], begin: float, end: float = 2.0,
           radius: float = 20.0) -> dict[str, object]:
    for value in sequence(record["windows"]):
        entry = mapping(value)
        if entry["time_window_s"] == [begin, end] and entry["excluded_radius_m"] == radius:
            return mapping(entry["components"])
    raise ValueError("В отчёте нет требуемого окна и физической маски приёмников")


@dataclass(frozen=True, slots=True)
class Case:
    model_id: int
    kind: str
    p4: RunData | None
    p5: RunData | None
    dt_half: RunData | None


def select_cases(runs: Sequence[RunData]) -> list[Case]:
    """Найти ровно основной p4/p5 и p4 с удвоенным числом подшагов на модель/среду."""
    if any(run.source_x_m != 350.0 or run.end_s != 2.0 for run in runs):
        raise ValueError("Приёмка относится только к центральному источнику x=350 м и полным 2 с")
    keys = [(run.model_id, run.kind, run.order, run.substeps) for run in runs]
    if len(keys) != len(set(keys)):
        raise ValueError("Дублирующиеся запуски одного порядка и шага")
    result: list[Case] = []
    for model_id in (12404, 65):
        for kind in ("raster", "curved"):
            subset = [run for run in runs if (run.model_id, run.kind) == (model_id, kind)]
            p5_runs = [run for run in subset if run.order == 5]
            if len(p5_runs) > 1:
                raise ValueError("Для приёмки передать только основной p5, без дополнительных p5 временных серий")
            p5 = p5_runs[0] if p5_runs else None
            p4_runs = sorted((run for run in subset if run.order == 4), key=lambda run: run.substeps)
            base_k = p5.substeps if p5 is not None else p4_runs[0].substeps if p4_runs else None
            p4 = next((run for run in p4_runs if run.substeps == base_k), None)
            dt_half = next((run for run in p4_runs if base_k is not None and run.substeps == 2 * base_k), None)
            if any(run is not p4 and run is not dt_half for run in p4_runs):
                raise ValueError("Найдены лишние или несовместимые p4 временные шаги")
            result.append(Case(model_id, kind, p4, p5, dt_half))
    return result


def case_record(case: Case) -> dict[str, object]:
    record: dict[str, object] = {"model_id": case.model_id, "kind": case.kind, "source_x_m": 350.0,
                                "acceptance_scope": "full_0_to_2s_only_receivers_abs_x_minus_source_ge_20m"}
    missing = [name for name, run in (("p4", case.p4), ("p5", case.p5), ("p4_dt_half", case.dt_half)) if run is None]
    record["missing_runs"] = missing
    record["all_components_pass"] = None
    if case.p4 is None or case.p5 is None:
        record["status"] = "pending_main_cases"
        return record
    p_record = metrics(case.p4, case.p5, "p")
    dt_record = metrics(case.p4, case.dt_half, "dt") if case.dt_half is not None else None
    p_full = window(p_record, 0.0)
    dt_full = window(dt_record, 0.0) if dt_record is not None else None
    components: dict[str, object] = {}
    decisions: list[bool | None] = []
    for name, unit in zip(COMPONENTS, UNITS, strict=True):
        decision = judge_component(mapping(p_full[name]), mapping(dt_full[name]) if dt_full is not None else None)
        components[name] = {"unit": unit, **asdict(decision)}
        decisions.append(decision.accepted)
    record["full_far20"] = {"time_window_s": [0.0, 2.0], "excluded_radius_m": 20.0, "components": components}
    record["full_all_receivers"] = {"time_window_s": [0.0, 2.0], "excluded_radius_m": 0.0,
                                    "p": window(p_record, 0.0, radius=0.0),
                                    "dt": window(dt_record, 0.0, radius=0.0) if dt_record is not None else None,
                                    "includes_force_application_node": True, "used_for_far20_acceptance": False}
    record["late_far20_diagnostic"] = {"time_window_s": [1.0, 2.0], "excluded_radius_m": 20.0,
                                       "p": window(p_record, 1.0),
                                       "dt": window(dt_record, 1.0) if dt_record is not None else None,
                                       "used_for_acceptance": False}
    record["base_substeps_per_ms"] = case.p4.substeps
    record["reference_norms"] = "p correction normalized to p5; dt correction normalized to fine-time p4; one-fifth comparison uses absolute discrete L2 differences"
    if any(value is None for value in decisions):
        record["status"] = "pending_time_control" if case.dt_half is None else "indeterminate_near_zero_reference"
    else:
        passed = all(decisions)
        record["all_components_pass"] = passed
        record["status"] = "criteria_far20_passed" if passed else "criteria_far20_failed"
    return record


def attribution(raster: Case, curved: Case) -> dict[str, object]:
    """Геометрическая разность / сумма абсолютных p- и dt-поправок обеих сред."""
    result: dict[str, object] = {"model_id": raster.model_id, "source_x_m": 350.0}
    if raster.p4 is None or raster.p5 is None or raster.dt_half is None or curved.p4 is None or curved.p5 is None or curved.dt_half is None:
        result["status"] = "pending_inputs"
        return result
    compatible(curved.p4, raster.p4, "geometry")
    for case in (raster, curved):
        assert case.p4 is not None and case.p5 is not None and case.dt_half is not None
        compatible(case.p4, case.p5, "p")
        compatible(case.p4, case.dt_half, "dt")
    selected = receiver_mask(raster.p4.coordinates_m, 350.0, 20.0)
    windows: list[dict[str, object]] = []
    for begin in (0.0, 1.0):
        selected_time = raster.p4.time_s >= begin
        components: dict[str, object] = {}
        for index, (name, unit) in enumerate(zip(COMPONENTS, UNITS, strict=True)):
            def norm_difference(left: RunData, right: RunData) -> float:
                return float(np.linalg.norm((left.surface - right.surface)[selected_time][:, selected, index]))

            effect = norm_difference(curved.p4, raster.p4)
            terms = {
                "raster_p": norm_difference(raster.p4, raster.p5),
                "curved_p": norm_difference(curved.p4, curved.p5),
                "raster_dt": norm_difference(raster.p4, raster.dt_half),
                "curved_dt": norm_difference(curved.p4, curved.dt_half),
            }
            denominator = sum(terms.values())
            components[name] = {"unit": unit, "geometry_effect_discrete_l2": effect,
                                "corrections_discrete_l2": terms, "correction_sum_discrete_l2": denominator,
                                "effect_over_correction_sum": effect / denominator if denominator > 0 else None,
                                "five_times_margin": effect >= 5 * denominator if denominator > 0 else None}
        windows.append({"time_window_s": [begin, 2.0], "excluded_radius_m": 20.0, "components": components})
    result.update({"status": "computed", "windows": windows,
                   "limitation": "Observed p/dt corrections are not rigorous error bounds; layer reflection and solver validation are separate checks"})
    return result


def acceptance(run_paths: Sequence[Path], output: Path) -> dict[str, object]:
    """Сохранить новый компактный отчёт; отсутствие контролей явно остаётся pending."""
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    runs = [load_run(path) for path in run_paths]
    cases = select_cases(runs)
    records = [case_record(case) for case in cases]
    decisions = [record["all_components_pass"] for record in records]
    finished = all(value is not None for value in decisions)
    passed = all(decisions) if finished else None
    record: dict[str, object] = {
        "chapter": "1.3", "scope": "central_source_350m_only", "expected_main_runs": 8,
        "expected_time_controls": 4, "completed_input_runs": len(runs),
        "status": "criteria_far20_passed" if passed else "criteria_far20_failed" if finished else "pending_inputs_or_indeterminate_reference",
        "acceptance_scope": "far20_receivers_only_full_0_to_2s; all-receiver results are disclosed separately in every complete case",
        "all_components_pass": passed,
        "criteria": {"full_window_s": [0.0, 2.0], "source_exclusion_radius_m": 20.0,
                     "receiver_mask_boundary_tolerance_m": RECEIVER_COORDINATE_TOLERANCE_M,
                     "p_relative_l2_max": 0.01, "dt_relative_l2_max": 0.001, "dt_absolute_over_p_absolute_max": 0.2,
                     "late_window_used_for_acceptance": False},
        "cases": records, "geometry_attribution": [attribution(cases[0], cases[1]), attribution(cases[2], cases[3])],
        "inputs": [{"path": str(run.path), "metadata_sha256": sha256(run.path / "metadata.json"),
                    "surface_sha256": sha256(run.path / "surface.npz")} for run in runs],
        "limitations": ["Только центральный источник; боковые положения не исследовались в этой серии.",
                        "Пороги 1%, 0,1% и отношение 1:5 — рабочие критерии протокола, а не допуск, заданный пользователем.",
                        "Статус criteria_far20 относится только к приёмникам на расстоянии не менее 20 м от источника; полная запись со всеми приёмниками приведена отдельно.",
                        "Сходимость самого узла приложения точечной силы не утверждается и не следует из выполнения критериев far20.",
                        "Приёмка по указанным p/dt-критериям не заменяет отдельного контроля отражения от слоя.",
                        "Наблюдаемые поправки не являются строгими оценками ошибки или независимой валидацией решателя."],
    }
    output.mkdir(parents=True)
    with (output / "acceptance.json").open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    record = acceptance(args.runs, args.out)
    print(f"Приёмка: {record['status']}; {args.out / 'acceptance.json'}")


if __name__ == "__main__":
    main()
