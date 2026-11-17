"""Отдельный контроль p3 относительно p4 и p5 без изменения прежней приёмки."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from scripts.dev_1_3_point_source.acceptance import judge_component, window
from scripts.dev_1_3_point_source.compare import (
    COMPONENTS, RECEIVER_COORDINATE_TOLERANCE_M, UNITS, RunData,
    compatible, load_run, metrics, receiver_mask, sha256,
)
from scripts.dev_1_3_point_source.source_diagnostic import localization, source_force_comparison
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping

BASE_SUBSTEPS = {12404: 64, 65: 224}


@dataclass(frozen=True, slots=True)
class P3Case:
    model_id: int
    kind: str
    p3: RunData | None
    p4: RunData | None
    p5: RunData | None
    p3_dt_half: RunData | None


def select_cases(runs: Sequence[RunData]) -> list[P3Case]:
    """Принять только 4 новых p3, их 4 dt-контроля и 8 прежних опорных опытов."""
    if any(run.source_x_m != 350.0 or run.end_s != 2.0 for run in runs):
        raise ValueError("Контроль p3 относится только к центральному источнику и полным 2 с")
    identities = [(r.model_id, r.kind, r.order, r.substeps) for r in runs]
    if len(identities) != len(set(identities)):
        raise ValueError("Дублирующиеся модель/геометрия/порядок/шаг")
    for run in runs:
        base = BASE_SUBSTEPS[run.model_id]
        if run.substeps != base and not (run.order == 3 and run.substeps == 2 * base):
            raise ValueError("Нужны p3/p4/p5 при K=64/224 и только p3 при удвоенном K")
    index = dict(zip(identities, runs, strict=True))
    return [P3Case(model, kind, index.get((model, kind, 3, BASE_SUBSTEPS[model])),
                   index.get((model, kind, 4, BASE_SUBSTEPS[model])),
                   index.get((model, kind, 5, BASE_SUBSTEPS[model])),
                   index.get((model, kind, 3, 2 * BASE_SUBSTEPS[model])))
            for model in (12404, 65) for kind in ("raster", "curved")]


def criteria_record(p_metrics: dict[str, object] | None, dt_metrics: dict[str, object] | None,
                    force_matches: bool | None) -> dict[str, object]:
    """Проверить каждую опору отдельно; отсутствие данных не считается успехом."""
    result: dict[str, object] = {
        "time_window_s": [0.0, 2.0], "excluded_radius_m": 20.0,
        "source_force_matches": force_matches, "all_components_pass": None,
    }
    if p_metrics is None:
        result["status"] = "pending_order_pair"
        return result
    p_full = window(p_metrics, 0.0)
    dt_full = window(dt_metrics, 0.0) if dt_metrics is not None else None
    decisions: list[bool | None] = []
    components: dict[str, object] = {}
    for name, unit in zip(COMPONENTS, UNITS, strict=True):
        decision = judge_component(mapping(p_full[name]), mapping(dt_full[name]) if dt_full is not None else None)
        components[name] = {"unit": unit, **asdict(decision)}
        decisions.append(decision.accepted)
    result["components"] = components
    if force_matches is False:
        result.update({"status": "incompatible_source_force", "all_components_pass": False})
    elif any(item is None for item in decisions) or force_matches is None:
        result["status"] = "pending_time_control" if dt_metrics is None else "indeterminate_reference"
    else:
        passed = all(decisions)
        result.update({"status": "criteria_far20_passed" if passed else "criteria_far20_failed",
                       "all_components_pass": passed})
    return result


def case_record(case: P3Case) -> dict[str, object]:
    missing = [name for name, run in (("p3", case.p3), ("p4", case.p4), ("p5", case.p5),
                                      ("p3_dt_half", case.p3_dt_half)) if run is None]
    dt_record = metrics(case.p3, case.p3_dt_half, "dt") if case.p3 is not None and case.p3_dt_half is not None else None
    record: dict[str, object] = {
        "model_id": case.model_id, "kind": case.kind, "source_x_m": 350.0,
        "base_substeps_per_ms": BASE_SUBSTEPS[case.model_id], "missing_runs": missing,
        "p3_dt": dt_record,
        "acceptance_scope": "full_0_to_2s_only_receivers_abs_x_minus_source_ge_20m",
        "all_receiver_and_late_results": "Included independently in each comparison.windows; only full far20 enters criteria",
    }
    criteria: dict[str, object] = {}
    force_records: dict[str, object] = {}
    source_records: dict[str, object] = {}
    for order, reference in ((4, case.p4), (5, case.p5)):
        key = f"p{order}"
        p_record = metrics(case.p3, reference, "p") if case.p3 is not None and reference is not None else None
        force = source_force_comparison(case.p3, reference) if case.p3 is not None and reference is not None else None
        record[f"p3_vs_{key}"] = p_record
        force_records[key] = force
        criteria[key] = criteria_record(p_record, dt_record, bool(force["identical_history"]) if force is not None else None)
        source_records[key] = (localization(case.p3.surface, reference.surface, case.p3.coordinates_m, 350.0)
                               if case.p3 is not None and reference is not None else None)
    record.update({"criteria_by_reference": criteria, "source_force_by_reference": force_records,
                   "source_localization_by_reference": source_records})
    decisions = [mapping(criteria[key])["all_components_pass"] for key in ("p4", "p5")]
    record["all_references_pass"] = None if missing or any(item is None for item in decisions) else all(decisions)
    record["status"] = ("pending_inputs" if missing else "indeterminate_reference" if any(item is None for item in decisions)
                        else "criteria_far20_passed" if all(decisions) else "criteria_far20_failed")
    return record


def geometry_attribution(raster: P3Case, curved: P3Case, reference_order: int) -> dict[str, object]:
    """Эффект геометрии p3 относительно суммы его p- и dt-поправок обеих сред."""
    if reference_order not in (4, 5):
        raise ValueError("Опорный порядок должен быть 4 или 5")
    raster_ref = raster.p4 if reference_order == 4 else raster.p5
    curved_ref = curved.p4 if reference_order == 4 else curved.p5
    result: dict[str, object] = {"model_id": raster.model_id, "order": 3,
                                "reference_order": reference_order, "source_x_m": 350.0}
    if (raster.p3 is None or curved.p3 is None or raster.p3_dt_half is None or curved.p3_dt_half is None
            or raster_ref is None or curved_ref is None):
        result["status"] = "pending_inputs"
        return result
    compatible(curved.p3, raster.p3, "geometry")
    for base, ref, control in ((raster.p3, raster_ref, raster.p3_dt_half),
                               (curved.p3, curved_ref, curved.p3_dt_half)):
        compatible(base, ref, "p")
        compatible(base, control, "dt")
    selected = receiver_mask(raster.p3.coordinates_m, 350.0, 20.0)
    windows: list[dict[str, object]] = []
    for begin in (0.0, 1.0):
        selected_time = raster.p3.time_s >= begin
        components: dict[str, object] = {}
        for index, (name, unit) in enumerate(zip(COMPONENTS, UNITS, strict=True)):
            def norm_difference(left: RunData, right: RunData) -> float:
                return float(np.linalg.norm((left.surface - right.surface)[selected_time][:, selected, index]))

            effect = norm_difference(curved.p3, raster.p3)
            terms = {"raster_p": norm_difference(raster.p3, raster_ref),
                     "curved_p": norm_difference(curved.p3, curved_ref),
                     "raster_dt": norm_difference(raster.p3, raster.p3_dt_half),
                     "curved_dt": norm_difference(curved.p3, curved.p3_dt_half)}
            total = sum(terms.values())
            components[name] = {"unit": unit, "geometry_effect_discrete_l2": effect,
                                "corrections_discrete_l2": terms, "correction_sum_discrete_l2": total,
                                "effect_over_correction_sum": effect / total if total > 0 else None,
                                "five_times_margin": effect >= 5 * total if total > 0 else None}
        windows.append({"time_window_s": [begin, 2.0], "excluded_radius_m": 20.0,
                        "receiver_count": int(selected.sum()), "components": components})
    result.update({"status": "computed", "windows": windows,
                   "limitation": "Observed corrections are not rigorous error bounds or independent solver validation"})
    return result


def build_report(runs: Sequence[RunData]) -> dict[str, object]:
    """Составить отчёт в памяти, сохранив явный pending для неполного набора."""
    cases = select_cases(runs)
    records = [case_record(case) for case in cases]
    missing = any(record["missing_runs"] for record in records)
    decisions = [record["all_references_pass"] for record in records]
    passed = None if missing or any(item is None for item in decisions) else all(decisions)
    geometry: list[dict[str, object]] = []
    attributions: list[dict[str, object]] = []
    for raster, curved in ((cases[0], cases[1]), (cases[2], cases[3])):
        geometry.append({"model_id": raster.model_id, "order": 3,
                         "status": "computed" if raster.p3 is not None and curved.p3 is not None else "pending_inputs",
                         "comparison": metrics(curved.p3, raster.p3, "geometry")
                         if raster.p3 is not None and curved.p3 is not None else None})
        attributions.extend(geometry_attribution(raster, curved, order) for order in (4, 5))
    return {
        "chapter": "1.3", "scope": "central_source_350m_p3_control", "completed_input_runs": len(runs),
        "expected_new_main_runs": 4, "expected_new_time_controls": 4, "expected_existing_reference_runs": 8,
        "is_final_complete": not missing, "all_references_pass": passed,
        "status": ("pending_inputs" if missing else "indeterminate_reference" if passed is None
                   else "criteria_far20_passed" if passed else "criteria_far20_failed"),
        "acceptance_scope": "full_0_to_2s_far20; p3_vs_p4_and_p3_vs_p5_judged_separately",
        "criteria": {"p_relative_l2_max": 0.01, "dt_relative_l2_max": 0.001,
                     "dt_absolute_over_p_absolute_max": 0.2, "source_exclusion_radius_m": 20.0,
                     "full_window_s": [0.0, 2.0], "late_window_used_for_acceptance": False,
                     "receiver_mask_boundary_tolerance_m": RECEIVER_COORDINATE_TOLERANCE_M},
        "cases": records, "geometry_effect_p3": geometry, "geometry_attribution_by_reference": attributions,
        "inputs": [{"path": str(run.path), "order": run.order,
                    "role": "new_p3_run" if run.order == 3 else "existing_reference_read_only",
                    "metadata_sha256": sha256(run.path / "metadata.json"),
                    "surface_sha256": sha256(run.path / "surface.npz")} for run in runs],
        "limitations": [
            "Пороги 1%, 0,1% и 1:5 — рабочие критерии протокола, а не допуск, заданный пользователем.",
            "Обе опоры (p4, p5) проверяются отдельно; результаты прежней приёмки p4/p5 не изменяются.",
            "Полная линия, радиусы 10/20 м, позднее окно и сам узел источника раскрыты отдельно; far20 не доказывает сходимость точечного значения.",
            "Относительная p-норма использует соответствующую опору; dt-норма — p3 на меньшем шаге; запас 1:5 проверяется по абсолютным разностям.",
            "История заданной силы сравнивается целиком, включая начальный отсчёт t=0.",
            "Наблюдаемые p/dt-поправки не являются строгими оценками ошибки; контроль отражений и независимая валидация решателя остаются отдельными задачами.",
        ],
    }


def write_report(run_paths: Sequence[Path], output: Path, allow_pending: bool = False) -> dict[str, object]:
    output = output.resolve()
    output.relative_to(PROJECT_ROOT / "data")
    if output.exists():
        raise FileExistsError(output)
    report = build_report([load_run(path) for path in run_paths])
    if not report["is_final_complete"] and not allow_pending:
        raise ValueError("Финальному p3-отчёту нужны все 16 входов; промежуточный каталог требует --allow-pending")
    output.mkdir(parents=True)
    with (output / "p3_report.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-pending", action="store_true", help="Только отдельный промежуточный каталог с явным pending")
    args = parser.parse_args(argv)
    report = write_report(args.runs, args.out, args.allow_pending)
    print(f"Контроль p3: {report['status']}; {args.out / 'p3_report.json'}")


if __name__ == "__main__":
    main()
