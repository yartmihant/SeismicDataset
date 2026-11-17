"""Отдельная центральная серия p3: четыре основных опыта и четыре временных контроля."""

from __future__ import annotations

import argparse
from math import isfinite, log
from pathlib import Path
from typing import Sequence

from scripts.dev_1_3_point_source.batch import (
    DATA_ROOT, Job, MeshEvidence, event, execute, file_sha256, plan_record,
    raw_storage_bytes, verify_completed, write_new_json,
)
from scripts.dev_1_3_point_source.run import RunSpec
from scripts.reporting import number, read_json


ALPHA_MAX = -3.0 * 4385.0 * log(0.01) / 175.0
COMMON_SUBSTEPS = {"12404": 64, "65": 224}


def read_p3_evidence(meshes: Path, stability_directory: Path, model: int, kind: str) -> MeshEvidence:
    """Связать прежнюю Q2-сетку с новым отдельным сертификатом p3."""
    mesh = (meshes / f"{model:05d}_{kind}_q2.npz").resolve()
    audit = read_json(mesh.with_suffix(".json"))
    mesh_hash = file_sha256(mesh)
    if audit.get("sha256") != mesh_hash:
        raise ValueError(f"Сетка не совпадает с геометрическим аудитом: {mesh}")
    if (audit.get("model_id"), audit.get("kind"), audit.get("conforming")) != (model, kind, True):
        raise ValueError(f"Неподходящий геометрический аудит: {mesh}")
    if number(audit["det_jacobian_certified_min_m2"]) <= 0:
        raise ValueError(f"Нет положительного сертификата Q2: {mesh}")
    stability = (stability_directory / f"{model:05d}_{kind}_p3_stability.json").resolve()
    record = read_json(stability)
    if Path(str(record["mesh"])).resolve() != mesh or record.get("order") != 3:
        raise ValueError(f"Сертификат относится к другой сетке или порядку: {stability}")
    if record.get("mesh_sha256") != mesh_hash:
        raise ValueError(f"Сертификат p3 не подтверждает SHA256 сетки: {stability}")
    if abs(number(record["alpha_max_s_inv"]) - ALPHA_MAX) > 1e-10:
        raise ValueError(f"Сертификат p3 использует другое демпфирование: {stability}")
    elements = int(number(record["elements"]))
    nodes = int(number(record["nodes"]))
    minimum = int(number(record["substeps_per_ms"]))
    dt_start = number(record["dt_start_s"])
    chosen = COMMON_SUBSTEPS[str(model)]
    if elements != int(number(audit["elements"])) or nodes < 1 or minimum < 1:
        raise ValueError(f"Неверные размеры/шаг сертификата: {stability}")
    if not isfinite(dt_start) or dt_start <= 0 or chosen < minimum or 0.001 / chosen > dt_start:
        raise ValueError(f"Прежний общий K={chosen} не удовлетворяет сертификату p3: {stability}")
    return MeshEvidence(mesh, mesh_hash, stability, file_sha256(stability), nodes, elements, minimum)


def build_jobs(root: Path, meshes: Path, stability_directory: Path) -> list[Job]:
    """Сохранить K основных p4/p5 и удвоить его только у контролей p3."""
    evidence = {(model, kind): read_p3_evidence(meshes, stability_directory, model, kind)
                for model in (12404, 65) for kind in ("raster", "curved")}
    jobs: list[Job] = []
    for phase in ("main", "dt_control"):
        for model in (12404, 65):
            for kind in ("raster", "curved"):
                fields = phase == "main"
                substeps = COMMON_SUBSTEPS[str(model)] * (1 if fields else 2)
                spec = RunSpec(model, kind, 3, 350.0, substeps, 2.0, "cu", "/gpu/cuda/gen",
                               ALPHA_MAX, fields, "compact")
                parent = root if fields else root / "controls_dt"
                output = parent / f"{model:05d}" / kind / "p3/x350"
                item = evidence[model, kind]
                jobs.append(Job(f"{phase}_{model:05d}_{kind}_p3_x350", phase, output, spec, item,
                                raw_storage_bytes(item.nodes, item.elements, spec)))
    return jobs


def manifest_record(jobs: Sequence[Job]) -> dict[str, object]:
    """Один неизменяемый общий план; выбор очереди не меняет его содержимого."""
    record = plan_record(jobs, COMMON_SUBSTEPS)
    record.update({
        "schema": "seismic_chapter_1_3_p3_batch_v1",
        "scope": "p3_only_central_source_350m_two_models_two_geometries",
        "dt_control": "p3, source x=350 m, substeps doubled, both models and both geometries; surface/source only",
        "reference_series": str(DATA_ROOT / "central_cu_v1"),
        "reference_policy": "Existing p4/p5 results are read-only references; no p4/p5 job is present in this plan",
        "execution_queues": {kind: [job.job_id for job in jobs if job.spec.kind == kind]
                             for kind in ("raster", "curved")},
        "queue_event_policy": "Scoped executor events live under execution_queues/<kind>; root selection_completed does not mean eight completed jobs",
    })
    return record


def verified_count(jobs: Sequence[Job]) -> int:
    """Частичный соседний процесс ожидается; неверный завершённый результат — ошибка."""
    count = 0
    for job in jobs:
        if (job.output / "metadata.json").is_file() and verify_completed(job):
            count += 1
    return count


def execute_selection(root: Path, jobs: Sequence[Job], kind: str) -> int:
    """Изолировать события четырёх выбранных заданий от общего итога восьми."""
    selected = [job for job in jobs if job.spec.kind == kind]
    queue_root = root / "execution_queues" / kind
    selection: dict[str, object] = {
        "scope": "selected_geometry_only", "kind": kind,
        "selected_run_count": len(selected), "expected_overall_run_count": len(jobs),
        "parent_manifest": str(root / "batch_manifest.json"),
        "parent_manifest_sha256": file_sha256(root / "batch_manifest.json"),
        "jobs": [job.record() for job in selected],
        "local_completion_event_scope": "all_planned_runs_completed under this queue means only its selected four jobs",
    }
    selection_path = queue_root / "selection_manifest.json"
    if queue_root.exists():
        if not selection_path.is_file() or read_json(selection_path) != selection:
            raise ValueError(f"Сохранённый выбор выполнения отличается: {queue_root}")
    else:
        queue_root.mkdir(parents=True)
        write_new_json(selection_path, selection)
    code = execute(queue_root, selected)
    if code != 0:
        event(root, "selection_pending_review", details={"kind": kind, "returncode": code})
        return code
    completed = verified_count(jobs)
    event(root, "selection_completed", details={"kind": kind, "selected_run_count": len(selected),
                                                "overall_verified_run_count": completed, "overall_expected_run_count": len(jobs)})
    if completed == len(jobs):
        event(root, "all_planned_runs_completed", details={"verified_run_count": completed})
    print(f"Выбор {kind}: завершено {len(selected)}; общий проверенный итог {completed}/{len(jobs)}", flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA_ROOT / "central_p3_cu_v1")
    parser.add_argument("--meshes", type=Path, default=DATA_ROOT / "meshes_v1")
    parser.add_argument("--stability-dir", type=Path, default=DATA_ROOT / "p3_preflight_v1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--verify", action="store_true", help="Только проверить все восемь результатов; не запускать решатель")
    parser.add_argument("--kind", choices=("raster", "curved"), help="Выбрать одну очередь исполнения из общего плана")
    args = parser.parse_args(argv)
    if args.kind is not None and not args.run:
        parser.error("--kind выбирает только исполнение с --run; общий manifest всегда содержит восемь заданий")
    root = args.root.resolve()
    root.relative_to(DATA_ROOT.resolve())
    if root == DATA_ROOT.resolve():
        raise ValueError("Нужен отдельный каталог новой серии внутри data/dev_1_3_point_source")
    jobs = build_jobs(root, args.meshes.resolve(), args.stability_dir.resolve())
    record = manifest_record(jobs)
    manifest = root / "batch_manifest.json"
    if root.exists():
        if not args.run and not args.verify:
            raise FileExistsError(f"Dry-run создаёт только новый каталог: {root}")
        if not manifest.is_file() or read_json(manifest) != record:
            raise ValueError(f"План отсутствует или отличается; прежние данные не изменены: {root}")
    else:
        if args.verify:
            raise FileNotFoundError(manifest)
        root.mkdir(parents=True)
        write_new_json(manifest, record)
    print(f"План p3: 4 основных + 4 контроля; K={COMMON_SUBSTEPS}; до сжатия {number(record['total_uncompressed_upper_bound_bytes']) / 1024**3:.2f} GiB", flush=True)
    if args.verify:
        for job in jobs:
            if not verify_completed(job):
                raise RuntimeError(f"Результат отсутствует: {job.output}")
        print("Все восемь p3-результатов проверены; решатель не запускался.", flush=True)
    elif args.run:
        raise SystemExit(execute(root, jobs) if args.kind is None else execute_selection(root, jobs, args.kind))
    else:
        print(f"Только план, решатель не запущен: {manifest}", flush=True)


if __name__ == "__main__":
    main()
