"""Неизменяемый план и последовательный запуск серии 1.3 с проверкой места."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from math import ceil, isfinite, log
from pathlib import Path
import shutil
import subprocess
import sys
from time import time_ns
from typing import Sequence

from scripts.dev_1_3_point_source.physics import ExperimentSpec
from scripts.dev_1_3_point_source.run import RunSpec
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, number, read_json, sequence

DATA_ROOT = PROJECT_ROOT / "data/dev_1_3_point_source"
RESERVE_BYTES = 1024**3


@dataclass(frozen=True, slots=True)
class MeshEvidence:
    """Зафиксированные сетка, её аудит и оценка устойчивости заданного p."""

    mesh_path: Path
    mesh_sha256: str
    stability_path: Path
    stability_sha256: str
    nodes: int
    elements: int
    minimum_substeps: int


@dataclass(frozen=True, slots=True)
class Job:
    """Одно отдельное возбуждение, неизменяемый физический и файловый контракт."""

    job_id: str
    phase: str
    output: Path
    spec: RunSpec
    evidence: MeshEvidence
    uncompressed_upper_bound_bytes: int

    def command(self) -> list[str]:
        result = [sys.executable, "-m", "scripts.dev_1_3_point_source.run",
                  "--model", str(self.spec.model_id), "--kind", self.spec.kind,
                  "--order", str(self.spec.order), "--source-x", f"{self.spec.source_x_m:g}",
                  "--substeps", str(self.spec.substeps), "--backend", self.spec.backend,
                  "--ceed-resource", self.spec.ceed_resource,
                  "--field-layout", self.spec.field_layout,
                  "--alpha-max", repr(self.spec.alpha_max_s_inv), "--t-end", "2.0",
                  "--mesh", str(self.evidence.mesh_path), "--stability", str(self.evidence.stability_path),
                  "--out", str(self.output)]
        if not self.spec.fields:
            result.append("--no-fields")
        return result

    def record(self) -> dict[str, object]:
        return {"job_id": self.job_id, "phase": self.phase, "output": str(self.output),
                "run_spec": asdict(self.spec), "mesh_path": str(self.evidence.mesh_path),
                "mesh_sha256": self.evidence.mesh_sha256,
                "stability_path": str(self.evidence.stability_path),
                "stability_sha256": self.evidence.stability_sha256,
                "nodes": self.evidence.nodes, "elements": self.evidence.elements,
                "uncompressed_upper_bound_bytes": self.uncompressed_upper_bound_bytes,
                "required_free_bytes_before_start": self.uncompressed_upper_bound_bytes + RESERVE_BYTES,
                "command": self.command()}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def read_evidence(directory: Path, model: int, kind: str, order: int,
                  alpha_max: float) -> MeshEvidence:
    """Связать текущий NPZ с геометрическим аудитом и оценкой устойчивости."""
    mesh_path = (directory / f"{model:05d}_{kind}_q2.npz").resolve()
    audit = read_json(mesh_path.with_suffix(".json"))
    mesh_hash = file_sha256(mesh_path)
    if audit.get("sha256") != mesh_hash:
        raise ValueError(f"Сетка не совпадает с геометрическим аудитом: {mesh_path}")
    if (audit.get("model_id"), audit.get("kind"), audit.get("conforming")) != (model, kind, True):
        raise ValueError(f"Неподходящий геометрический аудит: {mesh_path}")
    if number(audit["det_jacobian_certified_min_m2"]) <= 0:
        raise ValueError(f"Нет положительного сертификата Q2: {mesh_path}")
    stability_path = (directory / f"{model:05d}_{kind}_p{order}_stability.json").resolve()
    stability = read_json(stability_path)
    if Path(str(stability["mesh"])).resolve() != mesh_path or stability.get("order") != order:
        raise ValueError(f"Оценка устойчивости относится к другой сетке/p: {stability_path}")
    if not isfinite(alpha_max) or abs(number(stability["alpha_max_s_inv"]) - alpha_max) > 1e-10:
        raise ValueError(f"Для выбранного α нужна соответствующая оценка устойчивости: {stability_path}")
    elements = int(number(stability["elements"]))
    nodes = int(number(stability["nodes"]))
    minimum = int(number(stability["substeps_per_ms"]))
    if elements != int(number(audit["elements"])) or nodes < 1 or minimum < 1:
        raise ValueError(f"Неверные размеры/шаг в оценке устойчивости: {stability_path}")
    return MeshEvidence(mesh_path, mesh_hash, stability_path, file_sha256(stability_path), nodes, elements, minimum)


def raw_storage_bytes(nodes: int, elements: int, spec: RunSpec) -> int:
    """Верхняя оценка полезной нагрузки массивов до сжатия, плюс запас ZIP/JSON."""
    frames = 201
    surface = 2001 * 141 * 4 * 8 + 2001 * 8 + 141 * (2 * 8 + 8)
    source_history = (spec.steps + 1) * 5 * 8
    saved_components = 8 if spec.field_layout == "full" else 4
    fields = frames * (nodes * saved_components * 8 + 8) if spec.fields else 0
    geometry = nodes * (2 * 8 + 8 + 1) + elements * ((spec.order + 1)**2 * 8 + 2 * 8) if spec.fields else 0
    energy = frames * 4 * 8 if spec.fields else 0
    # 16 MiB covers ZIP headers/NPY descriptors, reference coordinates and JSON.
    return surface + source_history + fields + geometry + energy + 16 * 1024**2


def build_jobs(root: Path, mesh_directory: Path, sources: Sequence[float], backend: str,
               resource: str, alpha_max: float, fields: bool,
               field_layout: str = "full", fields_orders: Sequence[int] = (4, 5)) -> tuple[list[Job], dict[str, int]]:
    """Восемь оценок дают один общий K на модель, округлённый вверх до восьми."""
    if not sources or len(set(sources)) != len(sources) or any(x not in ExperimentSpec().source_x_m for x in sources):
        raise ValueError("Нужны различные источники из 0,175,350,525,700")
    if not fields_orders or len(set(fields_orders)) != len(fields_orders) or any(order not in (4, 5) for order in fields_orders):
        raise ValueError("Порядки записи полей должны быть различными значениями 4/5")
    evidence: dict[tuple[int, str, int], MeshEvidence] = {}
    common: dict[str, int] = {}
    for model in (12404, 65):
        for kind in ("raster", "curved"):
            for order in (4, 5):
                evidence[model, kind, order] = read_evidence(mesh_directory, model, kind, order, alpha_max)
        maximum = max(evidence[model, kind, order].minimum_substeps for kind in ("raster", "curved") for order in (4, 5))
        common[str(model)] = 8 * ceil(maximum / 8)
    jobs: list[Job] = []

    def add(model: int, kind: str, order: int, source: float, phase: str) -> None:
        substeps = common[str(model)] * (2 if phase == "dt_control" else 1)
        save_fields = fields and order in fields_orders and phase == "main"
        spec = RunSpec(model, kind, order, source, substeps, 2.0, backend, resource, alpha_max, save_fields, field_layout)
        parent = root / "controls_dt" if phase == "dt_control" else root
        output = parent / f"{model:05d}" / kind / f"p{order}" / f"x{source:g}"
        item = evidence[model, kind, order]
        jobs.append(Job(f"{phase}_{model:05d}_{kind}_p{order}_x{source:g}", phase, output, spec,
                        item, raw_storage_bytes(item.nodes, item.elements, spec)))

    # First central pairs and their time controls; expensive remaining shots follow.
    if 350.0 in sources:
        for model in (12404, 65):
            for kind in ("raster", "curved"):
                for order in (4, 5):
                    add(model, kind, order, 350.0, "main")
    for model in (12404, 65):
        for kind in ("raster", "curved"):
            add(model, kind, 4, 350.0, "dt_control")
    for source in sources:
        if source == 350.0:
            continue
        for model in (12404, 65):
            for kind in ("raster", "curved"):
                for order in (4, 5):
                    add(model, kind, order, source, "main")
    return jobs, common


def plan_record(jobs: Sequence[Job], common: dict[str, int]) -> dict[str, object]:
    return {"schema": "seismic_chapter_1_3_batch_v1", "status": "plan_only_no_runs_performed",
            "main_run_count": sum(job.phase == "main" for job in jobs), "dt_control_count": 4,
            "common_substeps_per_ms": common,
            "dt_control": "p4, source x=350 m, substeps doubled, both models and both geometries; surface/source records only, no all-node fields",
            "main_field_run_count": sum(job.phase == "main" and job.spec.fields for job in jobs),
            "main_fields_orders": sorted({job.spec.order for job in jobs if job.phase == "main" and job.spec.fields}),
            "disk_reserve_bytes_per_start": RESERVE_BYTES,
            "total_uncompressed_upper_bound_bytes": sum(job.uncompressed_upper_bound_bytes for job in jobs),
            "resume_policy": "Only completed exact-spec exact-mesh outputs are skipped; incomplete outputs are never removed or overwritten",
            "stability_limit": "Conservative starting estimates; p/dt convergence and layer reflection checks remain separate",
            "jobs": [job.record() for job in jobs]}


def verify_completed(job: Job) -> bool:
    """Вернуть False только для отсутствующего каталога; любой частичный запрещён."""
    if not job.output.exists():
        return False
    metadata_path = job.output / "metadata.json"
    if not metadata_path.is_file():
        raise RuntimeError(f"Незавершённый каталог сохранён; автоматический повтор запрещён: {job.output}")
    metadata = read_json(metadata_path)
    if metadata.get("status") != "completed" or metadata.get("run_spec") != asdict(job.spec):
        raise ValueError(f"Завершённый результат имеет другую постановку: {job.output}")
    if metadata.get("experiment_spec") != asdict(ExperimentSpec()):
        # JSON serializes source_x_m tuples as lists; normalize the expected value.
        expected = json.loads(json.dumps(asdict(ExperimentSpec())))
        if metadata.get("experiment_spec") != expected:
            raise ValueError(f"Изменены физические параметры: {job.output}")
    provenance = mapping(metadata["provenance"])
    found = False
    for value in sequence(provenance["files"]):
        record = mapping(value)
        if Path(str(record["path"])).resolve() == job.evidence.mesh_path:
            found = record.get("sha256") == job.evidence.mesh_sha256
    if not found:
        raise ValueError(f"Нет совпадающего SHA256 сетки: {job.output}")
    if mapping(metadata["stability_evidence"]).get("sha256") != job.evidence.stability_sha256:
        raise ValueError(f"Другая оценка устойчивости: {job.output}")
    runtime = mapping(metadata["runtime"])
    counts = mapping(metadata["recorded_counts"])
    if runtime.get("steps_count") != job.spec.steps or abs(number(runtime["current_time_s"]) - 2.0) > 2e-10:
        raise ValueError(f"Неполное физическое время: {job.output}")
    if counts != {"surface": 2001, "source": job.spec.steps + 1, "fields": 201 if job.spec.fields else 0}:
        raise ValueError(f"Неполные записи: {job.output}")
    for name in ("surface.npz", "source_history.npz"):
        if not (job.output / name).is_file():
            raise FileNotFoundError(job.output / name)
    if job.spec.fields:
        needed = (job.output / "fields/geometry.npz", job.output / "energy_balance.npz")
        for path in (*needed, *(job.output / f"fields/frame_{index:04d}.npz" for index in range(201))):
            if not path.is_file():
                raise FileNotFoundError(path)
    return True


def event(root: Path, status: str, job: Job | None = None,
          details: dict[str, object] | None = None) -> None:
    """Добавить новое событие, не меняя план или прошлые свидетельства."""
    directory = root / "batch_events"
    directory.mkdir(exist_ok=True)
    record: dict[str, object] = {"utc": datetime.now(timezone.utc).isoformat(), "status": status}
    if job is not None:
        record.update({"job_id": job.job_id, "output": str(job.output)})
    if details is not None:
        record.update(details)
    write_new_json(directory / f"{time_ns()}_{status}.json", record)


def execute(root: Path, jobs: Sequence[Job]) -> int:
    """Расчёты последовательно, вывод процесса остаётся видимым в терминале."""
    for job in jobs:
        try:
            if file_sha256(job.evidence.mesh_path) != job.evidence.mesh_sha256 or file_sha256(job.evidence.stability_path) != job.evidence.stability_sha256:
                raise ValueError("Входы изменились после составления плана")
            if verify_completed(job):
                event(root, "completed_verified_skipped", job)
                print(f"Проверен и пропущен: {job.job_id}", flush=True)
                continue
            free = shutil.disk_usage(root).free
            required = job.uncompressed_upper_bound_bytes + RESERVE_BYTES
            if free < required:
                event(root, "pending_disk_space", job, {"free_bytes": free, "required_bytes": required})
                print(f"Ожидает места: {job.job_id}; доступно {free / 1024**3:.2f} GiB, требуется {required / 1024**3:.2f} GiB. Данные сохранены.", flush=True)
                return 2
            event(root, "started", job, {"free_bytes": free, "required_bytes": required})
            print(f"Запуск: {job.job_id}", flush=True)
            completed = subprocess.run(job.command(), cwd=PROJECT_ROOT, check=False)
            if completed.returncode != 0:
                event(root, "process_failed", job, {"returncode": completed.returncode})
                return completed.returncode if 0 < completed.returncode < 256 else 1
            if not verify_completed(job):
                raise RuntimeError("Процесс завершился без каталога результата")
            event(root, "completed_verified", job)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            event(root, "pending_review", job, {"error": f"{type(error).__name__}: {error}"})
            print(f"Продолжение остановлено, результаты сохранены: {error}", flush=True)
            return 1
    event(root, "all_planned_runs_completed")
    return 0


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--meshes", type=Path, default=DATA_ROOT / "meshes_v1")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="Выполнить новый план или продолжить точно совпадающий существующий")
    mode.add_argument("--dry-run", action="store_true", help="Только новый manifest; этот режим действует по умолчанию")
    parser.add_argument("--backend", choices=("ceed", "cu"), default="ceed")
    parser.add_argument("--ceed-resource", default="/gpu/cuda/gen")
    parser.add_argument("--alpha-max", type=float, default=-3.0 * 4385.0 * log(0.01) / 175.0)
    parser.add_argument("--sources", type=float, nargs="+", choices=ExperimentSpec().source_x_m, default=list(ExperimentSpec().source_x_m))
    parser.add_argument("--no-fields", action="store_true", help="Явно исключить поля и анимации; использовать только при разрешённом сокращении записи")
    parser.add_argument("--field-layout", choices=("full", "compact"), default="full",
                        help="compact: float64 Vx/Vy и две плотности энергии; суммарные энергии отдельно")
    parser.add_argument("--fields-orders", type=int, nargs="+", choices=(4, 5), default=[4, 5],
                        help="Явно ограничить запись всех узлов выбранными порядками основных запусков; у временных контролей поля всегда выключены")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    root.relative_to(DATA_ROOT.resolve())
    if root == DATA_ROOT.resolve():
        raise ValueError("Нужен отдельный каталог новой серии внутри data/dev_1_3_point_source")
    jobs, common = build_jobs(root, args.meshes.resolve(), args.sources, args.backend,
                              args.ceed_resource, args.alpha_max, not args.no_fields, args.field_layout, args.fields_orders)
    record = plan_record(jobs, common)
    manifest = root / "batch_manifest.json"
    if root.exists():
        if not args.run:
            raise FileExistsError(f"Dry-run создаёт только новый каталог: {root}")
        if not manifest.is_file() or read_json(manifest) != record:
            raise ValueError(f"План отсутствует или отличается; существующая серия не изменена: {root}")
    else:
        root.mkdir(parents=True)
        write_new_json(manifest, record)
    print(f"План: {record['main_run_count']} основных + 4 временных контроля; K={common}; полная несжатая оценка {number(record['total_uncompressed_upper_bound_bytes']) / 1024**3:.2f} GiB", flush=True)
    if args.run:
        raise SystemExit(execute(root, jobs))
    print(f"Только план, решатель не запущен: {manifest}", flush=True)


if __name__ == "__main__":
    main()
