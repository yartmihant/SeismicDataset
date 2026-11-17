"""Чередующийся план EFVB/ECVB и общий транзакционный дисковый бюджет двух GPU.

Модуль не запускает решатель, SSH, передачу или удаление данных. Все пишущие
процессы одной серии должны использовать одну SQLite-базу на том же разделе.
Резерв защищает от собственных согласованных записей, но не от чужих процессов.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
from typing import Callable, Iterator, Literal

from scripts.project_paths import PROJECT_ROOT

Dataset = Literal["ECVB", "EFVB"]
GIB = 1024**3
MODELS_PER_GROUP = 30000
MODELS_PER_CHUNK = 5000
SURFACE_BYTES_PER_MODEL = 11_294_040
INPUT_BYTES_PER_MODEL = 78_400
MODEL_FILE_ALLOWANCE_BYTES = 256 * 1024


@dataclass(frozen=True)
class Chunk:
    number: int
    dataset: Dataset
    first_id: int
    last_id: int

    @property
    def name(self) -> str:
        return f"chunk_{self.number:02d}_{self.dataset}_{self.first_id:05d}_{self.last_id:05d}"

    @property
    def models(self) -> int:
        return self.last_id - self.first_id + 1


def build_chunks() -> tuple[Chunk, ...]:
    """Каждая группа имеет собственные реальные ID 00000..29999 из главы 1.1."""
    datasets: tuple[Dataset, Dataset] = ("EFVB", "ECVB")
    return tuple(Chunk(2 * block + position + 1, dataset,
                       block * MODELS_PER_CHUNK, (block + 1) * MODELS_PER_CHUNK - 1)
                 for block in range(MODELS_PER_GROUP // MODELS_PER_CHUNK)
                 for position, dataset in enumerate(datasets))


@dataclass(frozen=True)
class GuardConfig:
    reserve_bytes: int = 20 * GIB
    working_floor_bytes: int = 20 * GIB
    max_in_flight: int = 2
    execution_chunk_limit: int = 1
    model_budget_bytes: int = SURFACE_BYTES_PER_MODEL + INPUT_BYTES_PER_MODEL + MODEL_FILE_ALLOWANCE_BYTES

    def validate(self) -> None:
        if self.reserve_bytes < 20 * GIB or self.working_floor_bytes < 0:
            raise ValueError("Нужны резерв >=20 ГиБ и неотрицательный рабочий бюджет")
        if self.max_in_flight != 2 or not 1 <= self.execution_chunk_limit <= 12:
            raise ValueError("План рассчитан на две GPU и 1..12 явно включённых чанков")
        if self.model_budget_bytes < SURFACE_BYTES_PER_MODEL:
            raise ValueError("Бюджет модели меньше несжатых поверхностных массивов")


@dataclass(frozen=True)
class JobReservation:
    job_id: str
    chunk_number: int
    dataset: Dataset
    model_id: int
    gpu: int
    budget_bytes: int


class BudgetUnavailable(RuntimeError):
    """Недостаточно места или очередь заблокирована; новые записи не разрешены."""


class DiskGuard:
    """Общий журнал резервирования с SQLite BEGIN IMMEDIATE для всех работников.

    Полный бюджет модели удерживается до подтверждённого завершения. Уже
    записанная часть дополнительно уменьшает Available: временное двойное
    резервирование намеренно консервативно. Failed-резервы не освобождаются
    автоматически. После аварии сначала проверяют файлы и состояние журнала.
    """

    def __init__(self, database: Path, filesystem_path: Path,
                 config: GuardConfig = GuardConfig(),
                 available_bytes: Callable[[], int] | None = None) -> None:
        config.validate()
        self.database = database.resolve()
        self.filesystem_path = filesystem_path.resolve(strict=True)
        self.config = config
        self.database.parent.mkdir(parents=True, exist_ok=True)
        if self.database.parent.stat().st_dev != self.filesystem_path.stat().st_dev:
            raise ValueError("Журнал и расчётные выходы должны находиться на одном разделе")
        self._available_bytes = available_bytes or self._filesystem_available
        with self._transaction() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS config (singleton INTEGER PRIMARY KEY CHECK(singleton=1), document TEXT NOT NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS jobs (job_id TEXT PRIMARY KEY, chunk_number INTEGER NOT NULL, dataset TEXT NOT NULL, model_id INTEGER NOT NULL, gpu INTEGER NOT NULL, budget_bytes INTEGER NOT NULL, status TEXT NOT NULL, output_bytes INTEGER, reason TEXT, UNIQUE(dataset, model_id))")
            document = json.dumps({"schema_version": 1, "filesystem_path": str(self.filesystem_path),
                                   "config": asdict(config)}, sort_keys=True)
            existing = connection.execute("SELECT document FROM config WHERE singleton=1").fetchone()
            if existing is None:
                connection.execute("INSERT INTO config VALUES (1, ?)", (document,))
            elif existing[0] != document:
                raise ValueError("Конфигурация существующей очереди отличается; автоматическое расширение/сброс запрещены")

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.database, timeout=30.0, isolation_level=None)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _filesystem_available(self) -> int:
        info = os.statvfs(self.filesystem_path)
        return info.f_bavail * info.f_frsize

    def _check_budget(self, connection: sqlite3.Connection, new_bytes: int = 0) -> None:
        outstanding = int(connection.execute(
            "SELECT COALESCE(SUM(budget_bytes), 0) FROM jobs WHERE status!='completed'").fetchone()[0])
        required = self.config.reserve_bytes + self.config.working_floor_bytes + outstanding + new_bytes
        available = self._available_bytes()
        if available < required:
            raise BudgetUnavailable(f"Available={available}, требуется {required} байт с общими резервами")

    def reserve_next(self, gpu: int) -> JobReservation | None:
        """Атомарно выдать полную модель с пятью источниками, соблюдая барьер чанка."""
        if gpu not in (0, 1):
            raise ValueError("GPU должна быть 0 или 1")
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE status='failed' LIMIT 1").fetchone():
                raise BudgetUnavailable("Есть failed-модель; резерв сохранён, требуется явный разбор")
            active = connection.execute("SELECT gpu FROM jobs WHERE status='reserved'").fetchall()
            if len(active) >= self.config.max_in_flight or any(int(row[0]) == gpu for row in active):
                raise BudgetUnavailable("Все слоты заняты либо GPU уже имеет активную модель")
            for chunk in build_chunks()[:self.config.execution_chunk_limit]:
                completed = int(connection.execute("SELECT COUNT(*) FROM jobs WHERE chunk_number=? AND status='completed'", (chunk.number,)).fetchone()[0])
                if completed == chunk.models:
                    continue
                greatest = connection.execute("SELECT MAX(model_id) FROM jobs WHERE chunk_number=?", (chunk.number,)).fetchone()[0]
                model_id = chunk.first_id if greatest is None else int(greatest) + 1
                if model_id > chunk.last_id:
                    return None  # Last jobs of this chunk are still running; no next-group leap.
                self._check_budget(connection, self.config.model_budget_bytes)
                job_id = f"{chunk.dataset}/{model_id:05d}"
                connection.execute("INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, 'reserved', NULL, NULL)",
                    (job_id, chunk.number, chunk.dataset, model_id, gpu, self.config.model_budget_bytes))
                return JobReservation(job_id, chunk.number, chunk.dataset, model_id, gpu, self.config.model_budget_bytes)
            return None

    def check_before_write(self, job_id: str, expected_total_bytes: int) -> None:
        """Повторно проверить место и полный размер своих файлов с новой записью.

        Работник передаёт общий ожидаемый размер своих сохраняемых входов/выходов
        и временных выходных копий после записи, а не только нового блока. Mesh
        и сборка относятся к отдельному working_floor_bytes. Работник
        обязан вызвать метод перед каждой записью и не писать после исключения.
        """
        if expected_total_bytes < 0:
            raise ValueError("Отрицательный размер записи")
        with self._transaction() as connection:
            row = connection.execute("SELECT budget_bytes, status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise ValueError("Нет активного резерва этой модели")
            if expected_total_bytes > int(row[0]):
                raise BudgetUnavailable("Размер модели превышает резерв; изменение бюджета требует отдельного решения")
            self._check_budget(connection)

    def complete(self, job_id: str, verified_output_bytes: int) -> None:
        """Освободить резерв только после внешней проверки всех пяти выходов.

        Метод не проверяет содержимое NPZ вместо решателя/валидатора. Аргумент
        означает, что вызывающий код уже проверил пять источников, формы,
        время, конечность значений и сохранность файлов. Файлы не удаляются.
        """
        with self._transaction() as connection:
            row = connection.execute("SELECT budget_bytes, status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None or row[1] != "reserved":
                raise ValueError("Завершить можно только активную модель")
            if not 0 < verified_output_bytes <= int(row[0]):
                raise ValueError("Недопустимый подтверждённый объём выходов")
            connection.execute("UPDATE jobs SET status='completed', output_bytes=? WHERE job_id=?", (verified_output_bytes, job_id))

    def fail(self, job_id: str, reason: str) -> None:
        """Сохранить резерв аварийной модели; ничего автоматически не удалять."""
        with self._transaction() as connection:
            updated = connection.execute("UPDATE jobs SET status='failed', reason=? WHERE job_id=? AND status='reserved'", (reason, job_id))
            if updated.rowcount != 1:
                raise ValueError("Нет активной модели для отметки failed")

    def status(self) -> dict[str, object]:
        with self._transaction() as connection:
            rows = connection.execute("SELECT chunk_number, status, COUNT(*) FROM jobs GROUP BY chunk_number, status ORDER BY chunk_number, status").fetchall()
            outstanding = int(connection.execute("SELECT COALESCE(SUM(budget_bytes), 0) FROM jobs WHERE status!='completed'").fetchone()[0])
            return {"config": asdict(self.config), "available_bytes": self._available_bytes(),
                    "outstanding_reservation_bytes": outstanding,
                    "progress": [{"chunk_number": int(row[0]), "status": str(row[1]), "models": int(row[2])} for row in rows]}


def plan_document() -> dict[str, object]:
    chunks = build_chunks()
    return {
        "schema_version": 1, "status": "planned_not_executed", "geometry": "curved", "order": 4,
        "groups": ["EFVB", "ECVB"], "models_per_group": MODELS_PER_GROUP, "models_per_chunk": MODELS_PER_CHUNK,
        "initial_execution_chunk_limit": 1, "sources_x_m": [0, 175, 350, 525, 700], "source_y_m": 0,
        "receivers_x_m": list(range(5, 700, 10)), "receiver_y_m": 0, "components": ["Vx", "Vy"],
        "dtype": "float64", "time_samples": 2001, "duration_s": 2.0,
        "surface_bytes_per_model": SURFACE_BYTES_PER_MODEL, "input_bytes_per_model": INPUT_BYTES_PER_MODEL,
        "guard": asdict(GuardConfig()), "manual_pack_transfer_cleanup": True, "automatic_removal": False,
        "chunks": [{**asdict(chunk), "name": chunk.name, "models": chunk.models,
                    "sources": chunk.models * 5, "surface_bytes": chunk.models * SURFACE_BYTES_PER_MODEL,
                    "input_bytes": chunk.models * INPUT_BYTES_PER_MODEL} for chunk in chunks],
        "source_schema": {"source_file_pattern": "src/data_{ECVB|EFVB}_out/{vp|vs}_{id:05d}.txt",
                          "binary_shards": "ids:int64; vp,vs:float64[models,70,70]",
                          "id_validation_implementation": "scripts/dev_1_1_preprocessing/prepare_data.py:source_pairs"},
        "limits": ["Every writer must use the same DiskGuard database", "Concurrent external users are outside the guard",
                   "No automatic recovery of failed or stale reservations", "No solver or input/output implementation in this module",
                   "Chunk completion requires external output validation", "Initial execution stops after chunk 1; ECVB waits for speed review",
                   "20 GiB working floor is a budget assumption, not a measured pilot peak"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination: Path = args.output.resolve()
    destination.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    destination.mkdir(parents=True, exist_ok=False)
    document = plan_document()
    document["created_utc"] = datetime.now(timezone.utc).isoformat()
    source = Path(__file__)
    document["planner_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    (destination / "manifest.json").write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination / "manifest.json")


if __name__ == "__main__":
    main()
