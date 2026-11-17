"""Адресные CPU-проверки порядка чанков и общего дискового резерва."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import time
from typing import Callable

from scripts.dev_1_4_server_preparation.chunk_plan import (
    BudgetUnavailable, DiskGuard, GIB, GuardConfig, SURFACE_BYTES_PER_MODEL, build_chunks, plan_document,
)
from scripts.project_paths import PROJECT_ROOT


def expect_error(action: Callable[[], object], kind: type[Exception]) -> None:
    try:
        action()
    except kind:
        return
    raise AssertionError(f"Expected {kind.__name__}")


def race_worker(database: str, root: str, gpu: int, available: int, start: float) -> str:
    """Два отдельных процесса читают одно Available и резервируют через одну БД."""
    guard = DiskGuard(Path(database), Path(root), available_bytes=lambda: available)
    while time.monotonic() < start:
        time.sleep(0.005)
    try:
        job = guard.reserve_next(gpu)
        return "reserved:" + job.job_id if job else "none"
    except BudgetUnavailable:
        return "blocked"


def run_checks(root: Path) -> list[str]:
    passed: list[str] = []
    chunks = build_chunks()
    assert len(chunks) == 12 and sum(c.models for c in chunks) == 60000
    for dataset in ("ECVB", "EFVB"):
        ids = [model for c in chunks if c.dataset == dataset for model in range(c.first_id, c.last_id + 1)]
        assert ids == list(range(30000))
    assert [c.dataset for c in chunks] == ["EFVB", "ECVB"] * 6
    assert chunks[2].first_id == 5000 and chunks[2].last_id == 9999
    doc = plan_document()
    assert doc["receivers_x_m"] == list(range(5, 700, 10))
    assert 5000 * SURFACE_BYTES_PER_MODEL == 56_470_200_000
    passed.append("12 chunks alternate, cover 60000 distinct (group,id), exact 5000-model byte accounting")

    config = GuardConfig()
    floor = config.reserve_bytes + config.working_floor_bytes
    available = [floor + config.model_budget_bytes - 1]
    guard = DiskGuard(root / "boundary.sqlite", root, available_bytes=lambda: available[0])
    expect_error(lambda: guard.reserve_next(0), BudgetUnavailable)
    available[0] += 1
    first = guard.reserve_next(0)
    assert first is not None and first.job_id == "EFVB/00000"
    expect_error(lambda: guard.reserve_next(1), BudgetUnavailable)
    guard.check_before_write(first.job_id, config.model_budget_bytes)
    expect_error(lambda: guard.check_before_write(first.job_id, config.model_budget_bytes + 1), BudgetUnavailable)
    available[0] -= 1  # An external writer consumed space after admission.
    expect_error(lambda: guard.check_before_write(first.job_id, 1), BudgetUnavailable)
    available[0] += 1
    guard.complete(first.job_id, SURFACE_BYTES_PER_MODEL)
    expect_error(lambda: guard.complete(first.job_id, SURFACE_BYTES_PER_MODEL), ValueError)
    next_job = guard.reserve_next(1)
    assert next_job is not None and next_job.job_id == "EFVB/00001"
    guard.fail(next_job.job_id, "test failure")
    expect_error(lambda: guard.reserve_next(0), BudgetUnavailable)
    assert guard.status()["outstanding_reservation_bytes"] == config.model_budget_bytes
    expect_error(lambda: DiskGuard(root / "boundary.sqlite", root, GuardConfig(execution_chunk_limit=12)), ValueError)
    passed.append("boundary equality, one-byte shortage, recheck after external write, duplicate completion and failed reserve")

    race_db = root / "race.sqlite"
    DiskGuard(race_db, root, available_bytes=lambda: floor + config.model_budget_bytes)
    with ProcessPoolExecutor(max_workers=2) as pool:
        start = time.monotonic() + 0.5
        results = list(pool.map(race_worker, [str(race_db)] * 2, [str(root)] * 2,
                               [0, 1], [floor + config.model_budget_bytes] * 2, [start] * 2))
    assert sorted(results) == ["blocked", "reserved:EFVB/00000"], results
    passed.append("two processes with identical free-space observation cannot reserve the same remaining capacity twice")

    barrier_db = root / "barrier.sqlite"
    barrier = DiskGuard(barrier_db, root, available_bytes=lambda: 1000 * GIB)
    # Insert externally validated completion fixtures; no GPU work or bulk arrays.
    with sqlite3.connect(barrier_db) as connection:
        connection.executemany("INSERT INTO jobs VALUES (?, 1, 'EFVB', ?, 0, ?, 'completed', ?, NULL)",
            [(f"EFVB/{i:05d}", i, config.model_budget_bytes, SURFACE_BYTES_PER_MODEL) for i in range(4998)])
    a = barrier.reserve_next(0)
    b = barrier.reserve_next(1)
    assert a is not None and b is not None
    assert (a.model_id, b.model_id) == (4998, 4999)
    barrier.complete(a.job_id, SURFACE_BYTES_PER_MODEL)
    assert barrier.reserve_next(0) is None
    barrier.complete(b.job_id, SURFACE_BYTES_PER_MODEL)
    assert barrier.reserve_next(0) is None  # No implicit start of ECVB chunk 2.
    extended_db = root / "explicit_two_chunks.sqlite"
    extended = DiskGuard(extended_db, root, GuardConfig(execution_chunk_limit=2), available_bytes=lambda: 1000 * GIB)
    with sqlite3.connect(extended_db) as connection:
        connection.executemany("INSERT INTO jobs VALUES (?, 1, 'EFVB', ?, 0, ?, 'completed', ?, NULL)",
            [(f"EFVB/{i:05d}", i, config.model_budget_bytes, SURFACE_BYTES_PER_MODEL) for i in range(5000)])
    c = extended.reserve_next(0)
    assert c is not None and c.chunk_number == 2 and c.dataset == "ECVB" and c.model_id == 0
    passed.append("chunk barrier waits for last model; default stops after EFVB chunk 1; explicit two-chunk plan transitions to ECVB")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination: Path = args.output.resolve()
    destination.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tmp_dev_1_4_chunk_guard_", dir=PROJECT_ROOT) as temporary:
        passed = run_checks(Path(temporary))
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(), "checks": passed,
              "scope": "CPU and SQLite concurrency only; no SSH, GPU, solver or dataset run",
              "sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("chunk_plan.py", "check_chunk_plan.py")}}
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
