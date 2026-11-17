"""Read-only queue/launcher status; no solver, array scans or queue mutations."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import sqlite3

from scripts.reporting import mapping, number, read_json


def optional_json(path: Path) -> dict[str, object] | None:
    """An active worker can be writing its small metadata file during this read."""
    try:
        return read_json(path)
    except FileNotFoundError:
        return None
    except (ValueError, OSError) as error:
        return {"read_error": f"{type(error).__name__}: {error}", "path": str(path)}


def last_log_event(path: Path) -> dict[str, object] | None:
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 32768))
            lines = stream.read(32768).decode("utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        try:
            return mapping(json.loads(line))
        except ValueError:
            continue
    return None


def active_details(root: Path, job: dict[str, object]) -> dict[str, object]:
    chunk, model_id, gpu = (int(number(job[key])) for key in ("chunk_number", "model_id", "gpu"))
    dataset = str(job["dataset"])
    if dataset not in ("ECVB", "EFVB") or gpu not in (0, 1) or not 1 <= chunk <= 12 or not 0 <= model_id < 30000:
        raise ValueError("Invalid active job identity in the queue")
    first = ((chunk - 1) // 2) * 5000
    output = root / "chunks" / f"chunk_{chunk:02d}_{dataset}_{first:05d}_{first+4999:05d}" / "models" / f"{model_id:05d}"
    sources: list[dict[str, object]] = []
    for source in (0, 175, 350, 525, 700):
        directory = output / f"x{source:03d}"
        record: dict[str, object] = {"source_x_m": source, "status": "not_started"}
        for filename in ("metadata.failed.json", "metadata.json", "metadata.started.json"):
            metadata = optional_json(directory / filename)
            if metadata is not None:
                if "read_error" in metadata:
                    record["status"] = "metadata_unreadable"
                record.update({key: metadata[key] for key in ("status", "wall_seconds", "recorded_counts", "error", "read_error") if key in metadata})
                record["metadata_file"] = str(directory / filename)
                break
        sources.append(record)
    return {**job, "output": str(output), "geometry_record_exists": (output / "geometry.json").is_file(),
            "stability_record_exists": (output / "stability.json").is_file(), "sources": sources,
            "last_log_event": last_log_event(root / "logs" / f"gpu{gpu}.log")}


def parse_utc(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Expected an ISO timestamp")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("Launcher timestamp must specify a timezone")
    return result.astimezone(timezone.utc)


def inspect_status(root: Path, now: datetime | None = None) -> dict[str, object]:
    root = root.resolve(strict=True)
    timestamp = now or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("Status time must specify a timezone")
    database = root / "queue.sqlite"
    if not database.is_file():
        raise FileNotFoundError(database)
    # URI mode=ro cannot initialize or modify a queue. Keep the read transaction short.
    connection = sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=10)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        config_row = connection.execute("SELECT document FROM config WHERE singleton=1").fetchone()
        if config_row is None:
            raise ValueError("Queue has no configuration")
        document = mapping(json.loads(str(config_row["document"])))
        config = mapping(document["config"])
        progress_rows = connection.execute("SELECT chunk_number, status, COUNT(*) AS models FROM jobs GROUP BY chunk_number, status").fetchall()
        unsettled = connection.execute("SELECT job_id, chunk_number, dataset, model_id, gpu, budget_bytes, status, reason FROM jobs WHERE status!='completed' ORDER BY chunk_number, model_id").fetchall()
        totals = connection.execute("SELECT COUNT(*) AS models, COALESCE(SUM(output_bytes),0) AS bytes FROM jobs WHERE status='completed'").fetchone()
    finally:
        connection.close()
    limit = int(number(config["execution_chunk_limit"]))
    if not 1 <= limit <= 12:
        raise ValueError("Unexpected authorized chunk limit")
    counts = {(int(row["chunk_number"]), str(row["status"])): int(row["models"]) for row in progress_rows}
    chunks = [{"chunk_number": chunk, "dataset": "EFVB" if chunk % 2 else "ECVB", "authorized": chunk <= limit,
               "first_id": ((chunk - 1) // 2) * 5000, "last_id": ((chunk - 1) // 2) * 5000 + 4999,
               "target_models": 5000, **{status: counts.get((chunk, status), 0) for status in ("completed", "reserved", "failed")}}
              for chunk in range(1, 13)]
    jobs: list[dict[str, object]] = [{"job_id": str(row["job_id"]), "chunk_number": int(row["chunk_number"]),
        "dataset": str(row["dataset"]), "model_id": int(row["model_id"]), "gpu": int(row["gpu"]),
        "budget_bytes": int(row["budget_bytes"]), "status": str(row["status"]),
        "reason": None if row["reason"] is None else str(row["reason"])} for row in unsettled]
    active = [job for job in jobs if job["status"] == "reserved"]
    if len(active) > 2:
        raise ValueError("More than two active jobs; refusing an unbounded metadata scan")
    completed = int(totals["models"])
    launcher = optional_json(root / "launcher.json")
    start = parse_utc(launcher.get("started_utc")) if launcher else None
    finish = parse_utc(launcher.get("finished_utc")) if launcher else None
    elapsed = max(0., ((finish or timestamp) - start).total_seconds()) if start else None
    rate = completed * 3600. / elapsed if completed and elapsed and elapsed > 0 else None
    remaining = max(0, 5000 - counts.get((1, "completed"), 0))
    eta_seconds = remaining * 3600. / rate if rate else None
    disk = shutil.disk_usage(root)
    outstanding = sum(int(number(job["budget_bytes"])) for job in jobs)
    threshold = int(number(config["reserve_bytes"]) + number(config["working_floor_bytes"])) + outstanding
    return {"schema": "chapter14.status.v1", "snapshot_utc": timestamp.isoformat(), "root": str(root),
            "read_only": True, "database_mode": "ro", "config": config, "execution_chunk_limit": limit,
            "chunk_progress": chunks, "completed_models": completed, "verified_output_bytes": int(totals["bytes"]),
            "active": [active_details(root, job) for job in active], "failed": [job for job in jobs if job["status"] == "failed"],
            "outstanding_reservation_bytes": outstanding, "launcher": launcher,
            "launcher_started_utc": start.isoformat() if start else None, "elapsed_seconds": elapsed,
            "completed_models_per_hour": rate, "first_5000_remaining_models": remaining,
            "rough_first_5000_remaining_seconds": eta_seconds,
            "rough_first_5000_eta_utc_if_continued": (timestamp + timedelta(seconds=eta_seconds)).isoformat() if eta_seconds is not None else None,
            "eta_caveat": "Linear extrapolation from launcher wall time; a small or heterogeneous sample is unreliable. Includes startup and idle time; after a stop assumes explicit continuation. No ETA before the first completed model.",
            "disk": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free,
                     "guard_required_free_bytes": threshold, "guard_threshold_met_now": disk.free >= threshold},
            "snapshot_caveat": "Queue rows form one read snapshot; launcher, metadata, logs and disk can change separately. Stored completion is not a fresh revalidation of arrays; no process-liveness claim."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect_status(args.root), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
