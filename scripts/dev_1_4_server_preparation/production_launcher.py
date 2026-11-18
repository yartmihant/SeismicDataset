"""Supervise only this project's two GPU workers; stop the pair on any failure."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import FrameType

from scripts.project_paths import PROJECT_ROOT, SOURCE


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_status(path: Path, status: dict[str, object]) -> None:
    temporary = path.with_suffix(".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(status, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def stop_children(children: list[subprocess.Popen[bytes]], grace_seconds: float) -> None:
    """Signal only process groups created by this supervisor, never other jobs."""
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.monotonic() + grace_seconds
    while any(process.poll() is None for process in children) and time.monotonic() < deadline:
        time.sleep(.1)
    for process in children:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for process in children:
        process.wait()


def supervise(root: Path, commands: list[list[str]], grace_seconds: float = 10.) -> int:
    """Low-level supervisor; commands are explicit Python workers in the public CLI."""
    if len(commands) != 2 or grace_seconds < 0:
        raise ValueError("Exactly two workers and a nonnegative termination grace are required")
    root = root.resolve()
    root.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "launcher.json"
    if status_path.exists() or status_path.with_suffix(".partial").exists():
        raise FileExistsError("Launcher record already exists; no automatic overwrite or restart")
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    children: list[subprocess.Popen[bytes]] = []
    entries: list[dict[str, object]] = []
    status: dict[str, object] = {"schema": "chapter14.launcher.v1", "status": "starting", "launcher_pid": os.getpid(),
        "started_utc": utc(), "root": str(root), "workers": entries,
        "failure_policy": "stop both; keep outputs, temporary work and SQLite reservations; no retry or release"}
    write_status(status_path, status)
    stop_signal = 0

    def receive_signal(signum: int, frame: FrameType | None) -> None:
        nonlocal stop_signal
        stop_signal = signum

    previous = {sig: signal.signal(sig, receive_signal) for sig in (signal.SIGINT, signal.SIGTERM)}
    exit_code = 1
    try:
        with ExitStack() as streams:
            for gpu, command in enumerate(commands):
                log = logs / f"gpu{gpu}.log"
                stream = streams.enter_context(log.open("xb"))
                process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                children.append(process)
                entries.append({"gpu": gpu, "pid": process.pid, "command": command, "log": str(log), "returncode": None})
                write_status(status_path, status)
            status["status"] = "running"
            write_status(status_path, status)
            while True:
                codes = [process.poll() for process in children]
                if stop_signal:
                    status.update(status="interrupted", signal=stop_signal)
                    exit_code = 128 + stop_signal
                    break
                if any(code is not None and code != 0 for code in codes):
                    status.update(status="failed", reason="worker exited with nonzero status or signal")
                    break
                if all(code is not None for code in codes):
                    status["status"] = "completed"
                    exit_code = 0
                    break
                time.sleep(.25)
            if exit_code:
                stop_children(children, grace_seconds)
    except BaseException as error:
        exit_code = 1
        status.update(status="failed", reason=f"{type(error).__name__}: {error}")
        stop_children(children, grace_seconds)
        print(json.dumps({"event": "launcher_failure", "reason": status["reason"]}), file=sys.stderr, flush=True)
    finally:
        for entry, process in zip(entries, children, strict=True):
            entry["returncode"] = process.poll()
        status["finished_utc"] = utc()
        status["exit_code"] = exit_code
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        write_status(status_path, status)
    print(json.dumps({"event": "launcher_finished", "status": status["status"], "exit_code": exit_code,
                      "record": str(status_path)}), flush=True)
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--max-models", type=int, help="Optional number per GPU, e.g. one for the pilot")
    args = parser.parse_args()
    root: Path = args.root.resolve()
    if args.max_models is not None and args.max_models < 1:
        raise ValueError("max-models must be positive")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = "2" if variable == "OMP_NUM_THREADS" else "1"
    commands = [[sys.executable, "-m", "scripts.dev_1_4_server_preparation.production_worker", "--gpu", str(gpu),
                 "--root", str(root), "--source", str(args.source.resolve())] for gpu in (0, 1)]
    if args.max_models is not None:
        for command in commands:
            command.extend(("--max-models", str(args.max_models)))
    raise SystemExit(supervise(root, commands))


if __name__ == "__main__":
    main()
