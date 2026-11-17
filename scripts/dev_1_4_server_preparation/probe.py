"""Bounded read-only server inventory; streamed over SSH, never installed remotely."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import time


@dataclass(frozen=True)
class CommandResult:
    argv: list[str]
    returncode: int
    stdout: str
    stderr: str


def command(argv: list[str], timeout: int = 20) -> CommandResult:
    """Capture a bounded query without shell expansion or privilege elevation."""
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return CommandResult(argv, result.returncode, result.stdout, result.stderr)
    except FileNotFoundError:
        return CommandResult(argv, 127, "", "Command unavailable")
    except subprocess.TimeoutExpired:
        return CommandResult(argv, 124, "", "Query timed out; output not retained")


def home_usage(budget_seconds: float = 90.0) -> dict[str, object]:
    """Count allocated inode blocks once, on the home filesystem, without following links."""
    root = Path.home()
    device = root.stat().st_dev
    uid = os.getuid()
    seen: set[tuple[int, int]] = set()
    totals: dict[str, dict[str, int]] = {}
    errors: list[str] = []
    skipped_mounts: list[str] = []
    pending = [root]
    start = time.monotonic()
    while pending and time.monotonic() - start < budget_seconds:
        path = pending.pop()
        try:
            info = path.lstat()
            if info.st_dev != device:
                skipped_mounts.append(str(path))
                continue
            key = (info.st_dev, info.st_ino)
            if key in seen:
                continue
            seen.add(key)
            relative = path.relative_to(root)
            group = relative.parts[0] if relative.parts else "."
            record = totals.setdefault(group, {"allocated_bytes": 0, "antonov_allocated_bytes": 0, "inodes": 0})
            allocated = info.st_blocks * 512
            record["allocated_bytes"] += allocated
            record["antonov_allocated_bytes"] += allocated if info.st_uid == uid else 0
            record["inodes"] += 1
            if path.is_dir() and not path.is_symlink():
                with os.scandir(path) as entries:
                    pending.extend(Path(entry.path) for entry in entries)
        except OSError as exc:
            errors.append(f"{path}: {exc.strerror}")
    return {"root": str(root), "uid": uid, "complete": not pending and not errors and not skipped_mounts,
            "elapsed_seconds": time.monotonic() - start, "unvisited_paths": len(pending),
            "errors": errors, "skipped_mounts": skipped_mounts,
            "allocated_bytes": sum(value["allocated_bytes"] for value in totals.values()),
            "antonov_allocated_bytes": sum(value["antonov_allocated_bytes"] for value in totals.values()),
            "unique_inodes": len(seen), "top_level": totals,
            "scope": "Home only; hard links counted once; symlinks not followed; no other users' homes scanned"}


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    queries = {
        "identity": ["id"], "hostname": ["hostname"], "os": ["cat", "/etc/os-release"],
        "kernel": ["uname", "-a"], "cpu": ["lscpu", "--json"],
        "memory": ["cat", "/proc/meminfo"], "uptime": ["uptime"],
        "filesystems": ["df", "-B1", "-T", "-x", "tmpfs", "-x", "devtmpfs", "-x", "squashfs"],
        "inodes": ["df", "-i", str(Path.home())], "mounts": ["findmnt", "--json", "--real"],
        "disks": ["lsblk", "--json", "--bytes", "-o", "NAME,TYPE,SIZE,MODEL,ROTA,TRAN,MOUNTPOINT"],
        "gpu": ["nvidia-smi", "-q", "-x"], "gpu_topology": ["nvidia-smi", "topo", "-m"],
        "gpu_samples": ["nvidia-smi", "--query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit", "--format=csv", "-l", "2"],
        "vmstat": ["vmstat", "1", "4"], "quota": ["quota", "-s"],
        "user_processes": ["ps", "-u", str(os.getuid()), "-o", "pid,comm,pcpu,pmem,etime"],
        "cuda_compiler": ["nvcc", "--version"], "python": ["python3", "--version"],
        "compiler": ["gcc", "--version"], "cmake": ["cmake", "--version"],
        "dmi": ["sh", "-c", "for f in sys_vendor product_name product_version board_vendor board_name; do printf '%s: ' \"$f\"; cat /sys/class/dmi/id/\"$f\"; done"],
        "locations": ["ls", "-ld", "/home/antonov", "/media", "/mnt", "/srv", "/opt", "/data", "/scratch", "/work"],
        "shared_location_entries": ["find", "/media", "/mnt", "/srv", "/opt", "-maxdepth", "2", "-type", "d", "-user", str(os.getuid()), "-print"],
    }
    # Finite samples, no continuous monitor or GPU workload.
    del queries["gpu_samples"]
    records = {name: asdict(command(argv)) for name, argv in queries.items()}
    gpu_samples = []
    for index in range(3):
        gpu_samples.append(asdict(command(["nvidia-smi", "--query-gpu=timestamp,index,name,memory.total,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw,power.limit", "--format=csv"])))
        if index < 2:
            time.sleep(2)
    os.nice(19)
    io_priority = command(["ionice", "-c", "3", "-p", str(os.getpid())])
    usage = home_usage()
    print(json.dumps({"schema_version": 1, "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(), "commands": records,
        "gpu_samples": gpu_samples, "home_usage": usage, "io_priority": asdict(io_priority),
        "tools": {name: shutil.which(name) for name in ("squeue", "sinfo", "qstat", "quota", "docker", "uv", "nvcc", "nvidia-smi")},
        "policy": "Read-only; no sudo, installs, benchmarks, solver, containers or other users' home scans"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
