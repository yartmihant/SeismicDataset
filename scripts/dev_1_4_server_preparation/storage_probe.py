"""Read-only disk breakdown: own subdirectories and other users' home totals."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pwd
import subprocess
import time


@dataclass(frozen=True)
class Query:
    argv: list[str]
    returncode: int
    stdout: str
    error_lines: int
    error_roots: dict[str, int]
    elapsed_seconds: float


def query(argv: list[str], timeout: int = 70) -> Query:
    """Record only aggregated failure locations, never private descendant names."""
    started = time.monotonic()
    result = subprocess.run(["timeout", str(timeout), *argv], capture_output=True, text=True, check=False)
    failures: dict[str, int] = {}
    for line in result.stderr.splitlines():
        location = "other"
        for root in ("/home/", "/var/", "/usr/", "/opt/"):
            if root in line:
                location = root + line.split(root, 1)[1].split("/", 1)[0].split("’", 1)[0]
                break
        failures[location] = failures.get(location, 0) + 1
    return Query(argv, result.returncode, result.stdout, len(result.stderr.splitlines()), failures, time.monotonic() - started)


def main() -> None:
    started = datetime.now(timezone.utc).isoformat()
    os.nice(19)
    priority = query(["ionice", "-c", "3", "-p", str(os.getpid())])
    owners: dict[str, dict[str, object]] = {}
    for path in Path("/home").iterdir():
        info = path.lstat()
        try:
            name = pwd.getpwuid(info.st_uid).pw_name
        except KeyError:
            name = str(info.st_uid)
        owners[str(path)] = {"uid": info.st_uid, "owner": name, "mode": oct(info.st_mode & 0o7777), "symlink": path.is_symlink()}
    queries = {
        "home_totals": query(["du", "-x", "-B1", "--max-depth=1", "/home"], 90),
        "antonov_breakdown": query(["du", "-x", "-B1", "--max-depth=3", str(Path.home())], 45),
        "antonov_large_files": query(["find", str(Path.home()), "-xdev", "-type", "f", "-user", str(os.getuid()), "-size", "+100M", "-printf", "%b\t%s\t%TY-%Tm-%Td\t%p\n"], 45),
        "system_totals": query(["du", "-x", "-B1", "-s", "/var", "/usr", "/opt", "/srv", "/tmp", "/root"], 45),
        "disk_after": query(["df", "-B1", "/home"]),
        "load_after": query(["uptime"]),
    }
    print(json.dumps({"started_utc": started, "finished_utc": datetime.now(timezone.utc).isoformat(),
        "priority": asdict(priority), "home_owners": owners,
        "queries": {key: asdict(value) for key, value in queries.items()},
        "scope": "No sudo; metadata only; other users only home totals; permission failures imply lower bounds; no removal",
        "du_hardlink_policy": "Each du invocation counts an inode once; home-to-home hard links attributed to first encountered directory"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
