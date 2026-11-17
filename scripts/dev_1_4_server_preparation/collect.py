"""Collect one immutable server snapshot from this workstation over SSH."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from scripts.project_paths import PROJECT_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination: Path = args.output.resolve()
    destination.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("probe.py").read_bytes()
    ssh = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=12", "-o", "ConnectionAttempts=1",
           "-o", "StrictHostKeyChecking=yes", "-o", "UpdateHostKeys=no", "antonov@188.44.53.153", "python3 -B -"]
    result = subprocess.run(ssh, input=source, capture_output=True, timeout=200, check=False)
    (destination / "ssh_stderr.txt").write_bytes(result.stderr)
    if result.returncode:
        raise RuntimeError(f"SSH probe failed: {result.returncode}; see ssh_stderr.txt")
    json.loads(result.stdout)
    (destination / "snapshot.json").write_bytes(result.stdout)
    (destination / "probe.py.txt").write_bytes(source)
    (destination / "manifest.json").write_text(json.dumps({
        "collected_utc": datetime.now(timezone.utc).isoformat(), "ssh_command": ssh,
        "probe_sha256": hashlib.sha256(source).hexdigest(),
        "snapshot_sha256": hashlib.sha256(result.stdout).hexdigest(),
        "remote_files_created": False, "origin": "current workstation"}, indent=2) + "\n")
    print(destination / "snapshot.json")


if __name__ == "__main__":
    main()
