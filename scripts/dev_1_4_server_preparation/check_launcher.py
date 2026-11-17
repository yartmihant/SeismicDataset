"""Real short subprocess checks for the supervisor; no geometry/native/GPU."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from scripts.dev_1_4_server_preparation.launcher import supervise
from scripts.project_paths import PROJECT_ROOT
from scripts.reporting import mapping, read_json, sequence


def command(program: str) -> list[str]:
    return [sys.executable, "-c", program]


def checks(root: Path) -> list[str]:
    passed: list[str] = []
    completed = root / "completed"
    assert supervise(completed, [command("print('gpu0 mock done')"), command("print('gpu1 mock done')")]) == 0
    assert read_json(completed / "launcher.json")["status"] == "completed"
    try:
        supervise(completed, [command("pass"), command("pass")])
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing launcher evidence was overwritten")
    passed.append("both successful workers complete; existing launcher records cannot be overwritten")
    failed = root / "failed"
    commands = [command("import time,sys; time.sleep(.3); sys.exit(7)"),
                command("import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)")]
    assert supervise(failed, commands, .1) == 1
    status = read_json(failed / "launcher.json")
    codes = [mapping(entry)["returncode"] for entry in sequence(status["workers"])]
    assert status["status"] == "failed" and codes == [7, -signal.SIGKILL]
    passed.append("nonzero exit stops partner; SIGTERM-resistant partner is killed after bounded grace")
    killed = root / "killed"
    assert supervise(killed, [command("import os,signal; os.kill(os.getpid(),signal.SIGKILL)"), command("import time; time.sleep(60)")], .1) == 1
    status = read_json(killed / "launcher.json")
    assert mapping(sequence(status["workers"])[0])["returncode"] == -signal.SIGKILL
    assert mapping(sequence(status["workers"])[1])["returncode"] is not None
    passed.append("worker SIGKILL is detected and its live partner is stopped")
    interrupted = root / "interrupted"
    sleepers = [command("import time; time.sleep(60)"), command("import time; time.sleep(60)")]
    program = ("from pathlib import Path; from scripts.dev_1_4_server_preparation.launcher import supervise; "
               f"raise SystemExit(supervise(Path({str(interrupted)!r}), {sleepers!r}, .1))")
    with (root / "signal_supervisor.log").open("xb") as output:
        process = subprocess.Popen(command(program), cwd=PROJECT_ROOT, stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                status_path = interrupted / "launcher.json"
                if status_path.is_file() and read_json(status_path)["status"] == "running":
                    break
                assert process.poll() is None
                time.sleep(.05)
            else:
                raise AssertionError("Mock supervisor did not start")
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=5) == 128 + signal.SIGTERM
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
    status = read_json(interrupted / "launcher.json")
    assert status["status"] == "interrupted" and status["signal"] == signal.SIGTERM
    assert all(mapping(entry)["returncode"] is not None for entry in sequence(status["workers"]))
    passed.append("supervisor SIGTERM stops and reaps both children and preserves PID/exit status/log evidence")
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root: Path = args.out.resolve()
    root.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    root.mkdir(parents=True, exist_ok=False)
    result = {"status": "passed", "utc": datetime.now(timezone.utc).isoformat(), "checks": checks(root),
              "scope": "short Python subprocesses only; no worker queue, geometry, native solver or GPU",
              "sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                         for name in ("launcher.py", "check_launcher.py")}}
    (root / "validation.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
