"""Read the installed saturn runtime; no solver or workload is started."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


def query(argv: list[str]) -> dict[str, object]:
    result = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=20)
    return {"argv": argv, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    output = root / "data/dev_1_4_server_preparation/runtime_saturn_v2/profile.json"
    if output.exists():
        raise FileExistsError(output)
    native = importlib.import_module("semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu._native")
    native_path = Path(str(native.__file__))
    gmsh = importlib.import_module("gmsh")
    stat = os.statvfs(root)
    payload = {
        "status": "native_imported_numerical_smoke_pending", "utc": datetime.now(timezone.utc).isoformat(),
        "host": platform.node(), "python": sys.version, "executable": sys.executable,
        "semgpu_revision": "acec5b9fe11a6ea812a675bdfaac03540a63702f",
        "packages": {name: version(name) for name in ("semgpu2", "numpy", "scipy", "pillow", "matplotlib", "gmsh", "pybind11", "scikit-build-core", "ninja")},
        "native_file": str(native_path), "native_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        "gmsh_module_file": str(gmsh.__file__), "cuda_architectures": "80", "parallel_build_jobs": 2,
        "installation": "standard_pure_python_wheel_plus_unmodified_CMake_semgpu2_tenzor_cu_native_target",
        "build_source_archive_sha256": hashlib.sha256((root / ".runtime_stage_v1/runtime_saturn_v1/semgpu2_source.tar").read_bytes()).hexdigest(),
        "available_bytes": stat.f_bavail * stat.f_frsize,
        "commands": {
            "cuda": query(["/usr/local/cuda-12.0/bin/nvcc", "--version"]),
            "cmake": query(["/snap/bin/cmake", "--version"]),
            "compiler": query(["/usr/bin/g++", "--version"]),
            "gpu": query(["nvidia-smi", "--query-gpu=index,name,driver_version,compute_cap,memory.free,utilization.gpu", "--format=csv"]),
            "native_dependencies": query(["ldd", str(native_path)]),
        },
        "limitations": ["Only the 2D tenzor CU extension is built", "No CEED runtime", "No numerical validation performed by this profile"],
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(output)


if __name__ == "__main__":
    main()
