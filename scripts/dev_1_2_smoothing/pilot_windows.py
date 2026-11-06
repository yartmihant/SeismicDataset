"""Run the seismic pilot with an explicit native Windows CEED/JIT toolchain.

Start from a VS x64 developer shell so clang-cl can find MSVC/SDK headers,
libraries and link.exe. Arguments after the two toolchain options are passed
unchanged to pilot_run; a DLL search-directory handle lives through the run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
import shutil
import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--ceed-prefix", type=Path, required=True)
    parser.add_argument("--jit-cxx", type=Path, required=True)
    args, pilot_args = parser.parse_known_args(argv)
    if sys.platform != "win32":
        raise RuntimeError("This bootstrap is only for native Windows")
    prefix = args.ceed_prefix.resolve()
    compiler = args.jit_cxx.resolve()
    if not (prefix / "bin" / "ceed.dll").is_file():
        raise FileNotFoundError(prefix / "bin" / "ceed.dll")
    if not compiler.is_file():
        raise FileNotFoundError(compiler)
    if not os.environ.get("INCLUDE") or not os.environ.get("LIB") or shutil.which("link.exe") is None:
        raise RuntimeError("Start from the VS x64 developer shell (INCLUDE, LIB and link.exe are required)")
    os.environ["CEED_ROOT"] = str(prefix)
    os.environ["CEED_CPU_JIT_CXX"] = str(compiler)
    with os.add_dll_directory(str(prefix / "bin")):
        from scripts.dev_1_2_smoothing.pilot_run import main as pilot_main

        pilot_main(pilot_args)


if __name__ == "__main__":
    main()
