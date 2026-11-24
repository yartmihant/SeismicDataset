"""Экспорт точных кривых ACIS для автономного журнала, Python 3.8 Fidesys."""
import argparse
import json
from pathlib import Path
from typing import List

import cubit  # type: ignore[import-not-found]
import fidesys  # type: ignore[import-not-found]

if __package__:
    from .cubit_worker import build_curves, command
else:
    from cubit_worker import build_curves, command  # type: ignore[import-not-found,no-redef]


def export_curves(cad_path: Path, sat_path: Path, normals_path: Path) -> None:
    cad = json.loads(cad_path.read_text(encoding="utf-8"))
    cubit.init(["cubit", "-nojournal", "-nographics"])
    app = fidesys.FidesysComponent()
    app.start_up_no_args()
    command("reset")
    build_curves(cad)
    ids = [curve["id"] for curve in cad["curves"]]
    lengths = [cubit.get_curve_length(i) for i in ids]
    if "'" in str(sat_path) or "\n" in str(sat_path):
        raise ValueError("Unsupported quote or newline in ACIS staging path")
    command("export acis '" + str(sat_path) + "' curve all overwrite")
    # Derive normals in exactly the kernel state in which the journal creates faces.
    command("reset")
    command("import acis '" + str(sat_path) + "'")
    if set(cubit.get_entities("curve")) != set(ids):
        raise ValueError("ACIS round trip changed curve IDs")
    for i, length in zip(ids, lengths):
        if abs(cubit.get_curve_length(i) - length) > 1e-9:
            raise ValueError("ACIS round trip changed length of curve %d" % i)
    reverse: List[int] = []
    for region in cad["regions"]:
        edges = [str(abs(i)) for loop in region["loops"] for i in loop]
        command("create surface curve " + " ".join(edges))
        i = region["id"]
        if cubit.get_last_id("surface") != i:
            raise ValueError("ACIS round trip changed surface IDs")
        if cubit.get_surface_normal(i)[2] < 0.0:
            command("reverse surface %d" % i)
            reverse.append(i)
    normals_path.write_text(json.dumps(reverse) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["export"])
    parser.add_argument("cad", type=Path)
    parser.add_argument("sat", type=Path)
    parser.add_argument("normals", type=Path)
    args = parser.parse_args()
    export_curves(args.cad.resolve(), args.sat.resolve(), args.normals.resolve())
