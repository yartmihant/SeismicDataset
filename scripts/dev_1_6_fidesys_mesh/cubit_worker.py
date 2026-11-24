"""Выполняется интерпретатором Python 3.8 из Fidesys в отдельном процессе."""
import json
from pathlib import Path
import sys
from typing import List

import cubit  # type: ignore[import-not-found]
import fidesys  # type: ignore[import-not-found]


def command(line: str) -> None:
    before = cubit.get_error_count()
    cubit.cmd(line)
    if cubit.get_error_count() != before:
        raise RuntimeError(line)


def build_curves(cad: dict) -> None:
    for c in cad["curves"]:
        points = c["control_points_m"]
        if c["degree"] == 1:
            ids = []
            for x,y in points:
                command(f"create vertex {x} {y} 0")
                ids.append(cubit.get_last_id("vertex"))
            command(f"create curve vertex {ids[0]} {ids[1]}")
        else:
            xyz = [v for x,y in points for v in (x,y,0.)]
            created = cubit.create_nurbs_curve(c["degree"],xyz,[1.]*len(points),c["knots"][1:-1])
            if created != c["id"]:
                raise ValueError("Unexpected NURBS curve ID")
        if cubit.get_last_id("curve") != c["id"]:
            raise ValueError("Unexpected curve numbering")


def build(work: Path) -> None:
    cad = json.loads((work / "geometry.json").read_text())
    cubit.init(["cubit", "-nojournal", "-nographics"])
    app = fidesys.FidesysComponent()
    app.start_up_no_args()
    command("reset")
    build_curves(cad)
    for r in cad["regions"]:
        edges = [str(abs(cid)) for loop in r["loops"] for cid in loop]
        command("create surface curve "+" ".join(edges))
        if cubit.get_last_id("surface") != r["id"]:
            raise ValueError("Unexpected surface numbering")
        if cubit.get_surface_normal(r["id"])[2] < 0:
            command(f"reverse surface {r['id']}")
    command("merge curve all")
    command("merge vertex all")
    command("compress all")
    for c in cad["curves"]:
        expected = set(c["adjacent_regions"])-{-1}
        if expected != set(cubit.get_relatives("curve",c["id"],"surface")):
            raise ValueError(f"Lost contact at curve {c['id']}")
    command("surface all size 10")
    for c in cad["curves"]:
        points = c["control_points_m"]
        if all(abs(y)<1e-9 for x,y in points) and -1 in c["adjacent_regions"]:
            length = abs(points[-1][0]-points[0][0])
            intervals = round(length/10.)
            if intervals < 1 or abs(length-10*intervals)>1e-8:
                raise ValueError("Top edge misses the 10 m grid")
            command(f"curve {c['id']} scheme equal")
            command(f"curve {c['id']} interval {intervals}")
    commands: List[str] = []
    plan = work / "refinement.json"
    if plan.is_file():
        for key, size in json.loads(plan.read_text()).items():
            commands.append(f"surface {key} size {size}")
    commands.extend(["mesh surface all","set duplicate block elements off","set node constraint on"])
    for r in cad["regions"]:
        commands.append(f"block {r['id']} add surface {r['id']}")
    commands.extend(["block all element type QUAD8","analysis type dynamic elasticity dim2 planestrain preload off"])
    filename = "ECVB%05d.fc" % cad["model"]
    (work/"mesh_export.jou").write_text("\n".join(commands+[f"export fidesyscase '{filename}'"])+"\n")
    for line in commands:
        command(line)
    command('export fidesyscase "'+str(work/filename)+'"')


if __name__ == "__main__":
    build(Path(sys.argv[1]).resolve())
