"""Автономный журнал Fidesys для точной B-сплайновой геометрии ECVB (Linux)."""
from __future__ import annotations

import base64
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Mapping

DEFAULT_FIDESYS = Path("/home/antonov/.bin/CAE-Fidesys-10.0")


def _run(install: Path, work: Path, script: str, *args: Path | str) -> None:
    bin_dir = install.resolve() / "preprocessor/bin"
    python = bin_dir / "python3"
    config = work / "config"
    (config / "Coreform").mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONHOME": str(python), "PYTHONPATH": str(bin_dir),
           "LD_LIBRARY_PATH": f"{python}:{bin_dir}", "QT_QPA_PLATFORM": "offscreen",
           "XDG_CONFIG_HOME": str(config), "XDG_CACHE_HOME": str(work / "cache")}
    result = subprocess.run([python / "python3", Path(__file__).resolve().with_name(script), *args],
                            env=env, cwd=work, capture_output=True, timeout=600)
    if result.returncode:
        raise RuntimeError("Fidesys journal: " + (result.stdout + result.stderr)[-4000:].decode(errors="replace"))


def _top_intervals(cad: dict) -> dict[int, int]:
    result: dict[int, int] = {}
    for curve in cad["curves"]:
        poles = curve["control_points_m"]
        if -1 in curve["adjacent_regions"] and all(abs(y) <= 1e-9 for _, y in poles):
            x0, x1 = poles[0][0], poles[-1][0]
            if any(abs(x - 10 * round(x / 10)) > 1e-8 for x in (x0, x1)):
                raise ValueError("Top curve endpoint misses 10 m grid")
            intervals = round(abs(x1 - x0) / 10)
            if intervals <= 0:
                raise ValueError("Empty top curve")
            result[curve["id"]] = intervals
    if sum(result.values()) != 140:
        raise ValueError("Top curves must span 1400 m")
    return result


def write_journal(cad: dict, path: Path, surface_sizes: Mapping[int | str, float] | None = None,
                  install: Path = DEFAULT_FIDESYS) -> None:
    """Записать кривые без повторной аппроксимации, контуры регионов и настройки сетки."""
    with tempfile.TemporaryDirectory(prefix="ecvb-cad-") as name:
        work = Path(name)
        src, sat, normals = (work / f for f in ("cad.json", "curves.sat", "normals.json"))
        src.write_text(json.dumps(cad, allow_nan=False), encoding="utf-8")
        _run(install, work, "journal_kernel.py", "export", src, sat, normals)
        payload = sat.read_bytes()
        reverse = set(json.loads(normals.read_text(encoding="utf-8")))

    lines = [f"# ECVB{cad['model']:05d} | coordinates in metres | free surface y=0 | z=0",
             "# Linux Fidesys 10.0; replay this geometry journal in an empty model.",
             "# Exact ACIS edges: degrees, knots and control points are listed below.",
             "# Each interface is created once, including interfaces around holes."]
    for curve in cad["curves"]:
        lines.append(f"# C{curve['id']} degree={curve['degree']} regions={curve['adjacent_regions']}")
        lines.append(f"# knots: {curve['knots']}")
        lines.append(f"# poles (x,y): {curve['control_points_m']}")

    lines += ["", "reset", "# Decode exact edges to an Aprepro-created unique temporary file (GNU base64).",
              "#{ecvb_index = get_temp_filename()}",
              '#{ecvb_leaf = get_word(word_count(ecvb_index, "/"), ecvb_index, "/")}',
              '#{ecvb_tmpdir = getenv("TMPDIR") == "" ? "/tmp" : "$TMPDIR"}',
              "#{ecvb_errors = get_error_count()}",
              "system 'set -f; IFS=; mktemp /tmp/ecvb-XXXXXXXXXX.sat > {ecvb_tmpdir}/{ecvb_leaf}'",
              "#{if(get_error_count() > ecvb_errors)}", "delete file '{ecvb_index}'", "stop", "#{endif}",
              '#{ecvb_sat = extract(file_to_string(ecvb_index), "/tmp/", ".sat") // ".sat"}',
              "delete file '{ecvb_index}'"]
    # The shell receives a fixed directory expression and the alphanumeric mkstemp
    # basename. TMPDIR is expanded by the shell as data; IFS/globbing are disabled.
    for index in range(0, len(payload), 768):
        chunk = base64.b64encode(payload[index:index + 768]).decode("ascii")
        lines.append("system 'echo " + chunk +
                     " | base64 -d >> {ecvb_sat}'")
    lines += ["import acis '{ecvb_sat}'", "delete file '{ecvb_sat}'", "",
              "# Make one sheet per connected material region; signed loop IDs record orientation."]
    for region in cad["regions"]:
        rid = region["id"]
        edges = [str(abs(cid)) for loop in region["loops"] for cid in loop]
        lines += [f"# R{rid} Vp={region['vp']} m/s, Vs={region['vs']} m/s; loops {region['loops']}",
                  "create surface curve " + " ".join(edges)]
        if rid in reverse:
            lines.append(f"reverse surface {rid}")
    lines += ["merge curve all", "merge vertex all", "compress all", "", "surface all size 10"]
    for rid, size in (surface_sizes or {}).items():
        if int(rid) not in {r["id"] for r in cad["regions"]} or not math.isfinite(size) or size <= 0:
            raise ValueError("Invalid local surface size")
        lines.append(f"surface {int(rid)} size {size:g}")
    for cid, intervals in _top_intervals(cad).items():
        lines += [f"curve {cid} scheme equal", f"curve {cid} interval {intervals}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_journal(cad: dict, path: Path, install: Path = DEFAULT_FIDESYS) -> dict:
    """Проверить штатный playback в чистом процессе на независимых выборках SciPy."""
    import numpy as np
    from scipy.interpolate import BSpline

    samples: dict[int, list] = {}
    for curve in cad["curves"]:
        knots = np.asarray(curve["knots"])
        breaks = np.unique(knots)
        ts = np.unique(np.concatenate([np.linspace(a, b, 33) for a, b in zip(breaks[:-1], breaks[1:])]))
        samples[curve["id"]] = BSpline(knots, curve["control_points_m"], curve["degree"])(ts).tolist()
    with tempfile.TemporaryDirectory(prefix="ecvb-replay-") as name:
        work = Path(name)
        inputs, result = work / "reference.json", work / "result.json"
        inputs.write_text(json.dumps({"cad": cad, "samples": samples, "top": _top_intervals(cad)}),
                          encoding="utf-8")
        _run(install, work, "journal.py", "verify", path.resolve(), inputs, result)
        return json.loads(result.read_text(encoding="utf-8"))


def _verify_in_fidesys(journal: Path, reference: Path, result: Path) -> None:
    import cubit  # type: ignore[import-not-found]
    import fidesys  # type: ignore[import-not-found]

    ref = json.loads(reference.read_text(encoding="utf-8"))
    cad = ref["cad"]
    cubit.init(["cubit", "-nojournal", "-nographics"])
    app = fidesys.FidesysComponent()
    app.start_up_no_args()
    # A fixed local name keeps workspace paths containing quotes out of the command language.
    local = result.parent / "geometry.jou"
    local.write_bytes(journal.read_bytes())
    cubit.cmd(f"playback '{local}'")
    ids = {c["id"] for c in cad["curves"]}
    if (cubit.get_error_count() or set(cubit.get_entities("curve")) != ids
            or set(cubit.get_entities("surface")) != {r["id"] for r in cad["regions"]}
            or cubit.get_list_of_free_ref_entities("curve") or cubit.get_list_of_free_ref_entities("vertex")):
        raise ValueError("Journal playback failed to reconstruct complete sheet geometry")
    max_dist = 0.0
    for curve in cad["curves"]:
        cid = curve["id"]
        if set(cubit.get_relatives("curve", cid, "surface")) != set(curve["adjacent_regions"]) - {-1}:
            raise ValueError(f"Incorrect journal contact at curve {cid}")
        edge = cubit.curve(cid)
        for x, y in ref["samples"][str(cid)]:
            xyz = (x, y, 0.0)
            max_dist = max(max_dist, math.dist(edge.closest_point_trimmed(xyz), xyz))
    if not max_dist < 1e-7:
        raise ValueError(f"Reference spline projection error: {max_dist:g} m")
    for region in cad["regions"]:
        rid = region["id"]
        if (cubit.get_surface_num_loops(rid) != len(region["loops"]) or
                cubit.get_surface_normal(rid)[2] < .999999 or
                set(cubit.get_relatives("surface", rid, "curve")) !=
                {abs(cid) for loop in region["loops"] for cid in loop}):
            raise ValueError(f"Incorrect surface normal or boundary loops at region {rid}")
    area = sum(cubit.get_surface_area(r["id"]) for r in cad["regions"])
    top_curves = " ".join(ref["top"])
    cubit.cmd("mesh curve " + top_curves)
    nodes = cubit.parse_cubit_list("node", "in curve " + top_curves)
    coordinates = sorted(cubit.get_nodal_coordinates(n) for n in nodes)
    if len(coordinates) != 141 or cubit.get_error_count():
        raise ValueError("Incorrect journal upper boundary mesh")
    top_dist = max(math.dist(xyz, (10.0*k, 0.0, 0.0)) for k, xyz in enumerate(coordinates))
    if top_dist > 1e-8:
        raise ValueError("Journal upper boundary is not uniform")
    result.write_text(json.dumps({"engine": "Fidesys Cubit native playback; exact ACIS curves",
                      "curves": len(ids), "regions": len(cad["regions"]),
                      "reference_samples": sum(map(len, ref["samples"].values())),
                      "max_sample_to_cad_m": max_dist, "cubit_estimated_total_area_m2": area,
                      "top_nodes": len(coordinates), "top_max_coordinate_error_m": top_dist}),
                      encoding="utf-8")


if __name__ == "__main__" and sys.argv[1:2] == ["verify"]:
    _verify_in_fidesys(*(Path(x) for x in sys.argv[2:]))
