"""Сборка переносимого набора ECVB по ID. На диске хранится только основная область."""
from __future__ import annotations

import argparse
import errno
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np

from scripts.dev_1_2_smoothing.model65_geometry import _material_labels, _regions
from scripts.dev_1_4_server_preparation.model_input import ModelInput, read_model
from scripts.dev_1_6_fidesys_mesh.generate import DEFAULT_FIDESYS, generate, save, sha
from scripts.dev_1_6_fidesys_mesh.journal import verify_journal, write_journal
from scripts.dev_1_6_fidesys_mesh.materials import write_material_fc
from scripts.dev_1_6_fidesys_mesh.sem_input import prepare
from scripts.project_paths import PROJECT_ROOT, SOURCE

DEFAULT_ROOT = PROJECT_ROOT / "data/dev_1_6_fidesys_mesh/dataset_v1"
SCHEMA = "ecvb.core.dataset.v1"


def directory(root: Path, i: int) -> Path:
    first = (i // 1000) * 1000
    return root / f"{first:05d}-{first + 999:05d}" / f"ECVB{i:05d}"


def products(i: int) -> tuple[str, ...]:
    return ("cartesian.npz", "geometry.jou", "geometry.json", f"ECVB{i:05d}.fc", "sem_input.npz")


def cartesian(path: Path, model: ModelInput, cad: dict) -> None:
    """Один и тот же регион имеет один ID в растре, CAD, блоке FC и карте SEM."""
    if cad["method"] == "layered_locked_depth_splines":
        # Layers have identical ordered runs in every column; a material may recur at depth.
        change = (np.diff(model.vp, axis=0) != 0) | (np.diff(model.vs, axis=0) != 0)
        regions = 1 + np.vstack((np.zeros((1, 70), dtype=int), np.cumsum(change, axis=0)))
    else:
        labels, _ = _material_labels(model.vp, model.vs)
        regions, _ = _regions(labels)
    if set(np.unique(regions)) != {r["id"] for r in cad["regions"]}:
        raise ValueError("CAD and Cartesian region IDs differ")
    for region in cad["regions"]:
        mask = regions == region["id"]
        if not (np.all(model.vp[mask] == region["vp"]) and np.all(model.vs[mask] == region["vs"])):
            raise ValueError(f"Raster/CAD material mismatch in region {region['id']}")
        if "source_cells" in region and int(mask.sum()) != region["source_cells"]:
            raise ValueError("Region size differs from the original raster")
    native_edges = np.arange(71, dtype=np.float64) * 10.
    np.savez_compressed(path, vp_m_s=model.vp, vs_m_s=model.vs,
                        region_id=regions.astype(np.int32), native_x_edges_m=native_edges,
                        x_edges_m=2 * native_edges, y_edges_m=-native_edges)
    with np.load(path, allow_pickle=False) as arrays:
        if not (np.array_equal(arrays["vp_m_s"], model.vp) and
                np.array_equal(arrays["vs_m_s"], model.vs) and
                np.array_equal(arrays["region_id"], regions)):
            raise ValueError("Cartesian round-trip failed")


def verify(destination: Path, model: ModelInput, quality: float) -> None:
    manifest = json.loads((destination / "manifest.json").read_text())
    if (manifest["schema"] != SCHEMA or manifest["dataset"] != "ECVB" or
            manifest["model_id"] != model.model_id or manifest["status"] != "completed" or
            manifest["source"]["raw_sha256"] != model.provenance["raw_sha256"] or
            manifest["mesh_quality"]["min_scaled_jacobian"]["min"] < quality or
            set(manifest["files"]) != set(products(model.model_id))):
        raise ValueError(f"Incompatible model manifest: {destination}")
    for name, entry in manifest["files"].items():
        path = destination / name
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Artifact failed verification: {path}")


def build(i: int, root: Path, source: Path, install: Path, cache: Path | None,
          timeout: int, quality: float) -> str:
    model = read_model("ECVB", i, source)
    destination = directory(root, i)
    if destination.exists():
        verify(destination, model, quality)
        return "verified_existing"
    destination.parent.mkdir(parents=True, exist_ok=True)
    case = destination.name
    with tempfile.TemporaryDirectory(prefix=f".{case}-", dir=destination.parent) as temp:
        work = Path(temp)
        product = work / "product"
        product.mkdir()
        mesh_root = cache if cache is not None and (cache / case).is_dir() else work
        try:
            generate(i, mesh_root, source, install, timeout, quality)
        except Exception:
            if (work / "failures").is_dir():
                failures = root / "failures"
                failures.mkdir(exist_ok=True)
                for failed in (work / "failures").iterdir():
                    failed.rename(failures / failed.name)
            raise
        mesh_dir = mesh_root / case
        mesh_result = json.loads((mesh_dir / "result.json").read_text())
        shutil.copyfile(mesh_dir / "geometry.json", product / "geometry.json")
        cad = json.loads((product / "geometry.json").read_text())
        if cad["model"] != i or cad["source"]["raw_sha256"] != model.provenance["raw_sha256"]:
            raise ValueError("CAD provenance differs from the source")
        cartesian(product / "cartesian.npz", model, cad)
        write_journal(cad, product / "geometry.jou", mesh_result["surface_target_sizes_m"], install=install)
        journal_report = verify_journal(cad, product / "geometry.jou", install)
        fc_name = f"{case}.fc"
        fc_report = write_material_fc(mesh_dir / fc_name, cad, product / fc_name)
        sem_report = prepare(product / fc_name, cad, product / "sem_input.npz")
        files = {name: {"sha256": sha(product / name), "bytes": (product / name).stat().st_size}
                 for name in products(i)}
        save(product / "manifest.json", {
            "schema": SCHEMA, "status": "completed", "dataset": "ECVB", "model_id": i,
            "source": {key: model.provenance[key] for key in
                       ("members", "raw_sha256", "array_sha256", "axes", "shape", "dtype")},
            "coordinates": {"units": "m", "source_cell_dx_dy_m": [10, 10],
                            "model_cell_dx_dy_m": [20, 10], "core_xy_bounds_m": [[0, 1400], [-700, 0]],
                            "free_top_y_m": 0, "top_edges": 140, "top_step_m": 10,
                            "persisted_pml": False, "geometry_order": 2, "persisted_gll_p6": False},
            "geometry_method": cad["method"], "files": files,
            "reference_mesh_sha256": mesh_result["fc_sha256"],
            "surface_target_sizes_m": mesh_result["surface_target_sizes_m"],
            "mesh_quality": mesh_result["metrics"], "cad_replay": journal_report,
            "fc_materials": fc_report, "sem_input": sem_report})
        verify(product, model, quality)
        try:
            product.rename(destination)
        except OSError as exc:
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not destination.is_dir():
                raise
            verify(destination, model, quality)
            return "verified_existing"
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids", type=int, nargs="+")
    group.add_argument("--range", nargs=2, type=int, metavar=("FIRST", "LAST"))
    parser.add_argument("--out", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--fidesys", type=Path, default=DEFAULT_FIDESYS)
    parser.add_argument("--mesh-cache", type=Path, help="Читать уже проверенные QUAD8 по ID, если есть")
    parser.add_argument("--timeout", type=int, default=600, help="Таймаут попытки построения QUAD8")
    parser.add_argument("--min-scaled-jacobian", type=float, default=.05)
    parser.add_argument("--keep-going", action="store_true")
    args = parser.parse_args()
    selected = args.ids if args.ids is not None else range(args.range[0], args.range[1] + 1)
    if (not __debug__ or not selected or min(selected) < 0 or max(selected) >= 30000 or
            args.timeout <= 0 or not 0 <= args.min_scaled_jacobian <= 1):
        parser.error("Требуются ID в диапазоне 0…29999, timeout > 0, порог в [0,1] и активные assert")
    failed = []
    cache = args.mesh_cache.resolve() if args.mesh_cache else None
    for i in dict.fromkeys(selected):
        try:
            status = build(i, args.out.resolve(), args.source.resolve(), args.fidesys.resolve(),
                           cache, args.timeout, args.min_scaled_jacobian)
            print(f"ECVB{i:05d} {status}", flush=True)
        except Exception as exc:
            if not args.keep_going:
                raise
            failed.append(i)
            print(f"ECVB{i:05d} failed: {exc}", flush=True)
    if failed:
        raise SystemExit(f"Failed IDs: {failed}")


if __name__ == "__main__":
    main()
