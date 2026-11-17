"""Read-only input-layout preflight for the requested first EFVB chunk; no mesh/solver."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from time import perf_counter
from zipfile import ZipFile

import numpy as np

from scripts.dev_1_1_preprocessing.prepare_data import DatasetSource, read_field, source_pairs
from scripts.project_paths import PROJECT_ROOT, SOURCE


def scan(source: Path, count: int = 5000) -> dict[str, object]:
    """Check every requested VP/VS field, retaining only compact aggregate evidence."""
    if not 1 <= count <= 5000:
        raise ValueError("This initial-chunk preflight is limited to EFVB 00000..04999")
    archive_path = source / "data_EFVB_out.zip"
    dataset = DatasetSource(archive_path, ZipFile(archive_path)) if archive_path.is_file() else DatasetSource(source / "data_EFVB_out", None)
    started = perf_counter()
    nonflat: list[int] = []
    layer_histogram: Counter[int] = Counter()
    combined = hashlib.sha256()
    minimum_thickness = 700.0
    vp_min, vp_max, vs_min, vs_max = float("inf"), 0., float("inf"), 0.
    try:
        for model_id, vp_name, vs_name in source_pairs(dataset)[:count]:
            raw_vp, raw_vs = dataset.read(vp_name), dataset.read(vs_name)
            for name, raw in ((vp_name, raw_vp), (vs_name, raw_vs)):
                combined.update(name.encode() + b"\0" + hashlib.sha256(raw).digest())
            vp, vs = read_field(raw_vp, vp_name), read_field(raw_vs, vs_name)
            if np.any(vp**2 <= (4. / 3.) * vs**2):
                raise ValueError(f"Nonpositive isotropic bulk modulus in EFVB {model_id:05d}")
            vp_min, vp_max = min(vp_min, float(vp.min())), max(vp_max, float(vp.max()))
            vs_min, vs_max = min(vs_min, float(vs.min())), max(vs_max, float(vs.max()))
            if not (np.all(vp == vp[:, :1]) and np.all(vs == vs[:, :1])):
                nonflat.append(model_id)
                continue
            boundaries = np.r_[0, np.flatnonzero((np.diff(vp[:, 0]) != 0) | (np.diff(vs[:, 0]) != 0)) + 1, 70]
            layer_histogram[len(boundaries) - 1] += 1
            minimum_thickness = min(minimum_thickness, float(np.min(np.diff(boundaries))) * 10.)
    finally:
        dataset.close()
    return {"schema": "chapter14.efvb_flat_input_scan.v1", "dataset": "EFVB", "first_id": 0,
        "last_id": count - 1, "models_checked": count, "source": str(dataset.directory.resolve()),
        "ordered_member_hashes_sha256": combined.hexdigest(), "all_columns_identical": not nonflat,
        "nonflat_ids": nonflat, "layer_count_histogram": dict(sorted(layer_histogram.items())),
        "minimum_flat_layer_thickness_m": minimum_thickness, "vp_range_m_s": [vp_min, vp_max],
        "vs_range_m_s": [vs_min, vs_max], "wall_s": perf_counter() - started,
        "scope": "input layout/positivity only; no meshes, stability estimates, solver or GPU",
        "geometry_implication": "flat inputs use the already checked layered construction; each actual mesh still requires its own geometry and stability certificates"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--count", type=int, default=5000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output: Path = args.out.resolve()
    output.relative_to(PROJECT_ROOT / "data/dev_1_4_server_preparation")
    if output.exists():
        raise FileExistsError(output)
    report = scan(args.source, args.count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
