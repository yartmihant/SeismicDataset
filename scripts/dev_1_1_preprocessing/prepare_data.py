"""Convert Vasily's cell arrays losslessly to NPZ and preview global colours."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from scripts.project_paths import BINARY, DATA_11, DATA_12, IMG_11, IMG_12, PROJECT_ROOT, SOURCE, image_directory, report_font, resolve_provenance
from typing import Dict, List, Optional, Tuple
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont


FloatArray = NDArray[np.float64]
ROOT = DATA_11


@dataclass
class Bounds:
    vp_min: float = float("inf")
    vp_max: float = float("-inf")
    vs_min: float = float("inf")
    vs_max: float = float("-inf")

    def update(self, vp: FloatArray, vs: FloatArray) -> None:
        self.vp_min = min(self.vp_min, float(vp.min()))
        self.vp_max = max(self.vp_max, float(vp.max()))
        self.vs_min = min(self.vs_min, float(vs.min()))
        self.vs_max = max(self.vs_max, float(vs.max()))


@dataclass
class Shard:
    dataset: str
    file: str
    first_id: int
    last_id: int
    count: int
    sha256: str


@dataclass
class DatasetSource:
    """Read either an original ZIP or the existing unpacked dataset without writes."""

    directory: Path
    archive: Optional[ZipFile]

    @property
    def filename(self) -> str:
        return str(self.directory)

    def namelist(self) -> List[str]:
        if self.archive is not None:
            return self.archive.namelist()
        return [path.name for path in self.directory.glob("*.txt")]

    def read(self, name: str) -> bytes:
        if self.archive is not None:
            return self.archive.read(name)
        return (self.directory / name).read_bytes()

    def close(self) -> None:
        if self.archive is not None:
            self.archive.close()


def source_pairs(archive: DatasetSource) -> List[Tuple[int, str, str]]:
    vp_names = [name for name in archive.namelist() if Path(name).name.startswith("vp_") and name.endswith(".txt")]
    vs_names = [name for name in archive.namelist() if Path(name).name.startswith("vs_") and name.endswith(".txt")]
    vp = {int(Path(name).stem[3:]): name for name in vp_names}
    vs = {int(Path(name).stem[3:]): name for name in vs_names}
    if len(vp) != len(vp_names) or len(vs) != len(vs_names):
        raise ValueError(f"Duplicate IDs in {archive.filename}")
    if set(vp) != set(vs) or set(vp) != set(range(30000)):
        raise ValueError(f"Expected paired IDs 00000..29999 in {archive.filename}")
    return [(i, vp[i], vs[i]) for i in sorted(vp)]


def read_field(raw: bytes, name: str) -> FloatArray:
    field = np.loadtxt(io.BytesIO(raw), dtype=np.float64)
    if field.shape != (70, 70) or not np.isfinite(field).all() or (field <= 0).any():
        raise ValueError(f"Invalid field: {name}")
    return field


def convert(output: Path, shard_size: int, source: Path = SOURCE) -> None:
    if (output / "manifest.json").exists():
        raise FileExistsError("Completed output exists; use previews or a new --output")
    output.mkdir(parents=True, exist_ok=True)
    global_bounds = Bounds()
    class_bounds: Dict[str, Bounds] = {}
    shards: List[Shard] = []
    source_hash = hashlib.sha256()
    archive_hashes: Dict[str, str] = {}
    for dataset in ("EFVB", "ECVB"):
        archive_path = source / f"data_{dataset}_out.zip"
        if archive_path.is_file():
            archive_hashes[archive_path.name] = hashlib.sha256(archive_path.read_bytes()).hexdigest()
            archive = DatasetSource(archive_path, ZipFile(archive_path))
        else:
            directory = source / f"data_{dataset}_out"
            if not directory.is_dir():
                raise FileNotFoundError(f"Restore source ZIP or directory: {archive_path}")
            archive = DatasetSource(directory, None)
        pairs = source_pairs(archive)
        bounds = Bounds()
        class_bounds[dataset] = bounds
        for offset in range(0, len(pairs), shard_size):
            batch = pairs[offset:offset + shard_size]
            vp = np.empty((len(batch), 70, 70), dtype=np.float64)
            vs = np.empty_like(vp)
            ids = np.array([item[0] for item in batch], dtype=np.int64)
            for row, (model_id, vp_path, vs_path) in enumerate(batch):
                vp_raw, vs_raw = archive.read(vp_path), archive.read(vs_path)
                vp[row], vs[row] = read_field(vp_raw, vp_path), read_field(vs_raw, vs_path)
                if (vp[row] ** 2 <= (4.0 / 3.0) * vs[row] ** 2).any():
                    raise ValueError(f"Nonpositive isotropic bulk modulus: {dataset}/{model_id}")
                for name, raw in ((vp_path, vp_raw), (vs_path, vs_raw)):
                    source_hash.update((dataset + "/" + name).encode("utf-8"))
                    source_hash.update(b"\0")
                    source_hash.update(raw)
            bounds.update(vp, vs)
            global_bounds.update(vp, vs)
            name = f"{dataset}_{ids[0]:05d}_{ids[-1]:05d}.npz"
            target = output / name
            temporary = target.with_suffix(".partial")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, ids=ids, vp=vp, vs=vs)
            with np.load(temporary, allow_pickle=False) as restored:
                for key, expected in (("ids", ids), ("vp", vp), ("vs", vs)):
                    if restored[key].dtype != expected.dtype or not np.array_equal(restored[key], expected):
                        raise ValueError(f"Round-trip mismatch: {name}/{key}")
            temporary.replace(target)
            shards.append(Shard(dataset, name, int(ids[0]), int(ids[-1]), len(batch), hashlib.sha256(target.read_bytes()).hexdigest()))
            print(f"{dataset}: {offset + len(batch)}/{len(pairs)} verified", flush=True)
        archive.close()
    manifest = {
        "schema_version": 1, "dtype": "float64", "shape_per_field": [70, 70],
        "axes": ["model", "row_top_to_bottom", "column_left_to_right"],
        "units": "m/s", "cell_size_m": [10, 10], "density_kg_m3": 1000,
        "normalization": "global linear min/max per component over EFVB + ECVB; no clipping of input",
        "bounds": asdict(global_bounds),
        "class_bounds": {key: asdict(value) for key, value in class_bounds.items()},
        "source_archives_sha256": archive_hashes,
        "source_directory": str(source.resolve()),
        "source_digest_algorithm": "SHA256 of ordered dataset/source-relative-name + NUL + raw bytes; ZIP uses its original member names, unpacked input uses filenames; EFVB then ECVB, ascending ID, VP then VS",
        "source_sha256": source_hash.hexdigest(),
        "verification": "all pairs, exact 70x70 shape, finite, positive, positive isotropic bulk modulus; every NPZ array reloaded and exactly compared to parsed float64 input",
        "shards": [asdict(shard) for shard in shards],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def colour(p: FloatArray, s: FloatArray, variant: str) -> NDArray[np.uint8]:
    if variant == "rg":
        rgb = np.stack((p, s, np.zeros_like(p)), axis=-1)
    elif variant == "rb":
        rgb = np.stack((p, np.zeros_like(p), s), axis=-1)
    elif variant == "bivariate":
        # Bilinear blend: low/low white, high VP orange, high VS blue, high/high dark purple.
        c00 = np.array([245, 245, 240]) / 255.0
        c10 = np.array([230, 130, 45]) / 255.0
        c01 = np.array([65, 155, 210]) / 255.0
        c11 = np.array([70, 35, 95]) / 255.0
        p3, s3 = p[..., None], s[..., None]
        rgb = (1-p3)*(1-s3)*c00 + p3*(1-s3)*c10 + (1-p3)*s3*c01 + p3*s3*c11
    else:
        value = p if variant == "vp" else s
        # Fixed sequential dark-blue -> teal -> yellow palette for separate reference fields.
        stops = np.array([[25, 30, 85], [25, 130, 140], [250, 230, 60]]) / 255.0
        rgb = np.stack([np.interp(value, [0, 0.5, 1], stops[:, k]) for k in range(3)], axis=-1)
    return np.rint(np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def previews(output: Path) -> None:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    bounds = Bounds(**manifest["bounds"])
    destination = IMG_11 / "previews"
    destination.mkdir(parents=True, exist_ok=True)
    variants = ("rg", "rb", "bivariate", "vp", "vs")
    labels = ("A: R=VP, G=VS", "B: R=VP, B=VS", "C: VP orange / VS blue", "VP separately", "VS separately")
    samples = [(dataset, i) for dataset in ("EFVB", "ECVB") for i in (0, 15000, 29999)]
    font = report_font(18)
    small_font = report_font(15)
    sheet = Image.new("RGB", (5 * 370 + 20, 6 * 390 + 460), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((20, 10), f"Global scales, all 60,000 models: VP {bounds.vp_min:g}..{bounds.vp_max:g}; VS {bounds.vs_min:g}..{bounds.vs_max:g} m/s", fill="black", font=font)
    for row, (dataset, model_id) in enumerate(samples):
        shard = next(item for item in manifest["shards"] if item["dataset"] == dataset and item["first_id"] <= model_id <= item["last_id"])
        with np.load(output / shard["file"], allow_pickle=False) as arrays:
            index = int(np.flatnonzero(arrays["ids"] == model_id)[0])
            vp, vs = arrays["vp"][index], arrays["vs"][index]
        p = (vp - bounds.vp_min) / (bounds.vp_max - bounds.vp_min)
        s = (vs - bounds.vs_min) / (bounds.vs_max - bounds.vs_min)
        for col, (variant, label) in enumerate(zip(variants, labels)):
            picture = Image.fromarray(colour(p, s, variant)).resize((350, 350), Image.Resampling.NEAREST)
            picture.save(destination / f"{dataset}_{model_id:05d}_{variant}.png")
            x, y = 20 + col * 370, 55 + row * 390
            draw.text((x, y), f"{dataset} {model_id:05d} | {label}", fill="black", font=small_font)
            sheet.paste(picture, (x, y + 25))
    p_grid, s_grid = np.meshgrid(np.linspace(0, 1, 250), np.linspace(1, 0, 250))
    for col, variant in enumerate(variants):
        x, y = 20 + col * 370, 55 + 6 * 390
        legend = Image.fromarray(colour(p_grid, s_grid, variant))
        legend.save(destination / f"legend_{variant}.png")
        sheet.paste(legend, (x + 40, y + 30))
        draw.text((x, y), labels[col], fill="black", font=font)
        draw.text((x, y + 290), "VP increases left -> right", fill="black", font=small_font)
        draw.text((x, y + 310), "VS increases bottom -> top", fill="black", font=small_font)
    sheet.save(destination / "comparison.png")
    print(f"Preview sheet: {destination / 'comparison.png'}", flush=True)


def group_previews(output: Path) -> None:
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    bounds = Bounds(**manifest["bounds"])
    destination = IMG_11 / "previews" / "A"
    destination.mkdir(parents=True, exist_ok=True)
    font = report_font(20)
    samples = [int(i) for i in np.linspace(0, 29999, 16, dtype=np.int64)]
    counts: Dict[str, int] = {}
    for dataset in ("EFVB", "ECVB"):
        folder = destination / dataset
        folder.mkdir(exist_ok=True)
        sheet = Image.new("RGB", (1500, 1640), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((20, 12), f"{dataset} | A: red = VP, green = VS | 16 evenly spaced model IDs", fill="black", font=font)
        draw.text((20, 42), f"Global VP: {bounds.vp_min:g}..{bounds.vp_max:g} m/s; VS: {bounds.vs_min:.5f}..{bounds.vs_max:.5f} m/s", fill="black", font=font)
        count = 0
        for shard in manifest["shards"]:
            if shard["dataset"] != dataset:
                continue
            path = output / shard["file"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != shard["sha256"]:
                raise ValueError(f"NPZ checksum mismatch: {path}")
            with np.load(path, allow_pickle=False) as arrays:
                ids, vp, vs = arrays["ids"], arrays["vp"], arrays["vs"]
            def write_image(index: int) -> Optional[Tuple[int, Image.Image]]:
                model_id = int(ids[index])
                p = (vp[index] - bounds.vp_min) / (bounds.vp_max - bounds.vp_min)
                s = (vs[index] - bounds.vs_min) / (bounds.vs_max - bounds.vs_min)
                rgb = colour(p, s, "rg")
                picture = Image.fromarray(rgb).resize((350, 350), Image.Resampling.NEAREST)
                target = folder / f"{model_id:05d}.png"
                if not target.exists():
                    picture.save(target)
                with Image.open(target) as restored:
                    pixels = np.asarray(restored)
                    expected = np.repeat(np.repeat(rgb, 5, axis=0), 5, axis=1)
                    if restored.mode != "RGB" or restored.size != (350, 350) or not np.array_equal(pixels, expected):
                        raise ValueError(f"PNG mismatch: {target}")
                if model_id in samples:
                    return model_id, picture
                return None

            with ThreadPoolExecutor(max_workers=8) as executor:
                for sample in executor.map(write_image, range(len(ids))):
                    if sample is not None:
                        model_id, picture = sample
                        slot = samples.index(model_id)
                        x, y = 20 + (slot % 4) * 370, 85 + (slot // 4) * 385
                        draw.text((x, y), f"{dataset} {model_id:05d}", fill="black", font=font)
                        sheet.paste(picture, (x, y + 25))
                    count += 1
            print(f"PNG {dataset}: {count}/30000 verified", flush=True)
        if count != 30000 or len(list(folder.glob("*.png"))) != 30000:
            raise ValueError(f"Unexpected PNG count: {dataset}")
        counts[dataset] = count
        sheet.save(destination / f"{dataset}_overview.png")
    record = {
        "variant": "A", "rgb": ["normalized VP", "normalized VS", "0"],
        "bounds": asdict(bounds), "size_px": [350, 350],
        "resampling": "nearest; each cell 5x5 pixels", "counts": counts,
        "binary_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "overview_ids": samples,
        "verification": "Every NPZ checksum matched; every PNG reloaded and compared pixelwise with 5x5 repeated RGB cell array",
    }
    (destination / "manifest.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("convert", "previews", "group-previews"))
    parser.add_argument("--output", type=Path, default=BINARY)
    parser.add_argument("--shard-size", type=int, default=1000)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    if args.command == "convert":
        convert(args.output, args.shard_size, args.source)
    elif args.command == "previews":
        previews(args.output)
    else:
        group_previews(args.output)


if __name__ == "__main__":
    main()
