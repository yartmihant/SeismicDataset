"""Генерация проверенных QUAD8-сеток ECVB по номерам моделей или диапазону."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from scripts.dev_1_4_server_preparation.model_input import read_model
from scripts.dev_1_6_fidesys_mesh.geometry import geometry
from scripts.dev_1_6_fidesys_mesh.validate import RefineRegions, validate
from scripts.project_paths import PROJECT_ROOT, SOURCE

DEFAULT_OUT = PROJECT_ROOT / "data/dev_1_6_fidesys_mesh/pilot_v2"
DEFAULT_FIDESYS = Path("/home/antonov/.bin/CAE-Fidesys-10.0")
WORKER = Path(__file__).with_name("cubit_worker.py").resolve()
SCHEMA = "ecvb.fidesys.quad8.v2"


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",encoding="utf-8")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()


def generate(i: int, root: Path, source: Path, install: Path, timeout: int, min_scaled_jacobian: float) -> str:
    case = f"ECVB{i:05d}"
    destination = root/case
    fc_name = case+".fc"
    model = read_model("ECVB",i,source)
    if destination.exists():
        existing = json.loads((destination/"result.json").read_text())
        if (existing["schema"] != SCHEMA or existing["model_id"] != i or
                existing["source_sha256"] != model.provenance["raw_sha256"] or
                existing["metrics"]["min_scaled_jacobian"]["min"] < min_scaled_jacobian or
                existing["status"] != "completed" or existing["fc_sha256"] != sha(destination/fc_name) or
                existing["cad_sha256"] != sha(destination/"geometry.json")):
            raise ValueError(f"Existing result failed verification: {destination}")
        return "verified_existing"
    start = time.monotonic()
    with tempfile.TemporaryDirectory(dir=root,prefix=f".{case}-") as staging:
        work = Path(staging)
        try:
            cad = geometry(model)
            save(work/"geometry.json",cad)
            # A writable per-process configuration also isolates concurrent Cubit sessions.
            bindir = install/"preprocessor/bin"
            python = bindir/"python3"
            config = work/"config"
            (config/"Coreform").mkdir(parents=True)
            env = {**os.environ,"PYTHONPATH":str(bindir),"PYTHONHOME":str(python),
                   "LD_LIBRARY_PATH":f"{python}:{bindir}","QT_QPA_PLATFORM":"offscreen",
                   "XDG_CONFIG_HOME":str(config),"XDG_CACHE_HOME":str(work/"cache")}
            fc = work/fc_name
            refinements: dict[int, float] = {}
            attempts: list[dict] = []
            for attempt in range(4):
                with (work/"build.log").open("a") as stream:
                    subprocess.run([python/"python3",WORKER,work],cwd=work,env=env,
                                   stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=timeout)
                try:
                    report = validate(fc,cad,min_scaled_jacobian)
                    break
                except RefineRegions as exc:
                    attempts.append({"attempt":attempt+1,"error":exc.reason,"fc_elements":exc.element_ids,
                                     "region_ids":exc.region_ids})
                    if attempt == 3:
                        raise
                    for region_id in exc.region_ids:
                        refinements[region_id] = 7.5 if region_id not in refinements else refinements[region_id] * (2./3.)
                    save(work/"refinement.json",refinements)
                    fc.unlink()  # Rejected, unpublished candidate in this private staging directory.
            save(work/"result.json",{"schema":SCHEMA,"status":"completed","model_id":i,
                 "source_sha256":model.provenance["raw_sha256"],"method":cad["method"],
                 "fc_bytes":fc.stat().st_size,"fc_sha256":sha(fc),"cad_sha256":sha(work/"geometry.json"),
                 "min_scaled_jacobian_requested":min_scaled_jacobian,"rejected_attempts":attempts,"surface_target_sizes_m":refinements,
                 "elapsed_s":time.monotonic()-start,"metrics":report})
            # Publish only complete validated artifacts, with no replacement of another result.
            product = work/"product"
            product.mkdir()
            for name in (fc_name,"geometry.json","mesh_export.jou","result.json"):
                (work/name).rename(product/name)
            product.rename(destination)
        except Exception as exc:
            failure = root/"failures"/(case+"-"+uuid.uuid4().hex[:12])
            failure.mkdir(parents=True)
            for name in ("geometry.json","build.log","mesh_export.jou","refinement.json"):
                if (work/name).is_file():
                    (work/name).rename(failure/name)
            save(failure/"failure.json",{"id":i,"error":str(exc),"source":model.provenance})
            raise RuntimeError(f"{case}: {exc}. Diagnostics: {failure}") from exc
    return "completed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--ids",nargs="+",type=int)
    group.add_argument("--range",nargs=2,type=int,metavar=("FIRST","LAST"),help="Оба конца включены")
    parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--fidesys",type=Path,default=DEFAULT_FIDESYS)
    parser.add_argument("--timeout",type=int,default=600,help="Таймаут одного процесса Fidesys, секунд")
    parser.add_argument("--min-scaled-jacobian",type=float,default=.05,
                        help="Рабочий порог формы в GLL p6; 0 оставляет только сертификат det J")
    parser.add_argument("--keep-going",action="store_true",help="Продолжать после отказа отдельной модели")
    args = parser.parse_args()
    if not __debug__:
        parser.error("Способ запуска отключил проверки assert")
    if not 0 <= args.min_scaled_jacobian <= 1:
        parser.error("Scaled Jacobian threshold must be in [0,1]")
    selected = args.ids if args.ids else range(args.range[0],args.range[1]+1)
    if not selected or min(selected)<0 or max(selected)>=30000 or args.timeout<=0:
        parser.error("Допустимые ID 0…29999, интервал непустой, таймаут положительный")
    root = args.out.resolve()
    root.mkdir(parents=True,exist_ok=True)
    failed = []
    for i in selected:
        try:
            status = generate(i,root,args.source.resolve(),args.fidesys.resolve(),args.timeout,args.min_scaled_jacobian)
            print(f"ECVB{i:05d} {status}",flush=True)
        except Exception as exc:
            if not args.keep_going:
                raise
            print(str(exc),flush=True)
            failed.append(i)
    if failed:
        raise SystemExit(f"Failed model IDs: {failed}")


if __name__ == "__main__":
    main()
