"""Удалить только явно разрешённые старые серверные опыты перед новой серией."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import socket
from scripts.project_paths import PROJECT_ROOT

OLD_NAMES=('efvb_chunk01_saturn_v1','pilot_saturn_v1','smoke_saturn_v1')

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--delete',action='store_true');args=parser.parse_args()
    if not args.delete:raise ValueError('Explicit --delete is required')
    if socket.gethostname()!='saturn' or PROJECT_ROOT!=Path('/home/antonov/SeismicDataset'):
        raise ValueError('Cleanup is restricted to the authorized saturn project')
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            words=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError):continue
        if any(word in (b'scripts.dev_1_4_server_preparation.worker',b'scripts.dev_1_4_server_preparation.launcher',b'scripts.dev_1_4_server_preparation.production_worker',b'scripts.dev_1_4_server_preparation.production_launcher') for word in words):
            raise RuntimeError('Project workers must be stopped before cleanup')
    base=PROJECT_ROOT/'data/dev_1_4_server_preparation';report=base/'restart_saturn_v3'
    report.mkdir(exist_ok=False)
    records=[]
    for name in OLD_NAMES:
        path=base/name
        if path.is_symlink() or path.resolve()!=path or not path.is_dir():raise ValueError(f'Unexpected cleanup path: {path}')
        files=list(path.rglob('*'))
        if any(p.lstat().st_uid!=os.getuid() for p in [path,*files]):raise ValueError('Foreign-owned entry in cleanup directory')
        records.append({'path':str(path),'entries':len(files),'allocated_bytes':sum(p.lstat().st_blocks*512 for p in [path,*files])})
    before=shutil.disk_usage(base).free
    plan={'utc':datetime.now(timezone.utc).isoformat(),'authorization':'User explicitly requested deleting old saturn calculation attempts and restarting with k0.5684,1s,3000model chunks','directories':records,'free_before_bytes':before,'preserved':'src, environment, code, runtime audit, benchmark reference and other users/projects'}
    with (report/'cleanup_plan.json').open('x') as f:json.dump(plan,f,indent=2)
    for name in OLD_NAMES:shutil.rmtree(base/name)
    result={**plan,'status':'completed','free_after_bytes':shutil.disk_usage(base).free,'all_requested_directories_removed':all(not (base/n).exists() for n in OLD_NAMES)}
    with (report/'cleanup.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))

if __name__=='__main__':main()
