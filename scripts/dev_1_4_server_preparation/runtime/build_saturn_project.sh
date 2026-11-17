#!/usr/bin/env bash
# Run from the Git-delivered project, entirely inside the user's selected directory.
set -euo pipefail
TASK_ROOT=/home/antonov/SeismicDataset
TASK_PY="$TASK_ROOT/.venv/bin/python"
TASK_STAGE="$TASK_ROOT/.runtime_stage_v1"
TASK_RUNTIME="$TASK_ROOT/data/dev_1_4_server_preparation/runtime_saturn_v2"
TASK_BUILD="$TASK_ROOT/.runtime_build/tenzor_cu"
TASK_SOURCE="$TASK_STAGE/sources/semgpu2"
test "$(hostname)" = saturn
test "$(id -u)" = 1013
test -d "$TASK_ROOT/.git"
test -d "$TASK_SOURCE"
test ! -e "$TASK_ROOT/.venv"
test ! -e "$TASK_RUNTIME/profile.json"
cd "$TASK_ROOT"
mkdir -p "$TASK_RUNTIME/tmp" "$TASK_ROOT/.runtime_cache"
export TMPDIR="$TASK_RUNTIME/tmp"
export XDG_CACHE_HOME="$TASK_ROOT/.runtime_cache"
export MPLCONFIGDIR="$TASK_ROOT/.runtime_cache/matplotlib"
export PATH="$TASK_ROOT/.venv/bin:/snap/bin:/usr/local/cuda-12.0/bin:/usr/bin:/bin"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
export PIP_DISABLE_PIP_VERSION_CHECK=1
export SEMGPU2_BUILD_CUDA=0
export SEMGPU2_BUILD_CEED=0
export CMAKE_BUILD_PARALLEL_LEVEL=2

/home/antonov/.pyenv/versions/3.13.5/bin/python -m venv "$TASK_ROOT/.venv"
check_space() {
  "$TASK_PY" -c 'import os;v=os.statvfs("/home/antonov/SeismicDataset");free=v.f_bavail*v.f_frsize;print("AVAILABLE_BYTES",free,flush=True);assert free>40*1024**3,"Below 20 GiB reserve + 20 GiB work allowance"'
}
check_space
printf '%s\n' "5106106c6197d882e46c00de12d7647718977ba4a21b540ffab1f3f4b27a01e0  $TASK_STAGE/runtime_saturn_v1/semgpu2_source.tar" | sha256sum --check
"$TASK_PY" - "$TASK_STAGE/runtime_saturn_v1/pip_freeze.txt" "$TASK_RUNTIME/requirements.txt" <<'PY'
from pathlib import Path
import sys
lines=Path(sys.argv[1]).read_text().splitlines()
assert sum(line.lower().startswith('semgpu2') for line in lines)==1
Path(sys.argv[2]).write_text('\n'.join(line for line in lines if not line.lower().startswith('semgpu2'))+'\n')
PY
"$TASK_PY" -m pip install --no-cache-dir --only-binary=:all: --retries 1 --timeout 20 -r "$TASK_RUNTIME/requirements.txt"
check_space
"$TASK_PY" -m pip install --no-cache-dir --no-deps --no-build-isolation "$TASK_SOURCE"
cmake -S "$TASK_SOURCE" -B "$TASK_BUILD" -G Ninja \
  -DSEMGPU2_BUILD_CUDA=ON -DSEMGPU2_BUILD_CEED=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=80 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.0/bin/nvcc \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.0 -DCMAKE_CXX_COMPILER=/usr/bin/g++ \
  -DPython_EXECUTABLE="$TASK_PY" \
  -Dpybind11_DIR="$("$TASK_PY" -m pybind11 --cmakedir)"
cmake --build "$TASK_BUILD" --target semgpu2_tenzor_cu_native --parallel 2
check_space
"$TASK_PY" - <<'PY'
from pathlib import Path
import hashlib, json, shutil, sysconfig
root=Path('/home/antonov/SeismicDataset')
built=list((root/'.runtime_build/tenzor_cu/tenzor_cu').glob('_native*.so'))
assert len(built)==1,built
target=Path(sysconfig.get_path('platlib'))/'semgpu2/solvers/lin_elast_2d_tenzor_newmark_kim_cu'/built[0].name
assert not target.exists(),target
shutil.copy2(built[0],target)
print(json.dumps({'native_installed':str(target),'sha256':hashlib.sha256(target.read_bytes()).hexdigest()}))
import semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu as solver
print('NATIVE_IMPORT_OK',solver.SOLVER_ID)
import gmsh
print('GMSH_IMPORT_OK',gmsh.__version__)
PY
"$TASK_PY" -m pip freeze > "$TASK_RUNTIME/pip_freeze.txt"
"$TASK_PY" scripts/dev_1_4_server_preparation/runtime/profile.py
printf '%s\n' 'BUILD_AND_IMPORT_COMPLETE'
