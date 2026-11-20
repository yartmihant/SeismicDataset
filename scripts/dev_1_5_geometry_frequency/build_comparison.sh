#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-saturn}" == local ]]; then
  ROOT=$(pwd)
  CUDA_ROOT=/usr/local/cuda-12.8
  CUDA_ARCH=86
else
  cd /home/antonov/SeismicDataset
  ROOT=$PWD
  CUDA_ROOT=/usr/local/cuda-12.0
  CUDA_ARCH=80
fi
RUNTIME="$ROOT/.runtime_ch15_comparison_v1"
PY="$RUNTIME/venv/bin/python"
SOURCE="$ROOT/docs/runtime_sources/seismic_cu_v1"
test ! -e "$RUNTIME"
mkdir -p "$RUNTIME/cache" "$RUNTIME/tmp"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/cache" CUDA_CACHE_PATH="$RUNTIME/cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$ROOT/.venv/bin:/snap/bin:$CUDA_ROOT/bin:/usr/bin:/bin"
.venv/bin/python -m venv "$RUNTIME/venv"
.venv/bin/python - <<'PY'
import sysconfig,shutil,json,hashlib
import semgpu2
from semgpu2.solvers.lin_elast_2d_tenzor_newmark_kim_cu import _native
from pathlib import Path
root=Path.cwd(); runtime=root/'.runtime_ch15_comparison_v1'
base=Path(sysconfig.get_path('purelib')); target=next((runtime/'venv/lib').glob('python*/site-packages'))
(target/'project_dependencies.pth').write_text(str(base)+'\n')
shutil.copytree(Path(semgpu2.__file__).parent,target/'semgpu2',ignore=shutil.ignore_patterns('__pycache__','lin_elast_2d_seismic_newmark_kim_cu'))
old_native=Path(_native.__file__); shutil.copy2(old_native,target/'semgpu2/solvers/lin_elast_2d_tenzor_newmark_kim_cu'/old_native.name)
source=root/'docs/runtime_sources/seismic_cu_v1'
meta=json.loads((source/'provenance.json').read_text())
for path,sha in meta['files_sha256'].items(): assert hashlib.sha256((source/path).read_bytes()).hexdigest()==sha
shutil.copytree(source/'src/semgpu2/solvers/lin_elast_2d_seismic_newmark_kim_cu',target/'semgpu2/solvers/lin_elast_2d_seismic_newmark_kim_cu')
PY
cmake -S "$SOURCE" -B "$RUNTIME/build" -G Ninja -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DPython_EXECUTABLE="$PY" -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build "$RUNTIME/build" --target semgpu2_seismic_cu_native --parallel 2
"$PY" - <<'PY'
import sysconfig,shutil
from pathlib import Path
root=Path.cwd(); built=list((root/'.runtime_ch15_comparison_v1/build/seismic_cu').glob('_native*.so'));assert len(built)==1
shutil.copy2(built[0],Path(sysconfig.get_path('platlib'))/'semgpu2/solvers/lin_elast_2d_seismic_newmark_kim_cu'/built[0].name)
from semgpu2.solvers.lin_elast_2d_seismic_newmark_kim_cu import SOLVER_ID
print('NEW_SOLVER_IMPORTED',SOLVER_ID)
PY
