#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-saturn}" == local ]]; then
  CUDA_ROOT=/usr/local/cuda-12.8
  CUDA_ARCH=86
else
  cd /home/antonov/SeismicDataset
  CUDA_ROOT=/usr/local/cuda-12.0
  CUDA_ARCH=80
fi
ROOT=$PWD
RUNTIME="$ROOT/.runtime_ch15_p5_v1"
SOURCE="$ROOT/docs/runtime_sources/seismic_p5_v1"
PY="$RUNTIME/venv/bin/python"
test ! -e "$RUNTIME"
mkdir -p "$RUNTIME/cache" "$RUNTIME/tmp"
export TMPDIR="$RUNTIME/tmp" XDG_CACHE_HOME="$RUNTIME/cache" CUDA_CACHE_PATH="$RUNTIME/cache"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$ROOT/.venv/bin:/snap/bin:$CUDA_ROOT/bin:/usr/bin:/bin"
.venv/bin/python -m venv "$RUNTIME/venv"
.runtime_ch15_comparison_v1/venv/bin/python - <<'PY'
import sysconfig,shutil,json,hashlib
import semgpu2
from pathlib import Path
root=Path.cwd(); runtime=root/'.runtime_ch15_p5_v1'
base=next((root/'.venv/lib').glob('python*/site-packages'))
target=next((runtime/'venv/lib').glob('python*/site-packages'))
(target/'project_dependencies.pth').write_text(str(base)+'\n')
shutil.copytree(Path(semgpu2.__file__).parent,target/'semgpu2',ignore=shutil.ignore_patterns('__pycache__','lin_elast_2d_seismic_newmark_kim_cu','lin_elast_2d_seismic_newmark_kim_p5_cu'))
source=root/'docs/runtime_sources/seismic_p5_v1'
meta=json.loads((source/'provenance.json').read_text())
for path,sha in meta['files_sha256'].items(): assert hashlib.sha256((source/path).read_bytes()).hexdigest()==sha
for name in ('lin_elast_2d_seismic_newmark_kim_cu','lin_elast_2d_seismic_newmark_kim_p5_cu'):
 shutil.copytree(source/'src/semgpu2/solvers'/name,target/'semgpu2/solvers'/name)
PY
cmake -S "$SOURCE" -B "$RUNTIME/build" -G Ninja -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc" -DCUDAToolkit_ROOT="$CUDA_ROOT" -DCMAKE_CXX_COMPILER=/usr/bin/g++ -DPython_EXECUTABLE="$PY" -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build "$RUNTIME/build" --target semgpu2_seismic_cu_native semgpu2_seismic_p5_cu_native --parallel 2
"$PY" - <<'PY'
import sysconfig,shutil,importlib
from pathlib import Path
root=Path.cwd()
for output,name in [('seismic_cu','lin_elast_2d_seismic_newmark_kim_cu'),('seismic_p5_cu','lin_elast_2d_seismic_newmark_kim_p5_cu')]:
 built=list((root/'.runtime_ch15_p5_v1/build'/output).glob('_native*.so'));assert len(built)==1
 shutil.copy2(built[0],Path(sysconfig.get_path('platlib'))/'semgpu2/solvers'/name/built[0].name)
 print('IMPORTED',importlib.import_module('semgpu2.solvers.'+name).SOLVER_ID)
PY
