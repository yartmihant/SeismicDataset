#!/usr/bin/env bash
set -euo pipefail
cd /home/antonov/SeismicDataset
export TMPDIR="$PWD/.runtime_ch16/tmp" XDG_CACHE_HOME="$PWD/.runtime_ch16/cache" MPLCONFIGDIR="$PWD/.runtime_ch16/cache/matplotlib"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY="$PWD/.runtime_ch15_p67s_v1/venv/bin/python"
"$PY" -m scripts.dev_1_6_p67s_showcase.check
"$PY" -m scripts.dev_1_6_p67s_showcase.render --scales
pending=""
for model in homogeneous two_layer EFVB00000 EFVB00001 ECVB12404 ECVB00065; do
  "$PY" -m scripts.dev_1_6_p67s_showcase.render --case "$model" > ".runtime_ch16/logs/render_${model}.log" 2>&1 &
  current=$!
  if [[ -n "$pending" ]]; then
    wait "$pending"
    wait "$current"
    pending=""
  else
    pending=$current
  fi
done
"$PY" -m scripts.dev_1_6_p67s_showcase.render
