#!/usr/bin/env bash
set -euo pipefail
cd /home/antonov/SeismicDataset
mkdir -p .runtime_ch16/proximity_tmp .runtime_ch16/proximity_cache .runtime_ch16/logs
export TMPDIR="$PWD/.runtime_ch16/proximity_tmp" XDG_CACHE_HOME="$PWD/.runtime_ch16/proximity_cache" MPLCONFIGDIR="$PWD/.runtime_ch16/proximity_cache/matplotlib" CUDA_CACHE_PATH="$PWD/.runtime_ch16/proximity_cache/cuda"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=0
PY="$PWD/.runtime_ch15_p67s_v1/venv/bin/python"
"$PY" -m scripts.dev_1_6_p67s_showcase.proximity_run --prepare
"$PY" -m scripts.dev_1_6_p67s_showcase.proximity_run --series
