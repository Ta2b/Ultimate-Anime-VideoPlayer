#!/usr/bin/env bash
# Compile le banc du noyau et le plugin TensorRT (hors CMake du lecteur). Sortie : $OUT (défaut /tmp/claude-1000/fp8plug).
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-/tmp/claude-1000/fp8plug}
TRT=${TRT:-$HOME/ajp-sdk/tensorrt}
NVCC=${NVCC:-/opt/cuda/bin/nvcc}
KERN=../../../../src/gpu/cuda/backend/fp8_fused.cu
mkdir -p "$OUT"
"$NVCC" -O3 -std=c++20 -arch=sm_89 -o "$OUT/bench_kernel" bench_kernel.cu "$KERN"
"$NVCC" -O3 -std=c++20 -arch=sm_89 -shared -Xcompiler -fPIC -I"$TRT/include" -I/opt/cuda/include \
  -o "$OUT/libajp_span_tail_plugin.so" span_tail_plugin.cpp "$KERN" -L"$TRT/lib" -lnvinfer -L/opt/cuda/lib64 -lcudart
echo "ok : $OUT"
