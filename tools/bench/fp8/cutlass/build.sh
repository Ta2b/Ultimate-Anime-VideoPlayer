#!/bin/sh
# Compile un prototype CUTLASS : build.sh SOURCE.cu SORTIE
# CUTLASS (BSD-3, header-only) est lu depuis un dépôt externe, jamais embarqué dans le lecteur.
set -e
CUTLASS="${CUTLASS:-$HOME/ajp-work/cutlass-jetable}"
CUDA="${CUDA:-/opt/cuda}"
exec nice "$CUDA/bin/nvcc" -std=c++17 -O3 -arch=sm_89 --expt-relaxed-constexpr \
  -I"$CUTLASS/include" -I"$CUTLASS/tools/util/include" \
  -Xcudafe --diag_suppress=177 -Xcudafe --diag_suppress=550 \
  "$1" -o "$2"
