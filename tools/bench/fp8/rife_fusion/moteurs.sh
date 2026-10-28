#!/bin/sh
# Construit (trtexec --saveEngine, formes fixes, FP16 selon les types ONNX) et mesure (graphe CUDA) les
# 6 petits moteurs et le reseau entier. Usage : moteurs.sh <dossier_onnx_decoupes> <dossier_moteurs>
# Les mesures sortent en "GPU Compute Time" (mediane) ; tout passe sous le verrou GPU.
export LD_LIBRARY_PATH=$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64
TRTEXEC=$HOME/ajp-sdk/tensorrt/bin/trtexec
GPU=$HOME/ajp-tools/bin/ajp-gpu
src=$1
dst=$2
mkdir -p "$dst"
for m in enc bloc0 bloc1 bloc2 bloc3 bloc4 rife_fixe; do
  echo "== $m"
  nice -n 19 "$GPU" "$TRTEXEC" --onnx="$src/$m.onnx" --saveEngine="$dst/$m.engine" \
    --timingCacheFile="$dst/cache.bin" --warmUp=500 --iterations=200 --duration=2 --avgRuns=1 \
    --useCudaGraph --noDataTransfers 2>&1 | grep -E "GPU Compute Time|Error|error" | head -3
done
