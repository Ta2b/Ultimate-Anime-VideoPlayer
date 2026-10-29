#!/bin/sh
# Construit (trtexec --saveEngine, formes fixes, types ONNX = FP16) et mesure (graphe CUDA) les 4 petits moteurs
# (encodeur en lot de 3, blocs 0 à 2) et le réseau entier (modèle d'origine du lecteur).
# Usage : moteurs.sh <dossier_onnx_decoupes> <dossier_moteurs> <modele_entier.onnx>
# Les mesures sortent en « GPU Compute Time » ; tout passe sous le verrou GPU.
export LD_LIBRARY_PATH=$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64
TRTEXEC=$HOME/ajp-sdk/tensorrt/bin/trtexec
GPU=$HOME/ajp-tools/bin/ajp-gpu
src=$1
dst=$2
entier=$3
mkdir -p "$dst"
for m in enc bloc0 bloc1 bloc2 plie; do
  echo "== $m"
  nice -n 19 "$GPU" "$TRTEXEC" --onnx="$src/$m.onnx" --saveEngine="$dst/$m.engine" \
    --timingCacheFile="$dst/cache.bin" --warmUp=500 --iterations=200 --duration=2 --avgRuns=1 \
    --useCudaGraph --noDataTransfers 2>&1 | grep -E "GPU Compute Time|Error|error" | head -3
done
echo "== entier (modèle du lecteur)"
nice -n 19 "$GPU" "$TRTEXEC" --onnx="$entier" --saveEngine="$dst/entier.engine" \
  --timingCacheFile="$dst/cache.bin" --warmUp=500 --iterations=200 --duration=2 --avgRuns=1 \
  --useCudaGraph --noDataTransfers 2>&1 | grep -E "GPU Compute Time|Error|error" | head -3
