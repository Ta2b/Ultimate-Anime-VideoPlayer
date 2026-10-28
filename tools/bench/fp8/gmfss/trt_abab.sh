#!/bin/bash
# Mesure TensorRT FP16 en ABAB de plusieurs graphes ONNX (même méthode que les autres mesures).
# Usage : trt_abab.sh TOURS graphe1.onnx graphe2.onnx ...   (à lancer sous ~/ajp-tools/bin/ajp-gpu)
# Sortie : médiane et minimum du temps de calcul GPU par graphe et par tour.
set -e
export LD_LIBRARY_PATH="$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64"
TRTEXEC="$HOME/ajp-sdk/tensorrt/bin/trtexec"
TOURS="$1"
# TRT_EXTRA : options de construction supplémentaires (ex. --inputIOFormats=fp16:hwc8,fp16:hwc8)
shift
DOSSIER="${ABAB_DIR:-/tmp/claude-1000/gmfss-abab}"
mkdir -p "$DOSSIER"
# Un moteur par graphe, construit une fois (niveau 5), puis rechargé à chaque tour.
for g in "$@"; do
  e="$DOSSIER/$(basename "$g").engine"
  [ -f "$e" ] || "$TRTEXEC" --onnx="$g" --builderOptimizationLevel=5 $TRT_EXTRA --saveEngine="$e" > /dev/null 2>&1
done
for tour in $(seq 1 "$TOURS"); do
  for g in "$@"; do
    e="$DOSSIER/$(basename "$g").engine"
    r=$("$TRTEXEC" --loadEngine="$e" --useCudaGraph --warmUp=800 --iterations=300 --noDataTransfers 2>&1 |
      grep -E "GPU Compute Time: min" | tail -1)
    echo "tour $tour $(basename "$g") : $(echo "$r" | sed -E 's/.*min = ([0-9.]+) ms.*median = ([0-9.]+) ms.*/min \1 médiane \2/')"
  done
done
