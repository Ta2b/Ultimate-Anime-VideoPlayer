#!/usr/bin/env bash
# Mesure ABAB : construit (une fois) puis alterne N tours de trtexec sur chaque variante.
# Usage : ab.sh DOSSIER TOURS nom=modele.onnx[,nom2=modele2.onnx ...]
# Chaque appel GPU passe par ajp-gpu (verrou) ; un seul à la fois. Médiane GPU (ms) par tour.
set -euo pipefail
export PATH=$HOME/ajp-tools/bin:$PATH
export LD_LIBRARY_PATH=$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64
TRTEXEC=$HOME/ajp-sdk/tensorrt/bin/trtexec
PLUG=${PLUG:-/tmp/claude-1000/fp8plug/libajp_span_tail_plugin.so}
HW=${HW:-1088x1920}; H=${HW%x*}; W=${HW#*x}
dir=$1; rounds=$2; IFS=, read -ra variants <<<"$3"
mkdir -p "$dir"
shape="--minShapes=input:1x3x${H}x${W} --optShapes=input:1x3x${H}x${W} --maxShapes=input:1x3x${H}x${W}"
for v in "${variants[@]}"; do
  n=${v%%=*}; f=${v#*=}
  if [ ! -f "$dir/$n.engine" ]; then
    echo "== construction $n"
    ajp-gpu "$TRTEXEC" --onnx="$f" $shape --staticPlugins="$PLUG" --builderOptimizationLevel=4 \
      --saveEngine="$dir/$n.engine" > "$dir/build_$n.log" 2>&1 || { tail -20 "$dir/build_$n.log"; exit 1; }
  fi
done
for r in $(seq 1 "$rounds"); do
  for v in "${variants[@]}"; do
    n=${v%%=*}
    ajp-gpu "$TRTEXEC" --loadEngine="$dir/$n.engine" $shape --staticPlugins="$PLUG" --useCudaGraph --warmUp=800 \
      --iterations=300 --noDataTransfers > "$dir/run_${n}_$r.log" 2>&1
    echo "tour $r $n : $(grep -E 'GPU Compute Time.*median' "$dir/run_${n}_$r.log" | sed 's/.*min/min/')"
  done
done
