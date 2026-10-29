#!/usr/bin/env bash
# ABAB trtexec (graphe CUDA) entre deux graphes ONNX de même interface : A (référence FP16) contre B (FP8).
#   flock ~/ajp-reports/gpu-bench.lock tools/fp8/abab_trt.sh A.onnx B.onnx "entree:1x11x768x1280[;autre:...]"|static [tours=3]
# Construit chaque moteur une fois (niveau 5), puis alterne A B A B... : minimum / médiane GPU en ms.
# TensorRT 11 est fortement typé : pas de --fp16 / --fp8, la précision vient des types du graphe.
set -euo pipefail
A="$1"; B="$2"; SHAPES="$3"; ROUNDS="${4:-3}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec || echo "$HOME/ajp-sdk/tensorrt/bin/trtexec")}"
export LD_LIBRARY_PATH="${TRTEXEC%/bin/trtexec}/lib:${LD_LIBRARY_PATH:-}"
CACHE="${ABAB_CACHE:-$(mktemp -d)}"
mkdir -p "$CACHE"
mins=(); opts=(); maxs=()
if [ "$SHAPES" != static ]; then  # « static » : modèle à formes fixes (les morceaux), trtexec refuse alors les formes
  joined="${SHAPES//;/,}"  # plusieurs entrées : une seule option, séparées par des virgules
  mins=("--minShapes=$joined"); opts=("--optShapes=$joined"); maxs=("--maxShapes=$joined")
fi

build() { # onnx étiquette
  local eng="$CACHE/$2.engine"
  [ -f "$eng" ] && return 0
  "$TRTEXEC" --onnx="$1" "${mins[@]}" "${opts[@]}" "${maxs[@]}" --builderOptimizationLevel=5 \
    --saveEngine="$eng" >"$CACHE/$2.build.log" 2>&1 \
    || { echo "construction de $2 échouée ($CACHE/$2.build.log)" >&2; tail -5 "$CACHE/$2.build.log" >&2; exit 1; }
}
run() { # étiquette -> « min médiane »
  "$TRTEXEC" --loadEngine="$CACHE/$1.engine" "${opts[@]}" --useCudaGraph --noDataTransfers --warmUp=500 \
    --iterations=300 2>&1 | sed -n 's/.*GPU Compute Time: min = \([0-9.]*\) ms.*median = \([0-9.]*\) ms.*/\1 \2/p'
}
build "$A" A
build "$B" B
echo "tour  A(min/méd)  B(min/méd)  écart médiane B contre A"
for r in $(seq 1 "$ROUNDS"); do
  a=$(run A); b=$(run B)
  awk -v r="$r" -v a="$a" -v b="$b" 'BEGIN{split(a,x," ");split(b,y," ");printf "%s  %s/%s  %s/%s  %+.1f%%\n", r,x[1],x[2],y[1],y[2],(y[2]/x[2]-1)*100}'
done
echo "moteurs : $CACHE"
