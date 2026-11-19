#!/usr/bin/env bash
# Banc du MLP natif de GMFlow contre le même MLP sous TensorRT (docs/gmflow-natif.md, « MLP natif (PR 3) »), sur la
# carte : c'est la mesure qui décide si GMFlow natif continue (le seuil est dans la doc ; le script mesure et dit
# chaque ligne face au seuil, sans rien conclure d'autre).
#   flock ~/kirari-reports/gpu-bench.lock tools/gmfss/gmflow_mlp_banc.sh PAIRE_FP16.onnx DOSSIER [ENTREES]
# PAIRE_FP16.onnx : la paire du lecteur (gmfss_fortuna_union_pair.onnx). ENTREES (facultatif) : le dossier qu'écrit
# gmflow_mlp_emule.py --banc, pour contrôler d'abord les sorties de la carte (un noyau faux ne se chronomètre pas).
# Il faut cmake, nvcc, python3 avec onnx, et trtexec (TRTEXEC ou ~/kirari-sdk/tensorrt/bin/trtexec).
#
# Pour chaque charge (paire 1280x768 et 1920x1152, flot entier et demi-flot, le huitième au lot 2 et le quart au lot
# 4) : le MLP de la couche 0 extrait de la paire (entrées x et message, sortie out) construit par trtexec comme le
# lecteur construit la paire (--stronglyTyped, niveau 5, sans cuDNN ni cuBLAS), chronométré en graphe CUDA ; le
# noyau natif aux deux blocs (64 et 128 jetons), chronométré de même par gmflow_mlp_banc. Deux tours ABAB, médiane
# de chaque mesure, moyenne des deux tours.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PAIR="${1:?la paire FP16 du lecteur}"; OUT="${2:?dossier de travail}"; INPUTS="${3:-}"
PY="${PYTHON:-python3}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec || echo "$HOME/kirari-sdk/tensorrt/bin/trtexec")}"
export LD_LIBRARY_PATH="${TRTEXEC%/bin/trtexec}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
LOG="$OUT/banc-mlp.md"
: >"$LOG"
say() { echo "$*" | tee -a "$LOG"; }

cmake -S "$HERE/gmflow_mlp" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release >"$OUT/cmake.log" 2>&1
cmake --build "$OUT/build" >>"$OUT/cmake.log" 2>&1
BANC="$OUT/build/gmflow_mlp_banc"
"$OUT/build/gmflow_mlp_vide" "$PAIR" "$OUT/mlp16" >/dev/null
[ -f "$OUT/mlp.onnx" ] ||
  "$PY" -I "$HERE/gmflow_mlp_emule.py" --fp16 "$PAIR" --plan16 "$OUT/mlp16" --sous-graphe "$OUT/mlp.onnx" >/dev/null

say "# MLP de GMFlow : natif contre TensorRT ($(date -I), $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo carte inconnue))"
say ""
if [ -n "$INPUTS" ]; then
  "$BANC" "$PAIR" --check "$INPUTS" | tee -a "$LOG"
  say ""
fi

# Comme le lecteur construit la paire (prepare/trtexec_command.cpp, Fp32IslandsTuned) ; TensorRT 11 n'a plus
# --tacticSources.
TACTICS=""
"$TRTEXEC" --help 2>&1 | grep -q tacticSources && TACTICS="--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT"

trt_time() { # lot jetons étiquette -> médiane (ms)
  local shapes="x:${1}x${2}x128,message:${1}x${2}x128" eng="$OUT/mlp-$3.engine"
  # shellcheck disable=SC2086
  [ -f "$eng" ] || "$TRTEXEC" --onnx="$OUT/mlp.onnx" --minShapes="$shapes" --optShapes="$shapes" \
    --maxShapes="$shapes" --stronglyTyped --builderOptimizationLevel=5 $TACTICS --saveEngine="$eng" \
    >"$OUT/mlp-$3.build.log" 2>&1 || { echo "construction de $3 échouée ($OUT/mlp-$3.build.log)" >&2; return 1; }
  "$TRTEXEC" --loadEngine="$eng" --shapes="$shapes" --useCudaGraph --noDataTransfers --useSpinWait --warmUp=500 \
    --iterations=300 --dumpProfile --separateProfileRun --exportProfile="$OUT/mlp-$3.profile.json" \
    >"$OUT/mlp-$3.run.log" 2>&1
  sed -n 's/.*GPU Compute Time: min = [0-9.]* ms.*median = \([0-9.]*\) ms.*/\1/p' "$OUT/mlp-$3.run.log"
}
native_time() { # jetons bloc -> médiane (ms)
  "$BANC" "$PAIR" --block "$2" --tokens "$1" --iterations 300 | sed -n 's/^temps [0-9]* \([0-9.]*\) [0-9.]*$/\1/p'
}
mean2() { awk -v a="$1" -v b="$2" 'BEGIN { printf "%.4f", (a + b) / 2 }'; }

# étiquette|lot|jetons par image|seuil du meilleur natif / TensorRT
ROWS=("720p flot entier, 1/8|2|3840|1.00" "720p flot entier, 1/4|4|15360|0.80"
      "720p demi-flot, 1/8|2|960|1.00" "720p demi-flot, 1/4|4|3840|0.80"
      "1080p flot entier, 1/8|2|8640|1.00" "1080p flot entier, 1/4|4|34560|1.00"
      "1080p demi-flot, 1/8|2|2160|1.00" "1080p demi-flot, 1/4|4|8640|1.00")
say "| Charge | jetons | TensorRT (ms) | natif 64 (ms) | natif 128 (ms) | meilleur natif / TensorRT | seuil | tenu |"
say "|---|---:|---:|---:|---:|---:|---:|---|"
declare -A TRT NAT
for row in "${ROWS[@]}"; do
  IFS='|' read -r label lot per seuil <<<"$row"
  tokens=$((lot * per)); tag="${lot}x${per}"
  t1="$(trt_time "$lot" "$per" "$tag")"; a1="$(native_time "$tokens" 64)"; b1="$(native_time "$tokens" 128 || true)"
  t2="$(trt_time "$lot" "$per" "$tag")"; a2="$(native_time "$tokens" 64)"; b2="$(native_time "$tokens" 128 || true)"
  t="$(mean2 "$t1" "$t2")"; a="$(mean2 "$a1" "$a2")"
  b="—"; best="$a"
  if [ -n "$b1" ] && [ -n "$b2" ]; then
    b="$(mean2 "$b1" "$b2")"
    best="$(awk -v a="$a" -v b="$b" 'BEGIN { print (b < a ? b : a) }')"
  fi
  ratio="$(awk -v n="$best" -v t="$t" 'BEGIN { printf "%.3f", n / t }')"
  held="$(awk -v r="$ratio" -v s="$seuil" 'BEGIN { print (r <= s ? "oui" : "non") }')"
  TRT[$label]="$t"; NAT[$label]="$best"
  say "| $label | $tokens | $t | $a | $b | $ratio | $seuil | $held |"
done
say ""
say "Six couches aux deux échelles, par paire (6 × (1/8 + 1/4), meilleur bloc) :"
say "| Charge | TensorRT (ms) | natif (ms) | gain (ms) |"
say "|---|---:|---:|---:|"
for load in "720p flot entier" "720p demi-flot" "1080p flot entier" "1080p demi-flot"; do
  t="$(awk -v a="${TRT[$load, 1/8]}" -v b="${TRT[$load, 1/4]}" 'BEGIN { printf "%.3f", 6 * (a + b) }')"
  n="$(awk -v a="${NAT[$load, 1/8]}" -v b="${NAT[$load, 1/4]}" 'BEGIN { printf "%.3f", 6 * (a + b) }')"
  say "| $load | $t | $n | $(awk -v t="$t" -v n="$n" 'BEGIN { printf "%.3f", t - n }') |"
done
say ""
say "Profils TensorRT par couche : $OUT/mlp-*.profile.json (une seule région Myelin = le MLP fondu par TensorRT)."
