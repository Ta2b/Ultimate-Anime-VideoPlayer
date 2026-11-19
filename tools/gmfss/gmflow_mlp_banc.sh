#!/usr/bin/env bash
# Banc du MLP natif de GMFlow contre le même MLP sous TensorRT (docs/gmflow-natif.md, « MLP natif (PR 3) »), sur la
# carte : c'est la mesure qui décide si GMFlow natif continue (le seuil est dans la doc ; le script mesure et dit
# chaque ligne face au seuil, sans rien conclure d'autre).
#   flock ~/kirari-reports/gpu-bench.lock tools/gmfss/gmflow_mlp_banc.sh PAIRE_FP16.onnx DOSSIER [ENTREES]
# PAIRE_FP16.onnx : la paire du lecteur (gmfss_fortuna_union_pair.onnx). ENTREES (facultatif) : le dossier qu'écrit
# gmflow_mlp_emule.py --banc, pour contrôler d'abord les sorties de la carte : un contrôle non tenu arrête le banc
# (un noyau faux ne se chronomètre pas). Il faut cmake, nvcc, python3 avec onnx (PYTHON), et trtexec (TRTEXEC ou
# ~/kirari-sdk/tensorrt/bin/trtexec).
#
# Pour chaque charge (paire 1280x768 et 1920x1152, flot entier et demi-flot, le huitième au lot 2 et le quart au lot
# 4) : le MLP de la couche 0 extrait de la paire (entrées x et message, sortie out) construit par trtexec aux deux
# recettes de gmflow_banc_trt.sh (celle de la paire dans le lecteur, et la réglée) et chronométré en graphe CUDA ; le
# noyau natif aux deux blocs (64 et 128 jetons), chronométré de même par gmflow_mlp_banc. Deux tours ABAB (TensorRT,
# natif, TensorRT, natif), médiane de chaque mesure, moyenne des deux tours ; le natif face au meilleur TensorRT.
set -euo pipefail
shopt -s inherit_errexit
HERE="$(cd "$(dirname "$0")" && pwd)"
PAIR="${1:?la paire FP16 du lecteur}"; OUT="${2:?dossier de travail}"; INPUTS="${3:-}"
PY="${PYTHON:-python3}"
# shellcheck source=gmflow_banc_trt.sh
. "$HERE/gmflow_banc_trt.sh"
mkdir -p "$OUT/trt"
LOG="$OUT/banc-mlp.md"
: >"$LOG"
say() { echo "$*" | tee -a "$LOG"; }

cmake -S "$HERE/gmflow_mlp" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release >"$OUT/cmake.log" 2>&1
cmake --build "$OUT/build" >>"$OUT/cmake.log" 2>&1
BANC="$OUT/build/gmflow_mlp_banc"
[ -x "$BANC" ] || { echo "gmflow_mlp_banc absent : nvcc introuvable ($OUT/cmake.log)" >&2; exit 1; }
"$OUT/build/gmflow_mlp_vide" "$PAIR" "$OUT/mlp16" >/dev/null
[ -f "$OUT/mlp.onnx" ] ||
  "$PY" -I "$HERE/gmflow_mlp_emule.py" --fp16 "$PAIR" --plan16 "$OUT/mlp16" --sous-graphe "$OUT/mlp.onnx" >/dev/null

say "# MLP de GMFlow : natif contre TensorRT ($(date -I), $(bench_header))"
say ""
if [ -n "$INPUTS" ]; then
  "$BANC" "$PAIR" --check "$INPUTS" | tee -a "$LOG"
  say ""
fi

native_time() { # jetons bloc -> médiane (ms) ; rien si le bloc ne tourne pas sur cette carte (128 sur Turing)
  local text
  text="$("$BANC" "$PAIR" --block "$2" --tokens "$1" --iterations 300)" || return 1
  sed -n 's/^temps [0-9]* \([0-9.]*\) [0-9.]*$/\1/p' <<<"$text"
}
trt_pair() { # lot jetons -> « lecteur réglée » (ms)
  local shapes="x:${1}x${2}x128,message:${1}x${2}x128" r v times=()
  for r in "${RECIPES[@]}"; do
    v="$(trt_time "$OUT/mlp.onnx" "$shapes" "mlp-${1}x${2}" "$r" "$OUT/trt")"
    times+=("$v")
  done
  echo "${times[*]}"
}

# étiquette|lot|jetons par image|seuil du meilleur natif / meilleur TensorRT
ROWS=("720p flot entier, 1/8|2|3840|1.00" "720p flot entier, 1/4|4|15360|0.80"
      "720p demi-flot, 1/8|2|960|1.00" "720p demi-flot, 1/4|4|3840|0.80"
      "1080p flot entier, 1/8|2|8640|1.00" "1080p flot entier, 1/4|4|34560|1.00"
      "1080p demi-flot, 1/8|2|2160|1.00" "1080p demi-flot, 1/4|4|8640|1.00")
say "| Charge | jetons | TensorRT lecteur (ms) | TensorRT réglé (ms) | natif 64 (ms) | natif 128 (ms) |" \
  "meilleur natif / meilleur TensorRT | seuil | tenu |"
say "|---|---:|---:|---:|---:|---:|---:|---:|---|"
declare -A TRT NAT
for row in "${ROWS[@]}"; do
  IFS='|' read -r label lot per seuil <<<"$row"
  tokens=$((lot * per))
  p1="$(trt_pair "$lot" "$per")"; a1="$(native_time "$tokens" 64)"; b1="$(native_time "$tokens" 128 || true)"
  p2="$(trt_pair "$lot" "$per")"; a2="$(native_time "$tokens" 64)"; b2="$(native_time "$tokens" 128 || true)"
  [ -n "$a1" ] && [ -n "$a2" ] || { echo "$label : pas de temps natif" >&2; exit 1; }
  read -r l1 r1 <<<"$p1"; read -r l2 r2 <<<"$p2"
  l="$(mean2 "$l1" "$l2")"; r="$(mean2 "$r1" "$r2")"; a="$(mean2 "$a1" "$a2")"
  b="—"; best="$a"
  if [ -n "$b1" ] && [ -n "$b2" ]; then
    b="$(mean2 "$b1" "$b2")"; best="$(min2 "$a" "$b")"
  fi
  t="$(min2 "$l" "$r")"; q="$(ratio "$best" "$t")"
  TRT[$label]="$t"; NAT[$label]="$best"
  say "| $label | $tokens | $l | $r | $a | $b | $q | $seuil | $(held "$q" "$seuil") |"
done
say ""
say "Six couches aux deux échelles, par paire (6 × (1/8 + 1/4), meilleur bloc natif, meilleur TensorRT) :"
say "| Charge | TensorRT (ms) | natif (ms) | gain (ms) |"
say "|---|---:|---:|---:|"
for load in "720p flot entier" "720p demi-flot" "1080p flot entier" "1080p demi-flot"; do
  t="$(awk -v a="${TRT[$load, 1/8]}" -v b="${TRT[$load, 1/4]}" 'BEGIN { printf "%.3f", 6 * (a + b) }')"
  n="$(awk -v a="${NAT[$load, 1/8]}" -v b="${NAT[$load, 1/4]}" 'BEGIN { printf "%.3f", 6 * (a + b) }')"
  say "| $load | $t | $n | $(awk -v t="$t" -v n="$n" 'BEGIN { printf "%.3f", t - n }') |"
done
say ""
say "Profils TensorRT par couche : $OUT/trt/mlp-*.profile.json (une seule région Myelin = le MLP fondu par TensorRT)."
