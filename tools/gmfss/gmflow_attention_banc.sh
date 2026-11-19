#!/usr/bin/env bash
# Banc de l'attention native de GMFlow contre la même couche sous TensorRT (docs/gmflow-natif.md, « Attention native
# (PR 2) »), sur la carte. Le script mesure et dit chaque ligne face au seuil de la doc, sans rien conclure d'autre.
#   flock ~/kirari-reports/gpu-bench.lock tools/gmfss/gmflow_attention_banc.sh PAIRE_FP16.onnx DOSSIER [CONTROLE]
# PAIRE_FP16.onnx : la paire du lecteur (gmfss_fortuna_union_pair.onnx). CONTROLE (facultatif) : le dossier qu'écrit
# gmflow_attention_emule.py --dump, pour contrôler d'abord les sorties de la carte : un contrôle non tenu arrête le
# banc (un noyau faux ne se chronomètre pas). Il faut cmake, nvcc, python3 avec torch et onnxruntime (PYTHON), et
# trtexec (TRTEXEC ou ~/kirari-sdk/tensorrt/bin/trtexec). COUCHES (« 0 1 2 3 » par défaut) restreint les couches.
#
# Pour chaque charge (GMFlow des paires 1280x768 et 1920x1152, flot entier et demi-flot ; le huitième au lot 2, le
# quart au lot 4) et chaque couche (0 self, 1 cross, 2 self décalée, 3 cross décalée : les douze couches du
# transformeur ne sont que ces quatre-là, aux poids près) : la couche seule en ONNX à taille fixe
# (gmflow_attention_onnx.py, vérifiée sous ONNX Runtime) construite par trtexec aux deux recettes de
# gmflow_banc_trt.sh et chronométrée en graphe CUDA ; la couche native chronométrée de même par
# gmflow_attention_banc. Deux tours ABAB (TensorRT, natif, TensorRT, natif), médiane de chaque mesure, moyenne des
# deux tours ; le natif face au meilleur TensorRT. Puis, par paire, les douze couches aux deux échelles :
# 3 × (self + cross + self décalée + cross décalée) par échelle, et le quart seul, que la décision lit avec le MLP.
set -euo pipefail
shopt -s inherit_errexit
HERE="$(cd "$(dirname "$0")" && pwd)"
PAIR="${1:?la paire FP16 du lecteur}"; OUT="${2:?dossier de travail}"; CONTROL="${3:-}"
PY="${PYTHON:-python3}"
read -r -a LAYERS <<<"${COUCHES:-0 1 2 3}"
# shellcheck source=gmflow_banc_trt.sh
. "$HERE/gmflow_banc_trt.sh"
mkdir -p "$OUT/onnx" "$OUT/trt"
LOG="$OUT/banc-attention.md"
: >"$LOG"
say() { echo "$*" | tee -a "$LOG"; }

cmake -S "$HERE/gmflow_attention" -B "$OUT/build" -DCMAKE_BUILD_TYPE=Release -DKIRARI_BANC_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native >"$OUT/cmake.log" 2>&1
cmake --build "$OUT/build" >>"$OUT/cmake.log" 2>&1
BANC="$OUT/build/gmflow_attention_banc"
"$OUT/build/gmflow_attention_vide" "$PAIR" "$OUT/attn16" >/dev/null

say "# Attention de GMFlow : natif contre TensorRT ($(date -I), $(bench_header))"
say ""
if [ -n "$CONTROL" ]; then
  "$BANC" "$PAIR" --check "$CONTROL" | tee -a "$LOG"
  say ""
fi

# étiquette|échelle|hauteur|largeur. Seuil de chaque couche : le natif ne perd nulle part (≤ 1,00 × le meilleur
# TensorRT) ; la décision, sur la couche entière du transformeur au quart de 720p (attention et MLP, ≤ 0,80), se
# fait avec le rapport du MLP (docs/gmflow-natif.md, « Séance sur le PC »).
ROWS=("720p flot entier|0|48|80" "720p flot entier|1|96|160" "720p demi-flot|0|24|40" "720p demi-flot|1|48|80"
      "1080p flot entier|0|72|120" "1080p flot entier|1|144|240" "1080p demi-flot|0|36|60" "1080p demi-flot|1|72|120")
SEUIL=1.00

layer_onnx() { # échelle couche hauteur largeur -> la couche seule en ONNX, écrite une fois
  local f="$OUT/onnx/e$1_c$2_$3x$4.onnx"
  [ -f "$f" ] || "$PY" -I "$HERE/gmflow_attention_onnx.py" --plan "$OUT/attn16" --echelle "$1" --couche "$2" \
    --hauteur "$3" --largeur "$4" --sortie "$f" >>"$OUT/onnx/export.log" 2>&1
  echo "$f"
}
native_time() { # échelle couche hauteur largeur -> médiane (ms)
  local text
  text="$("$BANC" "$PAIR" --scale "$1" --layer "$2" --grid "$3x$4" --iterations 300)"
  sed -n 's/^temps [0-9]* [0-9]* \([0-9.]*\) [0-9.]*$/\1/p' <<<"$text"
}
trt_pair() { # ONNX étiquette -> « lecteur réglée » (ms)
  local r v times=()
  for r in "${RECIPES[@]}"; do
    v="$(trt_time "$1" "" "$2" "$r" "$OUT/trt")"
    times+=("$v")
  done
  echo "${times[*]}"
}

say "| Charge | échelle | couche | TensorRT lecteur (ms) | TensorRT réglé (ms) | natif (ms) | natif / meilleur TensorRT |" \
  "seuil | tenu |"
say "|---|---|---|---:|---:|---:|---:|---:|---|"
declare -A TRT NAT
for row in "${ROWS[@]}"; do
  IFS='|' read -r label scale h w <<<"$row"
  for c in "${LAYERS[@]}"; do
    onnx="$(layer_onnx "$scale" "$c" "$h" "$w")"; tag="attn-e${scale}-c${c}-${h}x${w}"
    p1="$(trt_pair "$onnx" "$tag")"; n1="$(native_time "$scale" "$c" "$h" "$w")"
    p2="$(trt_pair "$onnx" "$tag")"; n2="$(native_time "$scale" "$c" "$h" "$w")"
    [ -n "$n1" ] && [ -n "$n2" ] || { echo "$label, couche $c : pas de temps natif" >&2; exit 1; }
    read -r l1 r1 <<<"$p1"; read -r l2 r2 <<<"$p2"
    l="$(mean2 "$l1" "$l2")"; r="$(mean2 "$r1" "$r2")"; n="$(mean2 "$n1" "$n2")"
    t="$(min2 "$l" "$r")"; q="$(ratio "$n" "$t")"
    TRT[$label,$scale,$c]="$t"; NAT[$label,$scale,$c]="$n"
    say "| $label | 1/$((scale == 0 ? 8 : 4)) | $c | $l | $r | $n | $q | $SEUIL | $(held "$q" "$SEUIL") |"
  done
done

if [ "${#LAYERS[@]}" -eq 4 ]; then
  say ""
  say "Les douze couches d'attention par paire, 3 × (couches 0 + 1 + 2 + 3) à chaque échelle (meilleur TensorRT) :"
  say "| Charge | TensorRT 1/8 (ms) | natif 1/8 (ms) | TensorRT 1/4 (ms) | natif 1/4 (ms) | TensorRT (ms) | natif (ms) | gain (ms) |"
  say "|---|---:|---:|---:|---:|---:|---:|---:|"
  for load in "720p flot entier" "720p demi-flot" "1080p flot entier" "1080p demi-flot"; do
    sums=()
    for scale in 0 1; do
      for m in TRT NAT; do
        declare -n map="$m"
        sums+=("$(awk -v a="${map[$load,$scale,0]}" -v b="${map[$load,$scale,1]}" -v c="${map[$load,$scale,2]}" \
          -v d="${map[$load,$scale,3]}" 'BEGIN { printf "%.3f", 3 * (a + b + c + d) }')")
        unset -n map
      done
    done
    t="$(awk -v a="${sums[0]}" -v b="${sums[2]}" 'BEGIN { printf "%.3f", a + b }')"
    n="$(awk -v a="${sums[1]}" -v b="${sums[3]}" 'BEGIN { printf "%.3f", a + b }')"
    say "| $load | ${sums[0]} | ${sums[1]} | ${sums[2]} | ${sums[3]} | $t | $n |" \
      "$(awk -v t="$t" -v n="$n" 'BEGIN { printf "%.3f", t - n }') |"
  done
fi
say ""
say "Profils TensorRT par couche : $OUT/trt/attn-*.profile.json ; ONNX des couches et leur vérification : $OUT/onnx."
