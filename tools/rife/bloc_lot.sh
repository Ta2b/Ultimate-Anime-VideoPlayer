#!/usr/bin/env bash
# Banc des instants groupés (chantier 2 du fil RIFE) : chaque morceau de bloc de RIFE fusionné en lot de N instants
# contre le même morceau en lot de 1, sur trtexec (graphe CUDA, niveau 5).
#   flock ~/ajp-reports/gpu-bench.lock tools/rife/bloc_lot.sh DOSSIER_MORCEAUX [N=8] [tours=3] [filtre=768x1280]
# DOSSIER_MORCEAUX : le dossier « modeles » du lecteur, où derive_rife_fused a écrit les « *-blocK.onnx ». Le
# filtre garde ceux d'un modèle et d'une taille (768x1280 : 1080p vers 4K ; « rife_v4.22_lite-fusion3-768x1280 »
# si plusieurs modèles y sont). Pour chaque bloc, abab_trt.sh mesure A (lot
# 1) et B (lot N) ; « lot / N seules » est la médiane de B sur N fois celle de A : sous 100 %, grouper les instants
# gagne ce qui manque. La somme des blocs donne le gain d'une paire forte, colle et fin à part.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
DIR="${1:?dossier des morceaux}"; N="${2:-8}"; ROUNDS="${3:-3}"; FILTRE="${4:-768x1280}"
PY="${PYTHON:-python3}"
WORK="${BLOC_LOT_WORK:-$(mktemp -d)}"
mkdir -p "$WORK"
shopt -s nullglob
pieces=("$DIR"/*"$FILTRE"*-bloc[0-9].onnx)
[ ${#pieces[@]} -gt 0 ] || { echo "aucun « *$FILTRE*-blocK.onnx » dans $DIR" >&2; exit 1; }
printf 'morceaux : %s\n' "${pieces[@]##*/}"
total1=0; totaln=0
for f in "${pieces[@]}"; do
  b="$(basename "$f" .onnx)"
  "$PY" "$HERE/bloc_lot.py" "$f" "$WORK/$b-lot$N.onnx" "$N" || { echo "$b : pas de lot possible" >&2; exit 1; }
  echo "== $b"
  out="$(ABAB_CACHE="$WORK/cache-$b" "$HERE/../fp8/abab_trt.sh" "$f" "$WORK/$b-lot$N.onnx" static "$ROUNDS" |
    grep -v '^moteurs')"
  echo "$out"
  # La dernière ligne : « tour A(min/méd) B(min/méd) écart ».
  read -r a b_med < <(echo "$out" | tail -1 | awk '{split($2,a,"/");split($3,b,"/");print a[2], b[2]}')
  awk -v a="$a" -v b="$b_med" -v n="$N" 'BEGIN{printf "   lot de %d : %.3f ms contre %d × %.3f = %.3f ms, lot / N seules = %.0f %%\n", n, b, n, a, n*a, 100*b/(n*a)}'
  total1=$(awk -v t="$total1" -v a="$a" 'BEGIN{print t+a}')
  totaln=$(awk -v t="$totaln" -v b="$b_med" 'BEGIN{print t+b}')
done
awk -v a="$total1" -v b="$totaln" -v n="$N" 'BEGIN{printf "== blocs ensemble : lot de %d %.3f ms contre %.3f ms en passes seules, lot / N seules = %.0f %%\n", n, b, n*a, 100*b/(n*a)}'
echo "morceaux en lot et moteurs : $WORK"
