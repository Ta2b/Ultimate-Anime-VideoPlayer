#!/usr/bin/env bash
# ABAB des morceaux d'un réseau fusionné (RIFE, DistilDRBA) : chaque morceau FP16 contre son jumeau FP8.
#   flock ~/ajp-reports/gpu-bench.lock tools/fp8/abab_pieces.sh DOSSIER_FP16 DOSSIER_FP8 [tours=3]
# Les deux dossiers portent les mêmes noms de fichiers « *-bloc*.onnx » (interp_fp8.py quantize les a écrits). Les
# morceaux sont mesurés un par un puis additionnés : c'est la somme des moteurs du lecteur fusionné.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
D16="$1"; D8="$2"; ROUNDS="${3:-3}"
for f in "$D8"/*-bloc*.onnx; do
  b="$(basename "$f")"
  echo "== $b"
  ABAB_CACHE="${ABAB_CACHE_ROOT:-/tmp/abab}/${b%.onnx}" "$HERE/abab_trt.sh" "$D16/$b" "$f" static "$ROUNDS" | grep -v '^moteurs'
done
