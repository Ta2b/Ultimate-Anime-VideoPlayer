#!/usr/bin/env bash
# Produit les graphes FP8 de l'interpolation (docs/fp8-interpolation.md) depuis une vidéo d'anime :
#   tools/fp8/build_interp_fp8.sh VIDEO.mkv SORTIE_DIR
# Variables : RIFE_DIR (rife_v4.25.onnx, rife_v4.22_lite.onnx), INTERP_DIR (distil_drba_v2_lite*.onnx,
#   gmfss_fortuna_{base,union}_s05_pair.onnx), PYTHON (python3 avec numpy, onnx, onnxruntime, ml_dtypes, pillow).
# Étapes : 9 instants de la vidéo en 1280x720 (10 images consécutives chacun, les 2 couples qui bougent le plus) ; entrées bien formées (RIFE :
# 11 plans avec grille ; DRBA : pair01/pair12 ; GMFSS : 6 plans) ; calibration sur 4 instants (maximum) ;
# gardes de qualité sur 7 cas HORS calibration (PSNR moyen >= 42 dB, pire >= 38 dB, SSIM moyen >= 0,985).
# Sortie : SORTIE_DIR/models/{distil_drba_v2_lite_fp8,distil_drba_v2_lite_s05_fp8,gmfss_fortuna_base_s05fp8_pair,
#   gmfss_fortuna_union_s05fp8_pair}.onnx, quality.txt et SHA256SUMS.fp8 ; rejected/ : RIFE (gardes non tenues).
# GMFSS : la paire seule, sans le squelette de GMFlow (/flownet/backbone/, laissé en FP16 : FP8 dedans, le pire cas
# tombe à 33 dB) ; la fusion reste FP16 (FP8 dedans : 36,9 dB au pire cas avec union).
# Les durées (ABAB) se mesurent à part : tools/fp8/abab_trt.sh (docs/fp8-interpolation.md).
set -euo pipefail
VIDEO="$1"; OUT="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PYTHON:-python3}"
RIFE_DIR="${RIFE_DIR:-$HOME/.local/share/ajp/ajp/rife}"
INTERP_DIR="${INTERP_DIR:-$RIFE_DIR}"
TOOL="$HERE/interp_fp8.py"
mkdir -p "$OUT/frames" "$OUT/in" "$OUT/models" "$OUT/rejected"

for t in 95 240 420 610 800 960 1100 1230 1330; do
  [ -f "$OUT/frames/t${t}_9.png" ] || ffmpeg -v error -y -ss "$t" -i "$VIDEO" -vf "scale=1280:720:flags=lanczos" \
    -frames:v 10 -start_number 0 "$OUT/frames/t${t}_%d.png"
done
for kind in rife drba gmfss-pair; do
  [ -d "$OUT/in/$kind" ] || "$PY" "$TOOL" inputs "$kind" x "$OUT/frames" "$OUT/in/$kind" --size 768x1280 >/dev/null
done
CAL=(t95_0 t420_0 t800_0 t1100_0)
TEST=(t240_0 t610_0 t960_0 t1230_0 t1330_0 t800_1 t1100_1)
files() { local d="$1"; shift; for k in "$@"; do printf '%s/%s.npz ' "$d" "$k"; done; }

: >"$OUT/quality.txt"
one() { # kind source dest-name [options de quantize]
  local kind="$1" src="$2" dst="$3" dir="$OUT/in/$1"
  shift 3
  [ "$kind" = gmfss ] && dir="$OUT/in/gmfss-pair"
  "$PY" "$TOOL" quantize "$src" "$OUT/models/$dst" --inputs $(files "$dir" "${CAL[@]}") --min-channels 32 --calib max "$@"
}
quality() { # kind ref test
  echo "== $2 -> $3" | tee -a "$OUT/quality.txt"
  local dir="$OUT/in/$1"; [ "$1" = gmfss ] && dir="$OUT/in/gmfss-pair"
  "$PY" "$TOOL" quality "$1" --ref "$2" --test "$3" --inputs $(files "$dir" "${TEST[@]}") | tee -a "$OUT/quality.txt" || true
}

for m in rife_v4.25 rife_v4.22_lite; do  # écartés (gardes non tenues et plus lents) : rejected/, jamais publiés
  one rife "$RIFE_DIR/$m.onnx" "../rejected/${m}_fp8.onnx"
  quality rife "$RIFE_DIR/$m.onnx" "$OUT/rejected/${m}_fp8.onnx"
done
for m in distil_drba_v2_lite distil_drba_v2_lite_s05; do
  one drba "$INTERP_DIR/$m.onnx" "${m}_fp8.onnx"
  quality drba "$INTERP_DIR/$m.onnx" "$OUT/models/${m}_fp8.onnx"
done
for v in base union; do  # la paire seule : la fusion reste FP16 (le FP8 y perd 8 à 11 dB au pire cas)
  one gmfss "$INTERP_DIR/gmfss_fortuna_${v}_s05_pair.onnx" "gmfss_fortuna_${v}_s05fp8_pair.onnx" --skip /flownet/backbone/
  quality gmfss "$INTERP_DIR/gmfss_fortuna_${v}_s05_pair.onnx,$INTERP_DIR/gmfss_fortuna_${v}_s05_fusion.onnx" \
    "$OUT/models/gmfss_fortuna_${v}_s05fp8_pair.onnx,$INTERP_DIR/gmfss_fortuna_${v}_s05_fusion.onnx"
done
(cd "$OUT/models" && sha256sum ./*.onnx | sed 's# \./# #' >../SHA256SUMS.fp8)
echo "graphes : $OUT/models ; empreintes : $OUT/SHA256SUMS.fp8 ; qualité : $OUT/quality.txt"
