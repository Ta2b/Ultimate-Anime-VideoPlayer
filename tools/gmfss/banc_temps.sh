#!/usr/bin/env bash
# Temps de GMFSS Fortuna Union au trtexec (graphe CUDA, niveau 5), face à RIFE 4.26 à la même taille : la paire
# (flot entier et demi-flot), la fusion en FP16, puis la fusion en FP8 (tools/fp8/fp8_probe.py, Q/DQ de Model
# Optimizer sur entrées aléatoires : assez pour la vitesse, pas pour la qualité).
#   flock ~/ajp-reports/gpu-bench.lock tools/gmfss/banc_temps.sh ROUE/vsgmfss_fortuna rife_v4.26.onnx DOSSIER [LxH=1280x768]
# La taille est celle des plans remplis (multiple de 128) : 1280x768 pour 720p de travail, 1920x1152 pour 1080p.
# Il faut torch, onnx, onnxruntime (export), nvidia-modelopt[onnx] (FP8) et trtexec (TRTEXEC ou ~/ajp-sdk).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="${1:?dossier vsgmfss_fortuna de la roue}"; RIFE="${2:?rife_v4.26.onnx}"; OUT="${3:?dossier de travail}"
SIZE="${4:-1280x768}"; W="${SIZE%x*}"; H="${SIZE#*x}"
PY="${PYTHON:-python3}"
TRTEXEC="${TRTEXEC:-$(command -v trtexec || echo "$HOME/ajp-sdk/tensorrt/bin/trtexec")}"
export LD_LIBRARY_PATH="${TRTEXEC%/bin/trtexec}/lib:${LD_LIBRARY_PATH:-}"
mkdir -p "$OUT"
LOG="$OUT/banc-temps-$SIZE.md"
: >"$LOG"
say() { echo "$*" | tee -a "$LOG"; }

# Graphes FP16 : flot entier et demi-flot (même fusion, seule la paire change).
for scale in 1.0 0.5; do
  dir="$OUT/onnx-$scale"
  [ -n "$(ls "$dir"/*_pair.onnx 2>/dev/null)" ] ||
    "$PY" -I "$HERE/export_gmfss.py" "$PKG" --out "$dir" --model union --scale "$scale" >"$OUT/export-$scale.log" 2>&1
done
PAIR1="$(ls "$OUT"/onnx-1.0/*_pair.onnx)"; PAIR05="$(ls "$OUT"/onnx-0.5/*_pair.onnx)"
FUSION="$(ls "$OUT"/onnx-1.0/*_fusion.onnx)"

h2=$((H / 2)); w2=$((W / 2)); h4=$((H / 4)); w4=$((W / 4)); h8=$((H / 8)); w8=$((W / 8))
PAIR_SHAPES="frames:1x6x${H}x${W}"
FUSION_SHAPES="half0:1x3x${h2}x${w2},half1:1x3x${h2}x${w2},time:1x1x1x1"
for k in 0 1; do
  FUSION_SHAPES+=",warped${k}_1:1x67x${h2}x${w2},warped${k}_2:1x128x${h4}x${w4},warped${k}_3:1x192x${h8}x${w8}"
done
RIFE_SHAPES="input:1x11x${H}x${W}"

time_of() { # onnx formes étiquette -> « min médiane »
  local eng="$OUT/$3.engine"
  [ -f "$eng" ] || "$TRTEXEC" --onnx="$1" --minShapes="$2" --optShapes="$2" --maxShapes="$2" \
    --builderOptimizationLevel=5 --saveEngine="$eng" >"$OUT/$3.build.log" 2>&1 ||
    { echo "construction de $3 échouée ($OUT/$3.build.log)" >&2; return 1; }
  "$TRTEXEC" --loadEngine="$eng" --shapes="$2" --useCudaGraph --noDataTransfers --warmUp=500 --iterations=300 \
    --dumpProfile --separateProfileRun --exportProfile="$OUT/$3.profile.json" 2>&1 |
    sed -n 's/.*GPU Compute Time: min = \([0-9.]*\) ms.*median = \([0-9.]*\) ms.*/\1 \2/p'
}

RIFE_INPUT="$("$PY" -c "import onnx,sys;print(onnx.load(sys.argv[1]).graph.input[0].name)" "$RIFE")"
RIFE_SHAPES="${RIFE_INPUT}:1x11x${H}x${W}"
say "# GMFSS Union au trtexec, plans ${W}×${H} ($(date -I), $("$TRTEXEC" 2>&1 | grep -m1 -o 'TensorRT.v[0-9]*' || true))"
say ""
say "| Graphe | min (ms) | médiane (ms) |"
say "|---|---:|---:|"
for row in "RIFE 4.26 (graphe livré, une passe)|$RIFE|$RIFE_SHAPES|rife426" \
           "GMFSS paire, flot entier|$PAIR1|$PAIR_SHAPES|paire-1.0" \
           "GMFSS paire, demi-flot|$PAIR05|$PAIR_SHAPES|paire-0.5" \
           "GMFSS fusion (IFNet + GridNet), FP16|$FUSION|$FUSION_SHAPES|fusion-fp16"; do
  IFS='|' read -r label onnx shapes tag <<<"$row"
  read -r mn md < <(time_of "$onnx" "$shapes" "$tag")
  say "| $label | $mn | $md |"
done

# FP8 de la fusion : Softmax et redimensionnements gardés en FP16, comme pour la paire d'avant 0.10.0.
"$PY" "$HERE/../fp8/fp8_probe.py" "$FUSION" --out "$OUT/fp8" --trtexec "$TRTEXEC" --exclude Softmax,Resize \
  $(echo "$FUSION_SHAPES" | tr ',' '\n' | sed 's/^\(.*\):/--shape \1=/') >"$OUT/fp8.log" 2>&1 &&
  say "| GMFSS fusion, FP8 (sonde) | $("$PY" -c "import json,sys;r=json.load(open(sys.argv[1]));print(f\"{r['fp8_ms']:.2f} | (FP16 {r['fp16_ms']:.2f}, ×{r['speedup']:.2f})\")" "$OUT/fp8/$(basename "$FUSION" .onnx)_fp8.json") |" ||
  say "| GMFSS fusion, FP8 (sonde) | échec ($OUT/fp8.log) | |"
say ""
say "Profils par couche : $OUT/*.profile.json (la part de l'IFNet et de GridNet dans la fusion)."
