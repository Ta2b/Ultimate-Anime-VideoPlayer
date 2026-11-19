#!/usr/bin/env bash
# Banc unique du SPAN FP16 natif (lot C du chantier AnimeJaNai, docs/span-native16.md) : pour chaque modèle,
# TensorRT FP16 (trtexec, graphe CUDA) contre nos noyaux FP16 natifs (kirari_span_fp8_run, graphe CUDA), en
# ABAB, plus la justesse des noyaux natifs contre la référence FP16 à sommes f32 (--compare), sommes en f16
# (le défaut), en f32 et promues (--promoted, docs/span-native16.md, « Sommes promues »), et le balayage des
# bandes (« Bandes ») : le défaut du lecteur (64 lignes) en ABAB, puis sans bandes et à d'autres hauteurs
# ($BANDS). L'essai C0 (E/S HWC8 de TensorRT) a perdu au premier banc (15,96 ms contre 15,39) et n'y est
# plus. Un seul lancement donne tout ; le tableau finit dans $OUT/resume.md.
#
#   flock ~/kirari-reports/gpu-bench.lock tools/bench/span_native/banc.sh BUILD MODELE.onnx [MODELE.onnx…] \
#       [-- video.mkv]
#
# BUILD : dossier de construction linux-release (ou linux-prefix-cuda) où kirari_span_fp8_run est construit.
# La vidéo donne une vraie image 1080p (ffmpeg, image à 60 s) ; sans elle, une mire synthétique.
# Cibles (4070 Ti, 1080p → 4K) : b8f64 ≤ 11,2 ms (égaler AnimeJaNai), ≤ 10 (dépasser) ; b5f48 ≤ 5,8 et ≤ 5.
set -euo pipefail
BUILD="$1"; shift
MODELS=(); VIDEO=""
while (($#)); do
  if [ "$1" = "--" ]; then VIDEO="${2:-}"; break; fi
  MODELS+=("$1"); shift
done
H=1080; W=1920; ROUNDS="${ROUNDS:-3}"; BANDS="${BANDS:-0 32 128}"
RUN="$BUILD/src/gpu/cuda/backend/kirari_span_fp8_run"
[ -x "$RUN" ] || RUN="$(find "$BUILD" -name kirari_span_fp8_run -type f -perm -u+x | head -1)"
[ -x "$RUN" ] || { echo "kirari_span_fp8_run introuvable sous $BUILD" >&2; exit 2; }
TRTEXEC="${TRTEXEC:-$(command -v trtexec || echo "$HOME/kirari-sdk/tensorrt/bin/trtexec")}"
export LD_LIBRARY_PATH="${TRTEXEC%/bin/trtexec}/lib:${LD_LIBRARY_PATH:-}"
OUT="${OUT:-$HOME/kirari-reports/span-native-$(date +%Y%m%d-%H%M)}"
mkdir -p "$OUT"
IN="$OUT/entree.f32"

# L'entrée : 3×H×W float32 NCHW dans [0, 1].
python3 -I - "$IN" "$H" "$W" "$VIDEO" <<'PY'
import array, math, subprocess, sys
path, h, w, video = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
if video:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", "60", "-i", video, "-frames:v", "1", "-vf",
                          f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "gbrpf32le", "-"],
                         check=True, capture_output=True).stdout
    planes = [raw[i * h * w * 4:(i + 1) * h * w * 4] for i in range(3)]  # G, B, R
    data = planes[2] + planes[0] + planes[1]
else:
    a = array.array("f", (0.5 + 0.5 * math.sin(0.013 * x * (c + 1) + 0.021 * y) * (1 if (x // 64 + y // 64) % 2 else 0.6)
                          for c in range(3) for y in range(h) for x in range(w)))
    data = a.tobytes()
open(path, "wb").write(data)
PY

trt_ms() { # moteur -> médiane GPU en ms
  "$TRTEXEC" --loadEngine="$1" --useCudaGraph --noDataTransfers --warmUp=500 --iterations=300 2>&1 |
    sed -n 's/.*GPU Compute Time: min = [0-9.]* ms.*median = \([0-9.]*\) ms.*/\1/p'
}
native_ms() { # modèle [options] -> médiane en ms
  "$RUN" "$1" "$H" "$W" "$IN" "$OUT/sortie.f32" --fp16 --bench 300 "${@:2}" |
    sed -n 's/.*médiane \([0-9.]*\) ms.*/\1/p'
}

{
  printf '| Modèle | TensorRT FP16 (ms) | Natif f16 (ms) | Natif/TRT |'
  for b in $BANDS; do printf ' Bandes %s (ms) |' "$b"; done
  printf ' Natif f32 (ms) | Natif promu (ms) | Écart f16 | Écart f16 sans bandes | Écart f32 | Écart promu |\n'
  printf '|---|---:|---:|---:|'
  for b in $BANDS; do printf -- '---:|'; done
  printf -- '---:|---:|---|---|---|---|\n'
} >"$OUT/resume.md"
for model in "${MODELS[@]}"; do
  name="$(basename "$model" .onnx)"
  engine="$OUT/$name.engine"
  input="$(python3 -I -c 'import onnx,sys; print(onnx.load(sys.argv[1]).graph.input[0].name)' "$model" 2>/dev/null ||
    echo input)"
  shape="$input:1x3x${H}x${W}"
  "$TRTEXEC" --onnx="$model" --minShapes="$shape" --optShapes="$shape" --maxShapes="$shape" \
    --builderOptimizationLevel=5 --saveEngine="$engine" >"$OUT/$name.build.log" 2>&1 ||
    "$TRTEXEC" --onnx="$model" --builderOptimizationLevel=5 --saveEngine="$engine" >"$OUT/$name.build.log" 2>&1 ||
    { echo "construction TensorRT de $name échouée ($OUT/$name.build.log)" >&2; exit 1; }
  # Justesse sur l'image entière, contre la référence FP16 à sommes f32 : chaque accumulateur en bandes, et
  # sans bandes (deux écarts égaux à l'arrondi près : l'ordre des bandes ne change aucune somme).
  q16="$("$RUN" "$model" "$H" "$W" "$IN" "$OUT/$name.f16.f32" --fp16 --compare | sed -n 's/^écart[^:]*: //p')"
  q0="$("$RUN" "$model" "$H" "$W" "$IN" "$OUT/$name.f16b0.f32" --fp16 --band 0 --compare |
    sed -n 's/^écart[^:]*: //p')"
  q32="$("$RUN" "$model" "$H" "$W" "$IN" "$OUT/$name.f32.f32" --fp16 --acc32 --compare | sed -n 's/^écart[^:]*: //p')"
  qp="$("$RUN" "$model" "$H" "$W" "$IN" "$OUT/$name.promu.f32" --fp16 --promoted --compare |
    sed -n 's/^écart[^:]*: //p')"
  t=(); n=(); for r in $(seq 1 "$ROUNDS"); do  # ABAB : une dérive d'horloge ne pèse pas d'un seul côté
    if ((r % 2)); then t+=("$(trt_ms "$engine")"); n+=("$(native_ms "$model")"); else
      n+=("$(native_ms "$model")"); t+=("$(trt_ms "$engine")"); fi
    echo "$name tour $r : TensorRT ${t[-1]} ms, natif ${n[-1]} ms"
  done
  n32="$(native_ms "$model" --acc32)"; np="$(native_ms "$model" --promoted)"
  sweep=""; for b in $BANDS; do sweep+=" $(native_ms "$model" --band "$b") |"; done
  med() { printf '%s\n' "$@" | sort -g | sed -n "$((($# + 1) / 2))p"; }
  mt="$(med "${t[@]}")"; mn="$(med "${n[@]}")"
  printf '| %s | %s | %s | %s |%s %s | %s | %s | %s | %s | %s |\n' "$name" "$mt" "$mn" \
    "$(awk -v a="$mn" -v b="$mt" 'BEGIN{printf "%.2f", a/b}')" "$sweep" "$n32" "$np" "$q16" "$q0" "$q32" "$qp" \
    >>"$OUT/resume.md"
done
cat "$OUT/resume.md"
echo "rapport : $OUT"
