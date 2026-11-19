#!/usr/bin/env bash
# Banc unique du SPAN FP16 natif (lot C du chantier AnimeJaNai, docs/span-native16.md) : pour chaque modèle,
# TensorRT FP16 (trtexec, graphe CUDA) contre nos noyaux FP16 natifs (kirari_span_fp8_run, graphe CUDA), en
# ABAB, les natifs en f32, et le balayage des bandes (docs/span-native16.md, « Bandes ») : le défaut du
# lecteur (128 lignes, ou 64 si le L2 ne les tient pas) en ABAB, puis sans bandes et à d'autres hauteurs
# ($BANDS). Le tableau des temps finit dans $OUT/resume.md.
#
# Le PC ne fait que les temps : la justesse se juge sur processeur, dans le cloud. Avec EXPORT=DOSSIER, le banc
# écrit pour chaque modèle, dans DOSSIER/<modèle>/, le modèle, l'entrée et une sortie de chaque mode (f16 aux
# bandes du lecteur, f16 sans bandes, f32) en demi-flottants, un fichier par plan (moins de 20 Mo chacun) ;
# tools/bench/span_native/justesse_cpu.py les compare ensuite à ONNX Runtime FP32. Sans EXPORT, rien.
#
#   [EXPORT=/mnt/project-files/span-native/…] flock ~/kirari-reports/gpu-bench.lock \
#       tools/bench/span_native/banc.sh BUILD MODELE.onnx [MODELE.onnx…] [-- video.mkv]
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
H=1080; W=1920; ROUNDS="${ROUNDS:-3}"; BANDS="${BANDS:-0 64 192 270}"
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
  printf ' Natif f32 (ms) |\n|---|---:|---:|---:|'
  for b in $BANDS; do printf -- '---:|'; done
  printf -- '---:|\n'
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
  t=(); n=(); for r in $(seq 1 "$ROUNDS"); do  # ABAB : une dérive d'horloge ne pèse pas d'un seul côté
    if ((r % 2)); then t+=("$(trt_ms "$engine")"); n+=("$(native_ms "$model")"); else
      n+=("$(native_ms "$model")"); t+=("$(trt_ms "$engine")"); fi
    echo "$name tour $r : TensorRT ${t[-1]} ms, natif ${n[-1]} ms"
  done
  n32="$(native_ms "$model" --acc32)"
  sweep=""; for b in $BANDS; do sweep+=" $(native_ms "$model" --band "$b") |"; done
  med() { printf '%s\n' "$@" | sort -g | sed -n "$((($# + 1) / 2))p"; }
  mt="$(med "${t[@]}")"; mn="$(med "${n[@]}")"
  printf '| %s | %s | %s | %s |%s %s |\n' "$name" "$mt" "$mn" \
    "$(awk -v a="$mn" -v b="$mt" 'BEGIN{printf "%.2f", a/b}')" "$sweep" "$n32" >>"$OUT/resume.md"
done

# Les données de la justesse, après tous les temps : une sortie par mode, en demi-flottants (les noyaux
# n'écrivent que des demi-flottants, rien ne s'y perd), un plan par fichier.
if [ -n "${EXPORT:-}" ]; then
  for model in "${MODELS[@]}"; do
    name="$(basename "$model" .onnx)"; dest="$EXPORT/$name"; mkdir -p "$dest"
    "$RUN" "$model" "$H" "$W" "$IN" "$OUT/$name.f16.f32" --fp16 >/dev/null
    "$RUN" "$model" "$H" "$W" "$IN" "$OUT/$name.b0.f32" --fp16 --band 0 >/dev/null
    "$RUN" "$model" "$H" "$W" "$IN" "$OUT/$name.f32.f32" --fp16 --acc32 >/dev/null
    cp "$model" "$dest/modele.onnx"
    python3 -I - "$dest" "$IN" "$H" "$W" "$OUT/$name" <<'PY'
import sys
from pathlib import Path
import numpy as np
dest, entree, h, w, base = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
(dest / "dims.txt").write_text(f"{h} {w}\n")
np.fromfile(entree, dtype="<f4").astype("<f2").tofile(dest / "entree.f16")  # comme __float2half
for mode in ("f16", "b0", "f32"):
    sortie = np.fromfile(f"{base}.{mode}.f32", dtype="<f4").astype("<f2").reshape(3, 2 * h, 2 * w)
    for c in range(3):
        sortie[c].tofile(dest / f"{mode}.p{c}.f16")
PY
    rm -f "$OUT/$name".{f16,b0,f32}.f32
    echo "export : $dest"
  done
fi
cat "$OUT/resume.md"
echo "rapport : $OUT"
