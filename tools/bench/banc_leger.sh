#!/usr/bin/env bash
# Banc des petites cartes (P6.7, docs/phase6-a-tester.md) : ce que coûte et ce que rend chaque agrandissement léger,
# pour décider de kLightSpanInAuto (src/app/play/auto_config.h). Cinq variantes, sans interpolation, à la même
# hauteur de travail :
#
#   econome         la source telle quelle, sans réseau (la tranche Économe d'aujourd'hui)
#   span-fp16       SPAN V3.1 Sharp1 Performance (b5f48) en FP16
#   span-int8       le même au cran Performance : sous TensorRT, sa variante INT8 (tools/span/quantize_span.py,
#                   rangée dans le dossier « modeles ») ; ailleurs ou sans variante, le journal le dit
#   artcnn-c4f32    ArtCNN C4F32 (32 canaux)
#   artcnn-c4f16    ArtCNN C4F16 (16 canaux)
#
#   tools/bench/banc_leger.sh --extraits=extraits.tsv --sortie=DOSSIER [--tours=3] [--secondes=60]
#     [--hauteur=540] [--variantes=econome,span-fp16,...] [--reglages=settings.toml] [--ajp=CHEMIN]
#
# extraits.tsv : « fichier<TAB>début en secondes » par ligne (tools/bench/extraits.exemple.tsv). Chaque passage est
# un processus neuf, dans l'ordre des variantes tourné d'un tour à l'autre, après un préchauffage non compté (moteurs,
# caches). Sous Vulkan, --profile-dispatches donne le temps GPU de chaque réseau au journal ; sous TensorRT, le
# fichier .gpu.tsv à côté des stats. Une capture par variante et par extrait sert à juger l'image à l'œil.
# --reglages : un settings.toml à reprendre (le moteur choisi) ; sinon celui d'un poste neuf, assistant passé.
# Le résumé : tools/bench/banc_leger_resume.py DOSSIER.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTRAITS=""; OUT=""; ROUNDS=3; SECONDS_PLAYED=60; HEIGHT=540; SETTINGS=""; AJP="${AJP:-ajp}"
VARIANTS="econome,span-fp16,span-int8,artcnn-c4f32,artcnn-c4f16"
for arg in "$@"; do
  case "$arg" in
    --extraits=*) EXTRAITS="${arg#--extraits=}" ;;
    --sortie=*) OUT="${arg#--sortie=}" ;;
    --tours=*) ROUNDS="${arg#--tours=}" ;;
    --secondes=*) SECONDS_PLAYED="${arg#--secondes=}" ;;
    --hauteur=*) HEIGHT="${arg#--hauteur=}" ;;
    --variantes=*) VARIANTS="${arg#--variantes=}" ;;
    --reglages=*) SETTINGS="${arg#--reglages=}" ;;
    --ajp=*) AJP="${arg#--ajp=}" ;;
    *) echo "option inconnue : $arg" >&2; exit 2 ;;
  esac
done
[ -n "$EXTRAITS" ] && [ -n "$OUT" ] || { echo "il faut --extraits et --sortie" >&2; exit 2; }
IFS=, read -r -a LABELS <<<"$VARIANTS"
SPAN=2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx

# Les options de « ajp play » et le cran de précision de SPAN (play.toml) d'une variante.
options_of() {
  case "$1" in
    econome) echo "--no-upscale --work-height=0" ;;
    span-fp16 | span-int8) echo "--model=$SPAN --work-height=$HEIGHT" ;;
    artcnn-c4f32) echo "--model=2x_ArtCNN_C4F32_rgb_fp16.onnx --work-height=$HEIGHT" ;;
    artcnn-c4f16) echo "--model=2x_ArtCNN_C4F16_rgb_fp16.onnx --work-height=$HEIGHT" ;;
    *) echo "variante inconnue : $1" >&2; exit 2 ;;
  esac
}
precision_of() { [ "$1" = span-int8 ] && echo performance || echo balanced; }
for label in "${LABELS[@]}"; do options_of "$label" >/dev/null; done

mkdir -p "$OUT"
{
  echo "date : $(date -Is)"
  echo "ajp : $("$AJP" --version 2>&1 | head -1)"
  echo "noyau : $(uname -sr)"
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  command -v vulkaninfo >/dev/null && vulkaninfo --summary 2>/dev/null | grep -m2 -E "deviceName|driverInfo"
  echo "hauteur de travail : $HEIGHT ; variantes : $VARIANTS"
} >"$OUT/contexte.txt"

# Un dossier de réglages neuf par passage : son journal ne parle que de lui. Préréglage Personnalisé, pour que
# l'Auto ne réécrive pas la chaîne demandée.
settings_dir() {  # étiquette variante
  local dir="$OUT/reglages-$1"
  rm -rf "$dir" && mkdir -p "$dir"
  if [ -n "$SETTINGS" ]; then cp "$SETTINGS" "$dir/settings.toml"
  else printf 'version = 1\nfirst_run_done = true\nfirst_run_step = 8\n' >"$dir/settings.toml"; fi
  printf '[auto]\npreset = "custom"\n\n[image]\nspan_precision = "%s"\n' "$(precision_of "$2")" >"$dir/play.toml"
  echo "$dir"
}

play() {  # variante fichier début étiquette secondes [capture]
  local label="$1" file="$2" start="$3" tag="$4" seconds="$5" shot="${6:-}"
  local base="$OUT/$tag" dir extra=""
  dir="$(settings_dir "$tag" "$label")"
  [ -n "$shot" ] && extra="--snapshot=$OUT/capture-$shot-$label.ppm --snapshot-at=$((seconds / 2))"
  # shellcheck disable=SC2046,SC2086
  "$AJP" play "$file" --start="$start" --seconds="$seconds" --fullscreen --mute --no-resume --rife=0 \
    --profile-dispatches --data-dir="$dir" --stats="$base.tsv" $(options_of "$label") $extra \
    </dev/null >"$base.log" 2>&1 || echo "échec : $tag" >&2
  [ -f "$dir/ajp.log" ] && cp "$dir/ajp.log" "$base.ajp.log"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv,noheader >"$base.smi" || true
  fi
  rm -rf "$dir"
}

n=0
while IFS=$'\t' read -r file start; do
  [ -n "$file" ] && [[ "$file" != \#* ]] || continue
  n=$((n + 1))
  for label in "${LABELS[@]}"; do play "$label" "$file" "$start" "prechauffage-$n-$label" 10 "$n"; done
  for round in $(seq 1 "$ROUNDS"); do
    count=${#LABELS[@]}
    for k in $(seq 0 $((count - 1))); do  # l'ordre tourne : aucune variante toujours première
      label="${LABELS[$(((k + round) % count))]}"
      play "$label" "$file" "$start" "extrait$n-tour$round-$label" "$SECONDS_PLAYED"
    done
  done
done <"$EXTRAITS"

python3 -I "$HERE/banc_leger_resume.py" "$OUT" | tee "$OUT/resume.txt"
