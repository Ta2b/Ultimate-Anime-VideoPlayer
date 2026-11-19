#!/bin/bash
# Compare la découpe en plans de deux versions de src/media/shot_cutter.cpp sur des vidéos : les coupes de
# chacune, et pour chaque désaccord une bande de quatre images (deux avant la coupe, deux après) à juger à
# l'œil.
#   banc_coupes.sh RÉF_A RÉF_B SORTIE VIDÉO...      par ex. banc_coupes.sh origin/main HEAD ~/kirari-reports/coupes ep*.mkv
# RÉF_A et RÉF_B : des révisions git du dépôt. Il faut g++ (C++23), ffmpeg et ffprobe.
# Une version dont shot_cutter.h lit le flux optique (media/probe_motion.h) se construit avec la session NVOF
# de la pré-analyse (media/probe_flow.h) : il faut alors les en-têtes de FFmpeg (pkg-config libavutil) et une
# carte NVIDIA. BANC_SANS_FLUX=1 l'éteint à l'exécution (les coupes doivent alors être celles d'avant).
# Sortie : SORTIE/<vidéo>.a et .b (les coupes), SORTIE/<vidéo>.a.flux et .b.flux (la part de blocs sans
# correspondance de chaque image, si la version lit le flux), SORTIE/desaccords.tsv (vidéo, image, version
# qui coupe seule), SORTIE/bandes/<vidéo>-<image>-<a|b>.jpg.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
root=$(git -C "$here" rev-parse --show-toplevel)
ref_a=${1:?révision A attendue}
ref_b=${2:?révision B attendue}
out=${3:?dossier de sortie attendu}
shift 3
(($# > 0)) || { echo "au moins une vidéo attendue" >&2; exit 2; }
mkdir -p "$out/bandes"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

build() {  # build RÉF NOM
  mkdir -p "$work/$2/media"
  git -C "$root" show "$1:src/media/shot_cutter.h" > "$work/$2/media/shot_cutter.h"
  git -C "$root" show "$1:src/media/shot_cutter.cpp" > "$work/$2/shot_cutter.cpp"
  local flux=()
  if grep -q 'probe_motion.h' "$work/$2/media/shot_cutter.h"; then
    for f in probe_motion.h probe_flow.h; do git -C "$root" show "$1:src/media/$f" > "$work/$2/media/$f"; done
    for f in probe_flow.cpp probe_flow_nvof.cpp; do git -C "$root" show "$1:src/media/$f" > "$work/$2/$f"; done
    # shellcheck disable=SC2207 # les options de pkg-config, mot à mot
    flux=(-DKIRARI_BANC_FLUX -I"$root/src/media/nvof_cuda" -I"$root/third_party/nvidia-optical-flow"
      $(pkg-config --cflags libavutil libavcodec) "$work/$2/probe_flow.cpp" "$work/$2/probe_flow_nvof.cpp"
      "$root/src/media/cuda_driver.cpp" -ldl)
  fi
  g++ -std=c++23 -O2 -I"$work/$2" -I"$root/src" -o "$work/$2/banc" "$here/banc_coupes.cpp" \
    "$work/$2/shot_cutter.cpp" "$root/src/kernels/ref/frame_stats.cpp" "$root/src/kernels/ref/transfer.cpp" \
    "$root/src/media/faces.cpp" "${flux[@]}"
}
mode=$([[ ${BANC_SANS_FLUX:-0} == 1 ]] && echo sans-flux || echo avec-flux)
build "$ref_a" a
build "$ref_b" b

strip() {  # strip VIDÉO FPS IMAGE SORTIE
  local first=$(($3 - 2))
  ((first < 0)) && first=0
  ffmpeg -v error -y -ss "$(awk -v n="$first" -v f="$2" 'BEGIN { printf "%.4f", n / f }')" -i "$1" -frames:v 1 \
    -vf "scale=384:216,drawtext=text='%{eif\:n+$first\:d}':x=4:y=4:fontsize=24:fontcolor=yellow:box=1:boxcolor=black,tile=4x1" \
    -q:v 4 "$4" < /dev/null
}

printf 'video\timage\tcoupe_seule\n' > "$out/desaccords.tsv"
for video in "$@"; do
  name=$(basename "${video%.*}")
  fps=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$video" |
    awk -F/ '{ printf "%.6f", $2 ? $1 / $2 : $1 }')
  ffmpeg -v error -i "$video" -map 0:v:0 -vf scale=256:144:flags=area -pix_fmt rgb24 -f rawvideo - |
    tee >("$work/a/banc" "$fps" "$mode" > "$out/$name.a" 2> "$out/$name.a.flux") |
    "$work/b/banc" "$fps" "$mode" > "$out/$name.b" 2> "$out/$name.b.flux"
  sleep 1  # la substitution de A finit d'écrire
  for side in a b; do
    other=$([[ $side == a ]] && echo b || echo a)
    comm -23 <(sort -n "$out/$name.$side" | sort) <(sort "$out/$name.$other") | sort -n | while read -r frame; do
      printf '%s\t%s\t%s\n' "$name" "$frame" "$side" >> "$out/desaccords.tsv"
      strip "$video" "$fps" "$frame" "$out/bandes/$name-$frame-$side.jpg"
    done
  done
  echo "$name : A $(wc -l < "$out/$name.a") coupes, B $(wc -l < "$out/$name.b") coupes"
done
echo "désaccords : $(($(wc -l < "$out/desaccords.tsv") - 1)) (bandes dans $out/bandes)"
