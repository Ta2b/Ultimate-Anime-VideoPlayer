#!/usr/bin/env bash
# Les découpes de 256 × 256 des dix-huit plans de data/calibration/calibration.mkv (deux par plan : le centre
# et le quart haut gauche), en RGB 8 bits brut à la file, pour la mesure « grade avant le warp sur les plans
# de calibration » (tests/unit/render/grade_before_warp_test.cpp, [.mesure]).
set -euo pipefail
here="$(cd "$(dirname "$0")/../.." && pwd)"
out="${1:-$(dirname "$(mktemp -u)")/ajp-warp-stills.rgb}"
: >"$out"
for plan in $(seq 0 17); do
  at=$((plan * 6 + 3))
  for crop in "256:256:832:412" "256:256:480:270"; do
    ffmpeg -v error -ss "$at" -i "$here/data/calibration/calibration.mkv" -frames:v 1 \
      -vf "scale=in_range=limited:out_range=full,format=rgb24,crop=$crop" -f rawvideo - >>"$out"
  done
done
echo "$out : $(($(stat -c %s "$out") / 196608)) découpes"
