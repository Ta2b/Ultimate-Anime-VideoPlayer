#!/bin/bash
# Étape 1 du doc « Protéger les visages » : cadres de référence en x.pq.ppm (sortie HDR10 réelle, grade compris)
# pour les 18 plans de calibration.mkv. Une instance à la fois, fenêtre 1280x720, --data-dir isolé.
#   capture_plans.sh <config: actuel|ancien|sanspile|ancien-sanspile|sansgrade|anime|anime-sanspile> [plans...]
# Environnement : R (dossier de sortie), L (lancer.sh), CAL (calibration.mkv), BASE (copie des réglages HDR
# adaptatif : play.toml, user.toml, settings.toml), SNAP_AT (seconde de la capture, 6 par défaut : le grade
# adaptatif a une constante de temps).
R=${R:-$HOME/ajp-reports/grade-ref}
L=${L:-$HOME/ajp-test/lancer.sh}
CAL=${CAL:-$HOME/ajp-test/ajp-linux-x64/calibration/calibration.mkv}
BASE=${BASE:-/tmp/snap/abdata-defaut}
SNAP_AT=${SNAP_AT:-6}
cfg=$1
shift
plans=${*:-$(seq 1 18)}
mkdir -p "$R/frames"
for k in $plans; do
  d=/tmp/snap/gref-$cfg
  rm -rf "$d"
  mkdir -p "$d"
  cp "$BASE/settings.toml" "$BASE/user.toml" "$BASE/play.toml" "$d/"
  extra=""
  case $cfg in
    ancien) extra="--skin-mask" ;;
    anime) extra="--hdr-anime" ;;
    sanspile | ancien-sanspile | anime-sanspile)
      # pile de LUT sans effet (intensité 0), le grade lui-même intact
      sed -i "s/^look_mix = .*/look_mix = 0.0/; s/^looks_mix = .*/looks_mix = [ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]/" "$d/play.toml"
      [ "$cfg" = ancien-sanspile ] && extra="--skin-mask"
      [ "$cfg" = anime-sanspile ] && extra="--hdr-anime"
      ;;
    sansgrade)
      sed -i "/^\[grade\]/,/^\[hdr\]/ { s/^enabled = true/enabled = false/; s/^adaptive = true/adaptive = false/ }" "$d/play.toml"
      sed -i "s/^engine = 'adaptive'/engine = 'sdr'/" "$d/play.toml"
      ;;
  esac
  # un lecteur déjà ouvert à ce moment n'est pas le nôtre : on attend qu'il le ferme
  while pgrep -x ajp > /dev/null; do
    echo "$(date +%T) un lecteur tourne, j'attends" >> "$R/capture.log"
    sleep 20
  done
  t=$((6 * k - 3))
  n=$(printf %02d "$k")
  # shellcheck disable=SC2086
  timeout 120 "$L" "$CAL" --start=$t --no-resume --mute --window=1280x720 --data-dir="$d" \
    --snapshot="$R/frames/plan$n-$cfg.pq.ppm" --snapshot-at="$SNAP_AT" --snapshot-frames=1 --seconds=$((SNAP_AT + 3)) \
    $extra > "$R/frames/plan$n-$cfg.log" 2>&1
  echo "plan $k $cfg rc=$?" >> "$R/capture.log"
done
echo "$cfg FINI" >> "$R/capture.log"
