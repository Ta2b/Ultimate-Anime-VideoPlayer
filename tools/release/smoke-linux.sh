#!/usr/bin/env bash
# Lance le paquet Linux tel que l'utilisateur le recevra, extrait dans un dossier propre, avant qu'il soit joint puis
# signé (B51-04) : aucune bibliothèque introuvable (ldd), le lecteur répond (--help), et il joue un clip en Vulkan
# logiciel jusqu'à un instantané non vide (CLAUDE.md : xvfb-run ajp --software-gpu --snapshot=…). L'AppImage, si
# donnée, passe les mêmes essais par --appimage-extract-and-run.
#
#   tools/release/smoke-linux.sh <ajp-linux-x64.tar.gz> [ajp-linux-x86_64.AppImage]
#
# Demande xvfb-run, mesa-vulkan-drivers (lavapipe) et ffmpeg (pour le clip d'essai) ; sans ffmpeg ni xvfb-run, l'instantané
# est sauté avec un avertissement (le reste compte).
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,11p' "$0"; exit 2; }
archive=$1 appimage=${2:-}
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
tar -xzf "$archive" -C "$work"
root="$work/ajp-linux-x64"
[ -x "$root/ajp" ] || { echo "::error::$archive : pas de ajp exécutable dans ajp-linux-x64/" >&2; exit 1; }

# 1. Aucune dépendance introuvable, pour le lecteur et son moteur Vulkan.
for bin in "$root/ajp" "$root/ajp_backend_vulkan.so"; do
  [ -e "$bin" ] || { echo "::error::$(basename "$bin") absent du paquet" >&2; exit 1; }
  if missing="$(ldd "$bin" 2>&1 | grep 'not found')"; then
    echo "::error::$(basename "$bin") : bibliothèque introuvable depuis le paquet extrait :" >&2
    echo "$missing" >&2
    exit 1
  fi
done
echo "dépendances : toutes trouvées"

# 2. Le lecteur démarre.
"$root/ajp" --help >/dev/null || { echo "::error::ajp --help échoue" >&2; exit 1; }
echo "ajp --help : ok"

# 3. Un clip, joué en Vulkan logiciel jusqu'à un instantané.
snapshot() {  # lanceur… : dossier de données propre, instantané à 3 s
  local shot="$work/shot.ppm"
  rm -f "$shot"
  xvfb-run -a "$@" --software-gpu --mute --no-upscale --data-dir="$work/data" --snapshot="$shot" \
    --snapshot-at=3 "$work/clip.mkv"
  [ -s "$shot" ] || { echo "::error::instantané absent ou vide" >&2; exit 1; }
}
if command -v ffmpeg >/dev/null && command -v xvfb-run >/dev/null; then
  ffmpeg -loglevel error -f lavfi -i testsrc2=size=640x360:rate=24 -t 6 -c:v libx264 -pix_fmt yuv420p "$work/clip.mkv"
  snapshot "$root/ajp"
  echo "lecture et instantané : ok"
  if [ -n "$appimage" ]; then
    chmod +x "$appimage"
    snapshot "$appimage" --appimage-extract-and-run
    echo "AppImage : ok"
  fi
else
  echo "::warning::ffmpeg ou xvfb-run absent : lecture et instantané sautés"
fi
echo "$archive : paquet lancé avec succès"
