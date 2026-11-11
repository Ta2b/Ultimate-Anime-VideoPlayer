#!/usr/bin/env bash
# Contrôle d'une archive de moteur avant de la joindre à une release : elle ne contient que notre module et trtexec,
# jamais un fichier de NVIDIA (CLAUDE.md : aucun fichier de NVIDIA dans nos archives). Liste blanche, pas liste noire :
# une bibliothèque que la construction copierait en plus (libnvJitLink, libnvptxcompiler, libnvfatbin…) arrête
# la publication au lieu de passer (B51-10).
#
#   tools/release/check-engine-archive.sh <archive.tar.gz>
set -euo pipefail

[ $# -eq 1 ] || { sed -n '2,7p' "$0"; exit 2; }
archive=$1
[ -f "$archive" ] || { echo "$archive absent" >&2; exit 1; }
members="$(tar tzf "$archive")" || { echo "$archive : archive illisible" >&2; exit 1; }
bad=""
module=0 trtexec=0
while IFS= read -r member; do
  name="${member#./}"
  name="${name%/}"
  case "$name" in
    "" | .) ;;  # la racine de l'archive
    ajp_backend_tensorrt.so | ajp_backend_tensorrt.dll) module=1 ;;
    trtexec | trtexec.exe) trtexec=1 ;;
    *) bad="$bad$member"$'\n' ;;
  esac
done <<<"$members"
if [ -n "$bad" ]; then
  echo "$archive contient autre chose que notre module et trtexec (un fichier de NVIDIA ?) :" >&2
  printf '%s' "$bad" >&2
  exit 1
fi
[ "$module" -eq 1 ] || { echo "$archive : le module ajp_backend_tensorrt est absent" >&2; exit 1; }
[ "$trtexec" -eq 1 ] || { echo "$archive : trtexec est absent" >&2; exit 1; }
echo "$archive : module et trtexec seulement"
