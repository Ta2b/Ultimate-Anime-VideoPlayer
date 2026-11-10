#!/usr/bin/env bash
# Écrit l'index des moteurs NVIDIA d'une release (engines-nvidia.lock), que le lecteur relit au lancement
# (src/app/play/engine_online.h) : les blocs [[engine]] et [[vendor_file]] de tensorrt pris dans les index donnés.
# Pour chaque plateforme, le premier index qui en a gagne : passer d'abord celui joint après coup à la release
# (attach-nvidia-*), puis ceux livrés dans les archives.
#
#   tools/release/nvidia-index.sh <sortie> <index>...
set -euo pipefail

[ $# -ge 2 ] || { sed -n '2,7p' "$0"; exit 2; }
out=$1
shift
seen=" "
{
  echo "version = 1"
  for lock in "$@"; do
    [ -s "$lock" ] || continue
    blocks="$(tr -d '\r' < "$lock" | awk -v seen="$seen" '
      function flush() {
        if (block != "" && id == "tensorrt" && index(seen, " " platform " ") == 0) { printf "\n%s", block; got[platform] = 1 }
        block = ""; id = ""; platform = ""
      }
      /^\[\[(engine|vendor_file)\]\]$/ { flush(); block = $0 "\n"; inside = 1; next }
      /^\[/ { flush(); inside = 0; next }
      inside && /^id = / { id = $3; gsub(/"/, "", id) }
      inside && /^platform = / { platform = $3; gsub(/"/, "", platform) }
      inside && NF { block = block $0 "\n" }
      END { flush(); for (p in got) printf "\n#plateforme %s\n", p }')"
    for p in $(printf '%s\n' "$blocks" | sed -n 's/^#plateforme //p'); do seen="$seen$p "; done
    printf '%s\n' "$blocks" | grep -v '^#plateforme ' | sed '/./,$!d'
  done
} > "$out"
echo "$out : $(grep -c '^\[\[engine\]\]' "$out" || true) moteur(s) ($seen), $(grep -c '^\[\[vendor_file\]\]' "$out" || true) fichier(s) NVIDIA"
