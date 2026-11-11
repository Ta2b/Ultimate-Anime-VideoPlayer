#!/usr/bin/env bash
# L'index NVIDIA signé d'une release (engines-nvidia.lock) nomme, pour chaque plateforme, le même paquet TensorRT que
# l'engines.lock du paquet de cette plateforme : même adresse, même SHA-256, même taille. Sinon des utilisateurs
# seraient envoyés vers un autre module que celui livré avec le lecteur (B51-01).
#
#   tools/release/check-nvidia-index.sh <engines-nvidia.lock> <engines.lock Linux> <engines.lock Windows>
set -euo pipefail

[ $# -eq 3 ] || { sed -n '2,7p' "$0"; exit 2; }
index=$1 linux=$2 windows=$3

# « plateforme|url|sha256|size » de chaque [[engine]] tensorrt.
engines() {
  tr -d '\r' < "$1" | awk '
    function flush() { if (inside && id == "tensorrt") print platform "|" url "|" sha "|" size; id = platform = url = sha = size = "" }
    /^\[\[/ { flush(); inside = ($0 == "[[engine]]"); next }
    /^\[/ { flush(); inside = 0; next }
    inside && /^id = / { id = $3; gsub(/"/, "", id) }
    inside && /^platform = / { platform = $3; gsub(/"/, "", platform) }
    inside && /^url = / { url = $3; gsub(/"/, "", url) }
    inside && /^sha256 = / { sha = $3; gsub(/"/, "", sha) }
    inside && /^size = / { size = $3 }
    END { flush() }'
}
fail=0
for pair in "linux:$linux" "windows:$windows"; do
  platform=${pair%%:*} lock=${pair#*:}
  want="$(engines "$lock" | grep "^$platform|" | sort || true)"
  got="$(engines "$index" | grep "^$platform|" | sort || true)"
  if [ "$want" != "$got" ]; then
    echo "::error::$platform : l'index signé ne nomme pas le paquet TensorRT de l'engines.lock du paquet" >&2
    echo "  paquet : ${want:-(aucun)}" >&2
    echo "  index  : ${got:-(aucun)}" >&2
    fail=1
  else
    echo "$platform : index et paquet d'accord (${got:-aucun TensorRT})"
  fi
done
exit "$fail"
