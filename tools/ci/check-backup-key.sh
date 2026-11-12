#!/usr/bin/env bash
# Une release finale ne part pas sans clé de secours compilée dans le lecteur (C51-04) : sans elle, la clé de
# signature active (secret CI, seule copie en ligne) ne peut pas être révoquée chez les lecteurs déjà installés.
# Une préversion (étiquette avec « - ») n'en a pas besoin.
#
#   tools/ci/check-backup-key.sh <étiquette> [src/app/CMakeLists.txt]
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,7p' "$0"; exit 2; }
tag=$1
cmake="${2:-$(cd "$(dirname "$0")/../.." && pwd)/src/app/CMakeLists.txt}"
[[ "$tag" == *-* ]] && { echo "$tag : préversion, clé de secours facultative"; exit 0; }
active="$(sed -n 's/^set(AJP_UPDATE_PUBKEY "\([0-9a-fA-F]*\)".*/\1/p' "$cmake" | tr 'A-F' 'a-f')"
backup="$(sed -n 's/^set(AJP_UPDATE_PUBKEY_BACKUP "\([0-9a-fA-F]*\)".*/\1/p' "$cmake" | tr 'A-F' 'a-f')"
if [[ ! "$backup" =~ ^[0-9a-f]{64}$ ]]; then
  echo "::error::$tag : AJP_UPDATE_PUBKEY_BACKUP est vide dans src/app/CMakeLists.txt. Créer la paire de secours hors ligne et compiler sa clé publique avant une release finale (docs/signature-des-releases.md, « Clé de secours »)." >&2
  exit 1
fi
if [ "$backup" = "$active" ]; then
  echo "::error::$tag : la clé de secours est la clé active : elle ne protège de rien." >&2
  exit 1
fi
echo "$tag : clé de secours compilée"
