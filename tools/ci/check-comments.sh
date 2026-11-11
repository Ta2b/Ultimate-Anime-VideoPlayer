#!/usr/bin/env bash
# Commentaires de src/ (P5.37) : ni nom de personne ni date. La raison d'un choix tient dans le
# commentaire, une ADR (docs/adr/) ou un test ; qui l'a demandé et quand, dans l'historique git.
# Une date s'écrit avec son année (12/11/2026) : « 16/10 » ou « 30/12 » seuls sont des rapports, des échelles ou des compteurs.
set -euo pipefail
cd "$(dirname "$0")/../.."
pattern='(//|#|/\*).*(Tatoub|\b[0-3][0-9]/[01][0-9]/20[0-9]{2}\b)'
if found="$(git grep -n -I -E "$pattern" -- src)"; then
  echo "commentaire avec un nom ou une date (P5.37) :" >&2
  echo "$found" >&2
  exit 1
fi
echo "commentaires : ni nom ni date"
