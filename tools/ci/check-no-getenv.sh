#!/usr/bin/env bash
# Aucune option ne vient de l'environnement : profil ou DebugFlags seulement.
set -euo pipefail
if git grep -nE '\b(std::)?(getenv|secure_getenv|_wgetenv|GetEnvironmentVariable[AW]?)\s*\(' -- 'src/*'; then
  echo "::error::getenv est interdit dans src/ (règle « aucune bascule cachée »)"
  exit 1
fi
