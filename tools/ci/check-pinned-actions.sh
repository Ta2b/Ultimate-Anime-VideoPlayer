#!/usr/bin/env bash
# Chaque action tierce des workflows est épinglée par son SHA complet (P1.20) : une étiquette
# déplacée chez l'auteur de l'action ne change pas ce qui tourne ici. Dependabot propose les mises à jour.
set -euo pipefail
cd "$(dirname "$0")/../.."
bad=$(grep -nE '^\s*-?\s*uses:\s*' .github/workflows/*.yml | grep -vE 'uses:\s*\./' | grep -vE '@[0-9a-f]{40}(\s|$)' || true)
if [ -n "$bad" ]; then
  echo "actions non épinglées par SHA :" >&2
  echo "$bad" >&2
  exit 1
fi
