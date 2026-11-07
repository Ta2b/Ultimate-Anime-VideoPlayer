#!/usr/bin/env bash
# Les tests sautés ne passent plus pour réussis (P2.39) : leur nombre, lu dans le rapport JUnit de ctest,
# ne dépasse pas le plafond connu de la tâche (tools/ci/skip-ceiling.txt). Un saut de plus est un test qui ne
# tourne plus : relever le plafond se fait exprès, dans le même commit, avec la raison.
#   tools/ci/check-skips.sh build/linux-prefix/junit.xml lavapipe
set -euo pipefail
report="$1"
job="$2"
here="$(cd "$(dirname "$0")" && pwd)"
ceiling="$(awk -v job="$job" '$1 == job { print $2 }' "$here/skip-ceiling.txt")"
if [ -z "$ceiling" ]; then
  echo "check-skips : aucun plafond pour la tâche « $job » dans skip-ceiling.txt" >&2
  exit 1
fi
skipped="$(grep -c '<skipped' "$report" || true)"  # une ligne par cas sauté
echo "tests sautés : $skipped (plafond $ceiling)"
grep -o '<testcase name="[^"]*"[^>]*status="notrun"' "$report" | sed 's/<testcase name="\([^"]*\)".*/  \1/' || true
if [ "$skipped" -gt "$ceiling" ]; then
  echo "check-skips : $skipped tests sautés, plus que le plafond $ceiling de « $job »" >&2
  exit 1
fi
