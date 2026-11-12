#!/usr/bin/env bash
# Les tests sautés ne passent plus pour réussis (P2.39) : leur nombre, lu dans le rapport JUnit de ctest,
# ne dépasse pas le plafond connu de la tâche (tools/ci/skip-ceiling.txt). Un saut de plus est un test qui ne
# tourne plus : relever le plafond se fait exprès, dans le même commit, avec la raison.
#   tools/ci/check-skips.sh build/linux-prefix/junit.xml lavapipe
set -euo pipefail
report="$1"
job="$2"
here="$(cd "$(dirname "$0")" && pwd)"
# Un rapport absent ou vide ne prouve rien : sans cette garde, grep n'écrivait rien et le script finissait sur « réussi »
# (C28-07).
[ -s "$report" ] || { echo "check-skips : rapport JUnit absent ou vide : $report" >&2; exit 1; }
floor_file="$here/tests-floor.txt"
ceiling="$(awk -v job="$job" '$1 == job { print $2 }' "$here/skip-ceiling.txt")"
if [ -z "$ceiling" ]; then
  echo "check-skips : aucun plafond pour la tâche « $job » dans skip-ceiling.txt" >&2
  exit 1
fi
skipped="$(grep -c '<skipped' "$report" || true)"
skipped="${skipped:-0}"  # une ligne par cas sauté
echo "tests sautés : $skipped (plafond $ceiling)"
grep -o '<testcase name="[^"]*"[^>]*status="notrun"' "$report" | sed 's/<testcase name="\([^"]*\)".*/  \1/' || true
if [ "$skipped" -gt "$ceiling" ]; then
  echo "check-skips : $skipped tests sautés, plus que le plafond $ceiling de « $job »" >&2
  exit 1
fi
# Plancher : un rapport vide ou amputé (une cible qui n'a pas été construite, un filtre trop large) ne doit pas passer
# pour un succès (B40-09). Le nombre de cas du rapport ne descend pas sous le plancher de la tâche, tools/ci/tests-floor.txt.
total="$(grep -c '<testcase ' "$report" || true)"
total="${total:-0}"
minimum="$(awk -v job="$job" '$1 == job { print $2 }' "$floor_file" 2>/dev/null)"
if [ -z "$minimum" ]; then
  echo "check-skips : aucun plancher pour la tâche « $job » dans tests-floor.txt" >&2
  exit 1
fi
echo "tests joués ou sautés : $total (plancher $minimum)"
if [ "$total" -lt "$minimum" ]; then
  echo "check-skips : $total cas seulement dans le rapport, moins que le plancher $minimum de « $job »" >&2
  exit 1
fi
