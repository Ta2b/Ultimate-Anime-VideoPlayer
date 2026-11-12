#!/usr/bin/env bash
# Les tests cachés « nvidia » (anneau CUDA, SPAN FP8, RIFE et DistilDRBA fusionnés, NVDEC) sont enregistrés par la
# configuration (AJP_HARDWARE_TESTS), pas par la carte : une construction CUDA sans carte peut donc le vérifier.
# Sans cette garde, un préréglage qui oublie l'option laisse « ctest » vert sur le PC avec zéro test matériel.
# Usage : tools/ci/check-hardware-tests.sh <dossier de construction> [minimum]
set -euo pipefail
build="${1:?dossier de construction}"
minimum="${2:-6}"
count="$(ctest --test-dir "$build" -N -L nvidia | sed -n 's/^Total Tests: *//p')"
count="${count:-0}"
if (( count < minimum )); then
  echo "::error::$count test(s) étiqueté(s) nvidia dans $build, au moins $minimum attendus (AJP_HARDWARE_TESTS éteint ?)"
  exit 1
fi
echo "tests matériels enregistrés : $count"
