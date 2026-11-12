#!/usr/bin/env bash
# Le verdict reproductible sur le PC NVIDIA (CachyOS) : toute la suite, puis les tests cachés sur la vraie carte
# et les modèles installés (étiquettes nvidia|gpu|models), un JUnit, et un test matériel sauté compte comme un
# échec (C40-01, C40-04). À lancer avant chaque étiquette ; le résultat se joint à la release.
#   tools/test-pc.sh            construit linux-prefix-cuda (sm_89), puis teste.
# Prérequis : AJP_DEPS_PREFIX, AJP_TENSORRT_ROOT, AJP_MODELS_DIR (le dossier de tools/release/fetch-models.sh).
set -euo pipefail
cd "$(dirname "$0")/.."
cmake --preset linux-prefix-cuda -DAJP_HARDWARE_TESTS_STRICT=ON ${AJP_TENSORRT_ROOT:+"-DAJP_TENSORRT_ROOT=$AJP_TENSORRT_ROOT"}
cmake --build --preset linux-prefix-cuda
tools/ci/check-hardware-tests.sh build/linux-prefix-cuda
junit="build/linux-prefix-cuda/junit-pc.xml"
status=0
ctest --preset linux-prefix-cuda --output-junit "$PWD/$junit" || status=$?
ctest --preset linux-prefix-cuda-hardware --output-junit "$PWD/build/linux-prefix-cuda/junit-pc-hardware.xml" || status=$?
# Aucun test matériel sauté : un pilote absent ou un chemin de modèles faux ne passe pas pour un succès.
skipped="$(grep -c '<skipped' build/linux-prefix-cuda/junit-pc-hardware.xml || true)"
echo "tests matériels sautés : $skipped"
if [ "$skipped" -gt 0 ] || [ "$status" -ne 0 ]; then
  echo "test-pc : échec (statut ctest $status, $skipped sautés)" >&2
  exit 1
fi
echo "test-pc : vert. JUnit : $junit"
