#!/usr/bin/env bash
# Construit le paquet Linux et passe ses tests comme .github/workflows/release-linux.yml, dans un
# conteneur ubuntu:24.04 (Dockerfile à côté), avant toute poussée sur release/linux.
#
#   tools/release/preflight/run.sh [sortie]             défaut : <dépôt>/build/preflight
#   KIRARI_PREFLIGHT_TESTS=0 tools/release/preflight/run.sh  le paquet seul, sans les tests unitaires
#       (par défaut ils suivent le paquet : build/linux-release-tests, Vulkan logiciel et couche de
#       validation sous Xvfb, comme la tâche tests du workflow ; filtre KIRARI_PREFLIGHT_TESTS_REGEX)
#
# Le dépôt est monté en lecture seule et copié dans un volume (kirari-preflight-work) : la
# construction n'écrit jamais dans l'arbre de travail. La compilation continue après la première
# erreur (ninja -k 0) pour toutes les montrer en une passe ; journal : <sortie>/build.log.
# Volumes : kirari-preflight-work (construction incrémentale), kirari-preflight-cache (modèles).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
TESTS="${KIRARI_PREFLIGHT_TESTS:-1}"
MIN_FREE_GB="${KIRARI_PREFLIGHT_MIN_FREE_GB:-12}"
IMAGE=kirari-preflight

free_gb() {
  local dir
  dir="$(docker info -f '{{.DockerRootDir}}')"
  df -BG --output=avail "$dir" 2>/dev/null | tail -1 | tr -dc 0-9 \
    || df -BG --output=avail / | tail -1 | tr -dc 0-9
}
check_disk() {
  local free; free="$(free_gb)"
  echo "place libre (Docker) : ${free} Go"
  (( free >= MIN_FREE_GB )) || { echo "moins de ${MIN_FREE_GB} Go libres : arrêt" >&2; exit 1; }
}

# Contexte minimal : le Dockerfile, les versions et paquets communs et le script
# des dépendances, seuls fichiers du dépôt que l'image embarque (un étage se reconstruit quand
# ce qu'il lit change).
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/tools/deps" "$CTX/tools/release"
cp "$HERE/Dockerfile" "$CTX/"
cp "$ROOT/tools/deps/linux-from-source.sh" "$CTX/tools/deps/"
cp "$ROOT/tools/release/versions.env" "$CTX/tools/release/"
mkdir -p "$CTX/ports/libplacebo" "$CTX/ports/libass"
cp "$ROOT/ports/libplacebo/0001-kirari-present-timing.patch" "$CTX/ports/libplacebo/"
cp "$ROOT/ports/libass/rendu-sur-plusieurs-fils.patch" "$CTX/ports/libass/"
build_image() {  # étage
  check_disk
  docker build -q -t "$IMAGE:$1" --target "$1" "$CTX" >/dev/null
  echo "image $IMAGE:$1 prête"
}

OUT="$(mkdir -p "${1:-$ROOT/build/preflight}" && cd "${1:-$ROOT/build/preflight}" && pwd)"
STAGE=deps
if [ "$TESTS" != 0 ]; then
  build_image tests
  STAGE=tests
else
  build_image "$STAGE"
fi
check_disk

start=$(date +%s)
# Un volume neuf appartient à root : il revient à l'utilisateur runner du conteneur.
docker run --rm -u root -v kirari-preflight-work:/w -v kirari-preflight-cache:/c "$IMAGE:$STAGE" \
  chown runner:runner /w /c
set +e
docker run --rm \
  -v "$ROOT:/src:ro" -v "$OUT:/out" \
  -v kirari-preflight-work:/home/runner/work -v kirari-preflight-cache:/home/runner/.cache \
  -e KIRARI_PREFLIGHT_TESTS="$TESTS" -e KIRARI_PREFLIGHT_TESTS_REGEX="${KIRARI_PREFLIGHT_TESTS_REGEX:-}" \
  -e KIRARI_TEST_VULKAN_VALIDATION=require \
  "$IMAGE:$STAGE" bash -c '
    set -uo pipefail
    repo=/home/runner/work/repo
    mkdir -p "$repo" || exit 1
    # Copie des sources sans les constructions ni les fichiers de git ; build/ reste dans le volume,
    # le reste est remplacé (un fichier supprimé du dépôt ne survit pas).
    find "$repo" -mindepth 1 -maxdepth 1 ! -name build -exec rm -rf {} +
    tar -C /src --exclude=./build --exclude=./out --exclude=./.git --exclude="./.venv*" -cf - . | tar -C "$repo" -xf -
    cd "$repo" || exit 1
    cmake --preset linux-release -DKIRARI_WITH_CUDA=OFF 2>&1 | tee /out/configure.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: configuration en échec"; exit 2; }
    cmake --build --preset linux-release -- -k 0 2>&1 | tee /out/build.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: compilation en échec"; exit 3; }
    tools/release/package-linux.sh /out/package 2>&1 | tee /out/package.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: paquet en échec"; exit 4; }
    echo "PREFLIGHT: paquet construit"
    tar -tzf /out/package/kirari-linux-x64.tar.gz | sort
    [ "$KIRARI_PREFLIGHT_TESTS" != 0 ] || exit 0
    tests=build/linux-release-tests
    cmake --preset linux-release -B "$tests" -DKIRARI_BUILD_TESTS=ON -DKIRARI_WITH_CUDA=OFF > /out/tests-configure.log 2>&1 \
      || { echo "PREFLIGHT: configuration des tests en échec"; exit 5; }
    cmake --build "$tests" -- -k 0 > /out/tests-build.log 2>&1 \
      || { echo "PREFLIGHT: compilation des tests en échec (tests-build.log)"; exit 6; }
    xvfb-run -a ctest --test-dir "$tests" --output-on-failure -j "$(nproc)" \
      ${KIRARI_PREFLIGHT_TESTS_REGEX:+-R "$KIRARI_PREFLIGHT_TESTS_REGEX"} 2>&1 | tee /out/tests.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: tests en échec (tests.log)"; exit 7; }
    echo "PREFLIGHT: tests réussis"
  '
status=$?
set -e
echo "durée : $(( ($(date +%s) - start) / 60 )) min, statut $status ; journaux dans $OUT"
exit $status
