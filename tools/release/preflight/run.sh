#!/usr/bin/env bash
# Construit le paquet Linux comme .github/workflows/release-linux.yml, dans un conteneur
# ubuntu:24.04 (Dockerfile à côté), avant toute poussée sur release/linux.
#
#   tools/release/preflight/run.sh [sortie]             défaut : <dépôt>/build/preflight
#   AJP_WITH_MIGRAPHX=OFF tools/release/preflight/run.sh  sans ROCm (image plus légère)
#   AJP_PREFLIGHT_TESTS=1 tools/release/preflight/run.sh  puis les tests unitaires
#       (build/linux-release-tests, Vulkan logiciel sous Xvfb ; filtre AJP_PREFLIGHT_TESTS_REGEX)
#   tools/release/preflight/run.sh rocm-size            taille du téléchargement et de l'installation ROCm
#
# Le dépôt est monté en lecture seule et copié dans un volume (ajp-preflight-work) : la
# construction n'écrit jamais dans l'arbre de travail. La compilation continue après la première
# erreur (ninja -k 0) pour toutes les montrer en une passe ; journal : <sortie>/build.log.
# Volumes : ajp-preflight-work (construction incrémentale), ajp-preflight-cache (modèles).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
WITH_MIGRAPHX="${AJP_WITH_MIGRAPHX:-ON}"
MIN_FREE_GB="${AJP_PREFLIGHT_MIN_FREE_GB:-12}"
IMAGE=ajp-preflight

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

# Contexte minimal : le Dockerfile et le script des dépendances, seul fichier du dépôt que l'image
# embarque (l'étage deps se reconstruit quand il change).
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/tools/deps"
cp "$HERE/Dockerfile" "$CTX/"
cp "$ROOT/tools/deps/linux-from-source.sh" "$CTX/tools/deps/"
build_image() {  # étage [base de l'étage tests]
  check_disk
  docker build -q -t "$IMAGE:$1" --target "$1" ${2:+--build-arg TEST_BASE="$2"} "$CTX" >/dev/null
  echo "image $IMAGE:$1 prête"
}

if [ "${1:-}" = rocm-size ]; then
  build_image base
  docker run --rm "$IMAGE:base" bash -c '
    set -e
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg >/dev/null
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/7.2 noble main" \
      | sudo tee /etc/apt/sources.list.d/rocm.list >/dev/null
    printf "Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n" | sudo tee /etc/apt/preferences.d/rocm-pin-600 >/dev/null
    sudo apt-get update -qq
    sudo apt-get install --assume-no --no-install-recommends migraphx-dev hip-dev | grep -E "^(Need to get|After this operation)" || true'
  exit 0
fi

OUT="$(mkdir -p "${1:-$ROOT/build/preflight}" && cd "${1:-$ROOT/build/preflight}" && pwd)"
STAGE=openvino
[ "$WITH_MIGRAPHX" = ON ] && STAGE=rocm
if [ -n "${AJP_PREFLIGHT_TESTS:-}" ]; then
  build_image "$STAGE"
  build_image tests "$STAGE"
  STAGE=tests
else
  build_image "$STAGE"
fi
check_disk

start=$(date +%s)
# Un volume neuf appartient à root : il revient à l'utilisateur runner du conteneur.
docker run --rm -u root -v ajp-preflight-work:/w -v ajp-preflight-cache:/c "$IMAGE:$STAGE" \
  chown runner:runner /w /c
set +e
docker run --rm \
  -v "$ROOT:/src:ro" -v "$OUT:/out" \
  -v ajp-preflight-work:/home/runner/work -v ajp-preflight-cache:/home/runner/.cache \
  -e AJP_WITH_MIGRAPHX="$WITH_MIGRAPHX" -e AJP_WITH_OPENVINO=ON \
  -e AJP_PREFLIGHT_TESTS="${AJP_PREFLIGHT_TESTS:-}" -e AJP_PREFLIGHT_TESTS_REGEX="${AJP_PREFLIGHT_TESTS_REGEX:-}" \
  "$IMAGE:$STAGE" bash -c '
    set -uo pipefail
    export OpenVINO_DIR="$(python3 -c "from openvino.utils import get_cmake_path; print(get_cmake_path())")"
    repo=/home/runner/work/repo
    mkdir -p "$repo" || exit 1
    # Copie des sources sans les constructions ni les fichiers de git ; build/ reste dans le volume,
    # le reste est remplacé (un fichier supprimé du dépôt ne survit pas).
    find "$repo" -mindepth 1 -maxdepth 1 ! -name build -exec rm -rf {} +
    tar -C /src --exclude=./build --exclude=./out --exclude=./.git --exclude="./.venv*" -cf - . | tar -C "$repo" -xf -
    cd "$repo" || exit 1
    cmake --preset linux-release -DAJP_WITH_CUDA=OFF -DAJP_WITH_MIGRAPHX="$AJP_WITH_MIGRAPHX" \
      -DAJP_WITH_OPENVINO="$AJP_WITH_OPENVINO" 2>&1 | tee /out/configure.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: configuration en échec"; exit 2; }
    cmake --build --preset linux-release -- -k 0 2>&1 | tee /out/build.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: compilation en échec"; exit 3; }
    tools/release/package-linux.sh /out/package 2>&1 | tee /out/package.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: paquet en échec"; exit 4; }
    echo "PREFLIGHT: paquet construit"
    tar -tzf /out/package/ajp-linux-x64.tar.gz | sort
    [ -n "$AJP_PREFLIGHT_TESTS" ] || exit 0
    tests=build/linux-release-tests
    cmake --preset linux-release -B "$tests" -DAJP_BUILD_TESTS=ON -DAJP_WITH_CUDA=OFF \
      -DAJP_WITH_MIGRAPHX="$AJP_WITH_MIGRAPHX" -DAJP_WITH_OPENVINO="$AJP_WITH_OPENVINO" > /out/tests-configure.log 2>&1 \
      || { echo "PREFLIGHT: configuration des tests en échec"; exit 5; }
    cmake --build "$tests" -- -k 0 > /out/tests-build.log 2>&1 \
      || { echo "PREFLIGHT: compilation des tests en échec (tests-build.log)"; exit 6; }
    xvfb-run -a ctest --test-dir "$tests" --output-on-failure -j "$(nproc)" \
      ${AJP_PREFLIGHT_TESTS_REGEX:+-R "$AJP_PREFLIGHT_TESTS_REGEX"} 2>&1 | tee /out/tests.log
    [ "${PIPESTATUS[0]}" = 0 ] || { echo "PREFLIGHT: tests en échec (tests.log)"; exit 7; }
    echo "PREFLIGHT: tests réussis"
  '
status=$?
set -e
echo "durée : $(( ($(date +%s) - start) / 60 )) min, statut $status ; journaux dans $OUT"
exit $status
