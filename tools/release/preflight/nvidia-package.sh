#!/usr/bin/env bash
# Construit le paquet du moteur TensorRT (Linux) dans le conteneur ubuntu:24.04 de préflight, au commit de
# l'étiquette, pour attach-nvidia-linux.sh. Ce que l'on fait à la main pour chaque release depuis la 0.3, écrit une fois.
#
#   git worktree add ~/ajp-work/rel-v0.5.0 v0.5.0
#   tools/release/preflight/nvidia-package.sh ~/ajp-work/rel-v0.5.0 v0.5.0 ~/ajp-work/rel-v0.5.0-nv
#   tools/release/attach-nvidia-linux.sh ~/ajp-work/rel-v0.5.0-nv/package v0.5.0 --publish
#
# Entrées : le dépôt à l'étiquette (monté en lecture seule), l'étiquette, le dossier de sortie.
# Variables : AJP_TRT_ROOT (défaut ~/ajp-trt-release : bin/trtexec construit en conteneur, lib/ des ressources du
# constructeur), AJP_RELEASE_REPO (défaut Ta2b/ajp-releases), CUDA_DIR (défaut /opt/cuda), CCCL_DIR (/usr/include/cccl).
#
# L'IMAGE EST RECONSTRUITE d'abord (comme run.sh) : une image plus ancienne que tools/deps/linux-from-source.sh embarque
# un FFmpeg d'avant --disable-bzlib, et le contrôle de portabilité de package-linux.sh (libbz2.so.1.0 n'existe que sous
# Debian et Ubuntu) refuse alors le paquet. Ce contrôle n'est jamais contourné ici.
set -euo pipefail

REPO="${1:?dépôt a la version etiquetee (worktree)}"
TAG="${2:?etiquette, par exemple v0.5.0}"
OUT="$(mkdir -p "${3:?dossier de sortie}" && cd "$3" && pwd)"
REPO="$(cd "$REPO" && pwd)"
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
TRT="${AJP_TRT_ROOT:-$HOME/ajp-trt-release}"
RELEASE_REPO="${AJP_RELEASE_REPO:-Ta2b/ajp-releases}"
CUDA="${CUDA_DIR:-/opt/cuda}"
CCCL="${CCCL_DIR:-/usr/include/cccl}"
IMAGE=ajp-preflight
STAGE=rocm

[ -x "$TRT/bin/trtexec" ] || { echo "$TRT/bin/trtexec absent (AJP_TRT_ROOT)" >&2; exit 1; }
[ -d "$CUDA" ] || { echo "$CUDA absent (CUDA_DIR)" >&2; exit 1; }
[ "$(git -C "$REPO" rev-parse HEAD)" = "$(git -C "$REPO" rev-list -n1 "$TAG")" ] \
  || { echo "$REPO n'est pas à l'étiquette $TAG : le SHA-256 de l'archive en dépend" >&2; exit 1; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no)" ] || { echo "$REPO a des changements locaux" >&2; exit 1; }

# L'image, depuis les fichiers du dépôt à l'étiquette (le contexte de run.sh).
CTX="$(mktemp -d)"
trap 'rm -rf "$CTX"' EXIT
mkdir -p "$CTX/tools/deps" "$CTX/tools/release"
cp "$REPO/tools/release/preflight/Dockerfile" "$CTX/"
cp "$REPO/tools/deps/linux-from-source.sh" "$CTX/tools/deps/"
cp "$REPO/tools/release/versions.env" "$REPO/tools/release/rocm-apt.sh" "$CTX/tools/release/"
docker build -q -t "$IMAGE:$STAGE" --target "$STAGE" "$CTX" >/dev/null
echo "image $IMAGE:$STAGE à jour"

docker run --rm -u root -v ajp-preflight-work-cuda:/w -v ajp-preflight-cache:/c "$IMAGE:$STAGE" chown runner:runner /w /c
docker run --rm \
  -v "$REPO:/src:ro" -v "$OUT:/out" \
  -v ajp-preflight-work-cuda:/home/runner/work -v ajp-preflight-cache:/home/runner/.cache \
  -v "$CUDA:/opt/cuda:ro" -v "$CUDA:/usr/local/cuda:ro" -v "$CCCL:/usr/include/cccl:ro" -v "$TRT:/trt:ro" \
  -e AJP_WITH_CUDA=ON -e AJP_TENSORRT_ROOT=/trt -e AJP_WITHOUT_MODELS=1 -e AJP_PACKAGE_NVIDIA=1 \
  -e AJP_RELEASE_REPO="$RELEASE_REPO" -e AJP_RELEASE_TAG="$TAG" \
  "$IMAGE:$STAGE" bash -c '
    set -uo pipefail
    export PATH=/opt/cuda/bin:$PATH CUDACXX=/opt/cuda/bin/nvcc
    repo=/home/runner/work/repo
    mkdir -p "$repo" || exit 1
    find "$repo" -mindepth 1 -maxdepth 1 ! -name build -exec rm -rf {} +
    tar -C /src --exclude=./build --exclude=./out --exclude=./.git --exclude="./.venv*" -cf - . | tar -C "$repo" -xf -
    cd "$repo" || exit 1
    tools/release/package-linux.sh /out/package 2>&1 | tee /out/package.log
    exit "${PIPESTATUS[0]}"
  '
echo "paquet : $OUT/package ; ensuite : tools/release/attach-nvidia-linux.sh $OUT/package $TAG --publish"
