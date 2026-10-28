#!/usr/bin/env bash
# Sur le poste NVIDIA : le paquet TensorRT Linux de l'empreinte courante existe-t-il dans le dépôt des releases ?
# Sinon, le construire (préflight Docker) et le joindre à la release fixe « trt-<empreinte> », avec son index
# engines-nvidia.lock. Une seule fois par empreinte, indépendamment des étiquettes du lecteur : Release Linux le
# trouve déjà et n'est plus relancée (docs/nvidia-linux.md).
#
#   tools/release/trt-package.sh            # construit et joint si absent
#   tools/release/trt-package.sh --check    # dit seulement si le paquet existe (code 0) ou manque (code 3)
#
# Variables : AJP_RELEASE_REPO (défaut Ta2b/ajp-releases), AJP_TRT_WORK (dossier de travail, défaut ~/ajp-work/trt) ;
# celles de preflight/nvidia-package.sh (AJP_TRT_ROOT, CUDA_DIR, CCCL_DIR).
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
repo="${AJP_RELEASE_REPO:-Ta2b/ajp-releases}"
fp="$("$HERE/trt-fingerprint.sh" "$ROOT")"
tag="trt-$fp"
echo "empreinte : $fp (release $repo $tag)"

if gh release view "$tag" -R "$repo" --json assets -q '.assets[].name' 2>/dev/null | grep -qx 'engines-nvidia.lock'; then
  echo "le paquet TensorRT de cette empreinte existe déjà : rien à faire"
  exit 0
fi
[ "${1:-}" = "--check" ] && { echo "paquet absent : tools/release/trt-package.sh le construit"; exit 3; }

[ -z "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ] || { echo "$ROOT a des changements locaux : le paquet doit sortir d'un commit" >&2; exit 1; }
work="${AJP_TRT_WORK:-$HOME/ajp-work/trt}/$fp"
AJP_ALLOW_UNTAGGED=1 "$HERE/preflight/nvidia-package.sh" "$ROOT" "$tag" "$work"
AJP_RELEASE_REPO="$repo" "$HERE/attach-nvidia-linux.sh" "$work/package" "$tag" --publish
