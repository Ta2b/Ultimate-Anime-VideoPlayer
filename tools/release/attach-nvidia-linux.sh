#!/usr/bin/env bash
# Joint le paquet TensorRT construit sur un poste NVIDIA à la release d'une étiquette, AVANT de pousser
# l'étiquette. release-linux.yml télécharge ensuite engines-nvidia.lock de la release et en fusionne
# les moteurs (et leurs fichiers NVIDIA, [[vendor_file]]) dans l'engines.lock qu'il joint (package-linux.sh,
# AJP_ENGINES_LOCK) ; l'archive du moteur reste celle du poste, son SHA-256 est déjà dans l'index.
#
#   1. sur le poste, à l'étiquette visée (le SHA-256 de l'archive en dépend : pas de commit entre-temps) :
#        AJP_WITH_CUDA=ON AJP_TENSORRT_ROOT=<racine TensorRT> AJP_PACKAGE_NVIDIA=1 \
#        AJP_DEPS_PREFIX=<prefixe> AJP_RELEASE_REPO=<dépôt des releases> AJP_RELEASE_TAG=v0.3.0 \
#        tools/release/package-linux.sh <dossier>
#   2. tools/release/attach-nvidia-linux.sh <dossier> v0.3.0              # contrôle seulement
#      tools/release/attach-nvidia-linux.sh <dossier> v0.3.0 --publish    # crée la release si besoin et joint
#   3. git tag v0.3.0 && git push origin v0.3.0                            # la CI fait le reste
#
# AJP_RELEASE_REPO : le dépôt des releases (défaut : la variable AJP_RELEASE_REPO du dépôt, sinon le dépôt
# courant), le même qu'à l'étape 1 : les liens de l'index le nomment.
set -euo pipefail

dir="${1:?dossier de sortie de package-linux.sh}"
tag="${2:?étiquette (v0.3.0)}"
publish=""
[ "${3:-}" = "--publish" ] && publish=1

archive="$dir/ajp-engine-tensorrt-linux-x64.tar.gz"
lock="$dir/engines.lock"
[ -f "$archive" ] || { echo "$archive absent : le paquet TensorRT n'a pas été construit (AJP_PACKAGE_NVIDIA=1)" >&2; exit 1; }
[ -f "$lock" ] || { echo "$lock absent" >&2; exit 1; }

repo="${AJP_RELEASE_REPO:-$(gh variable get AJP_RELEASE_REPO 2>/dev/null || gh repo view --json nameWithOwner -q .nameWithOwner)}"

# Rien de NVIDIA dans l'archive : seulement notre module et trtexec (le lien direct tire le reste).
if tar tzf "$archive" | grep -qiE 'libnvinfer|libnvonnxparser|libnvrtc|libcu[a-z]|builder_resource|nvinfer|cudart'; then
  echo "L'archive contient des fichiers NVIDIA : elle n'a pas été construite en lien direct (sans AJP_NVIDIA_BUNDLE)" >&2
  tar tzf "$archive" | grep -iE 'libnvinfer|libnvonnxparser|libnvrtc|libcu[a-z]|builder_resource|nvinfer|cudart' >&2
  exit 1
fi

sha="$(sha256sum "$archive" | cut -d' ' -f1)"
size="$(stat -c %s "$archive")"
url="https://github.com/$repo/releases/download/$tag/ajp-engine-tensorrt-linux-x64.tar.gz"
# L'index doit nommer cette archive, à cette étiquette et dans ce dépôt, avec ce hachage et cette taille.
for want in "url = \"$url\"" "sha256 = \"$sha\"" "size = $size"; do
  grep -qxF "$want" "$lock" || { echo "$lock ne contient pas : $want (étiquette ou dépôt différents de ceux de la construction ?)" >&2; exit 1; }
done
vendor="$(grep -c '^\[\[vendor_file\]\]' "$lock" || true)"
[ "$vendor" -ge 1 ] || { echo "$lock n'a aucun [[vendor_file]] : sans eux le moteur s'installerait sans les bibliothèques de NVIDIA" >&2; exit 1; }

# engines-nvidia.lock : les moteurs et fichiers de NVIDIA seulement (TensorRT, et RTX HDR s'il est là).
out="$dir/engines-nvidia.lock"
awk '
  BEGIN { print "version = 1" }
  function flush() { if (block != "" && (id == "tensorrt" || id == "maxine_vfx")) printf "\n%s", block; block = ""; id = "" }
  /^\[\[(engine|vendor_file)\]\]$/ { flush(); block = $0 "\n"; inside = 1; next }
  /^\[/ { flush(); inside = 0; next }
  inside && /^id = / { id = $3; gsub(/"/, "", id) }
  inside && NF { block = block $0 "\n" }
  END { flush() }' "$lock" > "$out"
echo "archive      : $archive ($size octets, sha256 $sha)"
echo "index NVIDIA : $out ($(grep -c '^\[\[engine\]\]' "$out") moteur(s), $(grep -c '^\[\[vendor_file\]\]' "$out") fichier(s) NVIDIA)"
echo "release      : $repo $tag"

if [ -z "$publish" ]; then
  echo "Contrôle seulement. Pour joindre : relancer avec --publish, puis pousser l'étiquette $tag."
  exit 0
fi
gh release view "$tag" -R "$repo" >/dev/null 2>&1 || gh release create "$tag" -R "$repo" --prerelease --title "$tag" --notes "$tag"
gh release upload "$tag" -R "$repo" --clobber "$archive" "$out"
echo "Joint à $repo $tag. Pousser maintenant l'étiquette : git tag $tag && git push origin $tag"
