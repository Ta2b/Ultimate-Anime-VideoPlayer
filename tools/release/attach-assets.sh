#!/usr/bin/env bash
# Joint des fichiers à la release d'une étiquette du lecteur, pour les trois workflows de plateforme (release-linux,
# release-windows, release-macos). La release est créée en préversion si elle n'existe pas encore, avec les notes de
# tools/release/notes/<étiquette>.md ; elle n'est jamais rendue finale ni « latest » ici : c'est le job « publier » de
# release.yml qui le fait, une fois les trois plateformes et le manifeste signé joints (A28-03).
#
#   GH_TOKEN=… [AJP_RELEASE_REPO=…] tools/release/attach-assets.sh <étiquette> <fichier>…
#
# Un fichier déjà joint ne se remplace que tant que la release est une préversion : une release publiée ne se
# réécrit pas (P1.16).
set -euo pipefail

[ $# -ge 2 ] || { sed -n '2,10p' "$0"; exit 2; }
tag=$1; shift
root="$(cd "$(dirname "$0")/../.." && pwd)"
repo="${AJP_RELEASE_REPO:-$(sed -n 's/^AJP_RELEASE_REPO=//p' "$root/tools/release/versions.env" | tr -d '\r')}"
[ -n "$repo" ] || { echo "dépôt des releases inconnu" >&2; exit 1; }
notes="$root/tools/release/notes/$tag.md"
[ -f "$notes" ] || notes=/dev/null

# Les trois plateformes peuvent créer la release au même moment : un refus de création ne compte que si la release
# n'existe toujours pas.
gh release view "$tag" -R "$repo" >/dev/null 2>&1 \
  || gh release create "$tag" -R "$repo" --prerelease --latest=false --title "$tag" --notes-file "$notes" \
  || gh release view "$tag" -R "$repo" >/dev/null

pre="$(gh release view "$tag" -R "$repo" --json isPrerelease -q .isPrerelease)"
clobber=()
if [ "$pre" = true ]; then
  clobber=(--clobber)
  # Les notes reprennent celles du dépôt, même si la release a été créée avant (paquet TensorRT joint).
  [ "$notes" = /dev/null ] || gh release edit "$tag" -R "$repo" --notes-file "$notes"
fi
gh release upload "$tag" -R "$repo" "${clobber[@]}" "$@"
echo "joint à $repo $tag : $*"
