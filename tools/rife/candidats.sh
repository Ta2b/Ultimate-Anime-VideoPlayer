#!/usr/bin/env bash
# Les deux RIFE candidats au catalogue, pour les mesurer face aux livrés (docs/rife-candidats.md) :
#
#   rife_v4.26.onnx       conversion fp16 d'animejanai-inference, de la même archive (models-rife-fp16-1) et du même
#                         outil que le rife_v4.25.onnx livré : la comparaison 4.25 contre 4.26 ne porte que sur
#                         les poids ;
#   rife_v4.25_lite.onnx  absent de cette archive : l'export ONNX officiel de vs-mlrt (release external-models,
#                         dossier rife/, 11 plans, FP32) passé en fp16 par tools/rife/convert_rife_fp16.py, dont
#                         les Resize en FP32 évitent les NaN de TensorRT 11.3 sur ce réseau.
#
# Rien n'est repris d'un autre lecteur : ce sont les fichiers des auteurs des exports, vérifiés par SHA-256.
#
#   tools/rife/candidats.sh <dest> [dossier-de-cache]
#
# Il faut 7z, python3 avec numpy et onnx (pour la conversion). L'archive d'animejanai (322 Mo) est celle de
# tools/release/fetch-models.sh : son cache sert aux deux.
set -euo pipefail

DEST="${1:?dossier de destination}"
CACHE="${2:-${XDG_CACHE_HOME:-$HOME/.cache}/ajp-release-models}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SEVENZ="$(command -v 7z || command -v 7zz || true)"
[ -n "$SEVENZ" ] || { echo "7z manquant (apt install p7zip-full, brew install sevenzip)" >&2; exit 1; }
if command -v sha256sum >/dev/null; then SHA256=(sha256sum); else SHA256=(shasum -a 256); fi
mkdir -p "$DEST" "$CACHE"

check() { # sha fichier
  echo "$1  $2" | "${SHA256[@]}" -c --status - || { echo "SHA-256 inattendu : $2" >&2; exit 1; }
}
fetch() { # url sha fichier
  if ! echo "$2  $3" | "${SHA256[@]}" -c --status - >/dev/null 2>&1; then
    curl -fsSL --retry 5 --retry-delay 5 -o "$3" "$1"
    check "$2" "$3"
  fi
}

ANIMEJANAI_URL="https://github.com/the-database/animejanai-inference/releases/download/models-rife-fp16-1/rife-fp16-1.7z"
ANIMEJANAI_SHA=3fcf3429f3bb661d15c59aa58ed7ba28c0ca7f79e2388f8e0165dd8bbe0648f9
V426_SHA=fd6b06538898c9a94d20f0fe55febceadbcde5ad64952240733b7275ef4c9d59
VSMLRT_URL="https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.25_lite.7z"
VSMLRT_SHA=7d53e29fff5e67345b19f4ce97dfd1e34b490eedc752df2236904fda1a13842c
LITE_FP32_SHA=026605086bd5782581cb3d846e16eec326d368af109f78cf8113883bff2e9e66
LITE_FP16_SHA=124c2e158f626f81d762256b2b130fe609ce28119221b3adcd471e6e0b75bf92  # sortie de convert_rife_fp16.py

fetch "$ANIMEJANAI_URL" "$ANIMEJANAI_SHA" "$CACHE/rife-fp16-1.7z"
"$SEVENZ" e -y -bso0 -o"$DEST" "$CACHE/rife-fp16-1.7z" rife_v4.26.onnx
check "$V426_SHA" "$DEST/rife_v4.26.onnx"

fetch "$VSMLRT_URL" "$VSMLRT_SHA" "$CACHE/rife_v4.25_lite.7z"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
"$SEVENZ" x -y -bso0 -o"$work" "$CACHE/rife_v4.25_lite.7z" rife/rife_v4.25_lite.onnx
check "$LITE_FP32_SHA" "$work/rife/rife_v4.25_lite.onnx"
python3 -I "$ROOT/tools/rife/convert_rife_fp16.py" "$work/rife" "$DEST" rife_v4.25_lite.onnx
# Une autre version d'onnx peut sérialiser autrement : le réseau reste le même, seul l'avertissement le dit.
echo "$LITE_FP16_SHA  $DEST/rife_v4.25_lite.onnx" | "${SHA256[@]}" -c --status - ||
  echo "note : rife_v4.25_lite.onnx diffère de la conversion de référence (autre version d'onnx ?)" >&2

echo "candidats prêts dans $DEST : rife_v4.26.onnx, rife_v4.25_lite.onnx"
