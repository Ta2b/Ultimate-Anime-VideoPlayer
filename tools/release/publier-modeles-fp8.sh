#!/usr/bin/env bash
# Publie les graphes FP8 de l'interpolation dans la release models-interp-1 de Ta2b/ajp-releases
# (docs/fp8-interpolation.md). À lancer par Tatoub, jamais par un agent : rien n'est envoyé sans --publier.
#
#   tools/release/publier-modeles-fp8.sh DOSSIER_SORTIE [--publier]
#
# DOSSIER_SORTIE : celui de tools/fp8/build_interp_fp8.sh (models/ et SHA256SUMS.fp8). Sans --publier : vérifie les
# empreintes, montre ce qui serait envoyé et le SHA256SUMS fusionné, n'envoie rien.
#
# Un groupe, facultatif (dans SHA256SUMS.facultatif, JAMAIS téléchargé par fetch-models.sh) : distil_drba_v2_lite{,_s05}_fp8.onnx,
#    11 % plus rapides que le DistilDRBA entier en FP16 mais plus lents que le DistilDRBA fusionné, qu'ils coupent :
#    pour ceux qui les veulent (à poser à côté de distil_drba_v2_lite.onnx).
# Aucun graphe FP8 de RIFE : plus lents ou hors des gardes de qualité. Les paires FP8 de GMFSS, retiré du lecteur
# (0.10.0), sortent du SHA256SUMS de la release.
set -euo pipefail

OUT="${1:?dossier de sortie de build_interp_fp8.sh}"
PUBLISH=0
[ "${2:-}" = "--publier" ] && PUBLISH=1
REPO="${AJP_RELEASE_REPO:-Ta2b/ajp-releases}"
TAG=models-interp-1
MODELS="$OUT/models"
OPTIONAL=(distil_drba_v2_lite_fp8.onnx distil_drba_v2_lite_s05_fp8.onnx)

[ -f "$OUT/SHA256SUMS.fp8" ] || { echo "SHA256SUMS.fp8 absent de $OUT" >&2; exit 1; }
(cd "$MODELS" && sha256sum -c ../SHA256SUMS.fp8)

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
# SHA256SUMS de la release tel qu'il est, sans les graphes de GMFSS.
if gh release download "$TAG" -R "$REPO" -p SHA256SUMS -D "$work" 2>/dev/null; then
  grep -v ' gmfss_' "$work/SHA256SUMS" >"$work/SHA256SUMS.new" || true
else
  echo "release $TAG sans SHA256SUMS : un nouveau sera écrit" >&2
  : >"$work/SHA256SUMS.new"
fi
: >"$work/SHA256SUMS.facultatif"
for f in "${OPTIONAL[@]}"; do (cd "$MODELS" && sha256sum "$f") | sed 's#  # #' >>"$work/SHA256SUMS.facultatif"; done

echo "== SHA256SUMS fusionné (lu par fetch-models.sh)"; cat "$work/SHA256SUMS.new"
echo "== SHA256SUMS.facultatif (non lu par fetch-models.sh)"; cat "$work/SHA256SUMS.facultatif"
echo "== tailles"; (cd "$MODELS" && ls -l "${OPTIONAL[@]}" | awk '{print $5, $9}')

if [ "$PUBLISH" -ne 1 ]; then
  echo "Essai à blanc : rien n'est envoyé. Relancer avec --publier pour téléverser dans $REPO ($TAG)."
  exit 0
fi

# --latest=false : la release des modèles ne doit pas devenir la « dernière » release du dépôt.
gh release view "$TAG" -R "$REPO" >/dev/null 2>&1 \
  || gh release create "$TAG" -R "$REPO" --latest=false --title "Modèles d'interpolation (DistilDRBA)" \
       --notes "Graphes ONNX d'interpolation (DistilDRBA) ; SHA256SUMS les empreintes."
cp "$work/SHA256SUMS.new" "$work/SHA256SUMS"
files=()
for f in "${OPTIONAL[@]}"; do files+=("$MODELS/$f"); done
gh release upload "$TAG" -R "$REPO" --clobber "${files[@]}" "$work/SHA256SUMS" "$work/SHA256SUMS.facultatif"
echo "Publié : https://github.com/$REPO/releases/tag/$TAG"
