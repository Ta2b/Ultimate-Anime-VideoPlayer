#!/usr/bin/env bash
# Publie le SPAN des vieux animes en SD (AniSD AC, docs/modele-sd.md) dans la release models-span-sd-1 de
# Ta2b/ajp-releases. À lancer par Tatoub, jamais par un agent : rien n'est envoyé sans --publier.
#
#   tools/release/publier-modele-sd.sh DOSSIER [--publier]
#
# DOSSIER : celui qui contient le fichier source de Sirosky (2x_AniSD_AC_G6i2b_SPAN_190K_fp32.onnx, release AniSD
# d'Upscale-Hub) ; le fichier FP16 (tools/span/convert_anisd_fp16.py) y est refait s'il manque. Sans --publier :
# vérifie l'empreinte de la source, reconvertit, vérifie celle du FP16 et montre ce qui serait envoyé et le
# SHA256SUMS fusionné ; n'envoie rien. Avec --publier : crée la release si elle manque (jamais la « dernière »),
# ajoute le fichier et son SHA256SUMS sans écraser les lignes des autres fichiers de la release.
set -euo pipefail

DIR="${1:?dossier du fichier fp32 AniSD}"
PUBLISH=0
[ "${2:-}" = "--publier" ] && PUBLISH=1
REPO="${AJP_RELEASE_REPO:-Ta2b/ajp-releases}"
TAG=models-span-sd-1
HERE="$(cd "$(dirname "$0")" && pwd)"
SOURCE=2x_AniSD_AC_G6i2b_SPAN_190K_fp32.onnx
SOURCE_SHA=c0d52ae736a41803af7389911333f6a70f4b8f4bee795c01713a3b8bd81b284d  # publiée par Sirosky (release AniSD)
OUTPUT=2x_AniSD_AC_G6i2b_SPAN_190K_fp16.onnx
OUTPUT_SHA=6cdca47792cf40516443e40dfddfea26c5141461861587fba5373a3a3fd82bcb  # sortie de convert_anisd_fp16.py (onnx 1.x)
URL="https://github.com/Sirosky/Upscale-Hub/releases/download/AniSD/$SOURCE"

[ -f "$DIR/$SOURCE" ] || { echo "$SOURCE absent de $DIR : télécharger $URL" >&2; exit 1; }
echo "$SOURCE_SHA  $DIR/$SOURCE" | sha256sum -c --status - || { echo "SHA-256 inattendu : $SOURCE" >&2; exit 1; }

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
python3 -I "$HERE/../span/convert_anisd_fp16.py" "$DIR/$SOURCE" "$work/$OUTPUT"
actual="$(sha256sum "$work/$OUTPUT" | cut -d' ' -f1)"
if [ "$actual" != "$OUTPUT_SHA" ]; then
  # La sérialisation d'une autre version d'onnx peut différer sans changer les poids : on le dit, on continue.
  echo "avertissement : SHA-256 du FP16 refait $actual, attendu $OUTPUT_SHA (version d'onnx différente ?)" >&2
fi

# SHA256SUMS de la release tel qu'il est, sans la ligne que ce script remplace, puis la nôtre.
if gh release download "$TAG" -R "$REPO" -p SHA256SUMS -D "$work" 2>/dev/null; then
  grep -v -F -e "$OUTPUT" "$work/SHA256SUMS" >"$work/SHA256SUMS.new" || true
else
  echo "release $TAG sans SHA256SUMS : un nouveau sera écrit" >&2
  : >"$work/SHA256SUMS.new"
fi
(cd "$work" && sha256sum "$OUTPUT") | sed 's#  # #' >>"$work/SHA256SUMS.new"

echo "== SHA256SUMS fusionné (lu par fetch-models.sh)"; cat "$work/SHA256SUMS.new"
echo "== taille"; ls -l "$work/$OUTPUT" | awk '{print $5, $9}'

if [ "$PUBLISH" -ne 1 ]; then
  echo "Essai à blanc : rien n'est envoyé. Relancer avec --publier pour téléverser dans $REPO ($TAG)."
  exit 0
fi

# --latest=false : la release des modèles ne doit pas devenir la « dernière » release du dépôt.
gh release view "$TAG" -R "$REPO" >/dev/null 2>&1 \
  || gh release create "$TAG" -R "$REPO" --latest=false --title "SPAN pour vieux animes en SD (AniSD AC)" \
       --notes "AniSD AC (Sirosky, Upscale-Hub), SPAN x2 converti en FP16 ; licence CC BY-NC 4.0. SHA256SUMS les empreintes."
cp "$work/SHA256SUMS.new" "$work/SHA256SUMS"
# Le SHA256SUMS remplacé est celui de la release avec nos lignes ajoutées : les lignes des autres fichiers restent.
gh release upload "$TAG" -R "$REPO" --clobber "$work/$OUTPUT" "$work/SHA256SUMS"
echo "Publié : https://github.com/$REPO/releases/tag/$TAG"
