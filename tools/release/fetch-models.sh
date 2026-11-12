#!/usr/bin/env bash
# Les modèles livrés avec le lecteur, téléchargés et vérifiés par leur SHA-256 :
#
#   <dest>/onnx/   SPAN AnimeJaNai HD V3.1 (Sharp1 et normal, Performance et Balanced), depuis le
#                  dépôt mpv-AnimeJaNai-HDR à un commit fixé ; licence CC BY-NC-SA 4.0
#                  ArtCNN C4F32 et C4F16 enveloppés en RGB (P6.7), rangés dans tools/upscale/models ;
#                  licence MIT
#   <dest>/rife/   RIFE 4.26, 4.25 et 4.22 lite en fp16 (conversions de vs-mlrt), depuis la release
#                  models-rife-fp16-1 d'animejanai-inference ; RIFE 4.25 lite en fp16, converti par
#                  tools/rife/convert_rife_fp16.py depuis l'export de vs-mlrt (workflow
#                  export-interpolation-models.yml), depuis la release models-interp-1 d'ajp-releases : les
#                  quatre que le menu propose (kRifeCatalogue) ; RIFE est sous licence MIT
#                  DistilDRBA v2 lite en fp16, flot entier et demi-flot (_s05), exportés par le workflow
#                  export-interpolation-models.yml (tools/drba/export_distil_drba.py, depuis
#                  routineLife1/DistilDRBA db51bde, weights/v2_lite.pkl), depuis la release
#                  models-interp-1 d'ajp-releases ; DistilDRBA est sous licence MIT
#
#   tools/release/fetch-models.sh <dest> [dossier-de-cache]
#
# L'archive RIFE (322 Mo, 7z) reste dans le cache : un second appel ne la retélécharge pas.
set -euo pipefail

DEST="${1:?dossier de destination}"
CACHE="${2:-${XDG_CACHE_HOME:-$HOME/.cache}/ajp-release-models}"
# 7z (p7zip) ou 7zz (7-Zip, Homebrew sous macOS) ; sha256sum ou, sous macOS, shasum.
SEVENZ="$(command -v 7z || command -v 7zz || true)"
[ -n "$SEVENZ" ] || { echo "7z manquant (apt install p7zip-full, brew install sevenzip)" >&2; exit 1; }
if command -v sha256sum >/dev/null; then SHA256=(sha256sum); else SHA256=(shasum -a 256); fi
mkdir -p "$DEST/onnx" "$DEST/rife" "$CACHE"

SPAN_COMMIT=55a8ae53bb0e2dc531449b68a9a0717c61ae982b
SPAN_URL="https://raw.githubusercontent.com/Ta2b/mpv-AnimeJaNai-HDR/$SPAN_COMMIT/BuildMpvUpscale2xAnimeJaNai/mpv-upscale-2x_animejanai/animejanai/onnx"
SPAN=(
  "72246eded3aab52f8d1a71613c0a1d900734b358d1820b62eb36f1ed376e1cf4 2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
  "dde763c8202cfebac5ec030ea6644f3ee339ea35bbda8a76e257405399552f0f 2x_AnimeJaNai_HD_V3.1Sharp1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx"
  "7b804d87320f37b7269d6e27796046fbd975e95e2aa1f65d5557f52a36282716 2x_AnimeJaNai_HD_V3.1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
  "b308e93c3dda3c4f9f968295b2b7b8d1c0583a711bca53a08d51134d9d1cbd5c 2x_AnimeJaNai_HD_V3.1_Balanced_SPANF3_b8f64_unshuffle_fp16.onnx"
)
RIFE_URL="https://github.com/the-database/animejanai-inference/releases/download/models-rife-fp16-1/rife-fp16-1.7z"
RIFE_ARCHIVE_SHA=3fcf3429f3bb661d15c59aa58ed7ba28c0ca7f79e2388f8e0165dd8bbe0648f9
RIFE=(
  "fd6b06538898c9a94d20f0fe55febceadbcde5ad64952240733b7275ef4c9d59 rife_v4.26.onnx"
  "de0fb71ae50fe082eb3903fd5cf752063dea5e9fd61e8bbf3c4577e11b877c9c rife_v4.25.onnx"
  "8b30b93ef8ab0b3cd1bb8905ad8b911eb887ff4b2900558eb1914d99568e2ca5 rife_v4.22_lite.onnx"
)
RELEASE_REPO="${AJP_RELEASE_REPO:-$(sed -n 's/^AJP_RELEASE_REPO=//p' "$(dirname "$0")/versions.env")}"  # P5.39 : le dépôt des releases, paramétrable
# DistilDRBA : la grille de ses warps est en FP32 depuis l'export du 10 novembre 2026 (C05, A03-18). Les deux SHA
# sont épinglés ici : une republication de models-interp-1 ne change les paquets qu'avec une modification de ce
# fichier, donc avec une mesure au banc de qualité (dont le cache de modèles est indexé sur ce fichier).
INTERP_URL="${AJP_INTERP_MODELS_URL:-https://github.com/$RELEASE_REPO/releases/download/models-interp-1}"
# RIFE 4.25 lite : la conversion fp16 du workflow (onnx et numpy épinglés), épinglée de même.
INTERP=(
  "40bda3486348dd1ad33f728eeb669f5178e73051bd64ea10c767e0ab8cd17a50 distil_drba_v2_lite.onnx"
  "2a254537b573ff5777d7fe62336818c850313db8ab8b8e896ba496e06e5ea648 distil_drba_v2_lite_s05.onnx"
  "124c2e158f626f81d762256b2b130fe609ce28119221b3adcd471e6e0b75bf92 rife_v4.25_lite.onnx"
)

check() {  # sha256 fichier
  echo "$1  $2" | "${SHA256[@]}" -c --status - || { echo "SHA-256 inattendu : $2" >&2; exit 1; }
}

for entry in "${SPAN[@]}"; do
  read -r sha name <<<"$entry"
  curl -fsSL --retry 5 --retry-delay 5 -o "$DEST/onnx/$name" "$SPAN_URL/$name"
  check "$sha" "$DEST/onnx/$name"
done

archive="$CACHE/rife-fp16-1.7z"
if ! echo "$RIFE_ARCHIVE_SHA  $archive" | "${SHA256[@]}" -c --status - >/dev/null 2>&1; then  # pas encore en cache
  curl -fsSL --retry 5 --retry-delay 5 -o "$archive" "$RIFE_URL"
  check "$RIFE_ARCHIVE_SHA" "$archive"
fi
for entry in "${RIFE[@]}"; do
  read -r sha name <<<"$entry"
  "$SEVENZ" e -y -bso0 -o"$DEST/rife" "$archive" "$name"
  check "$sha" "$DEST/rife/$name"
done

# ArtCNN C4F32 et C4F16 (P6.7), enveloppés en RGB par tools/upscale/wrap_luma_rgb.py depuis ArtCNN_C4F32.onnx
# (SHA-256 b1a11304…) et ArtCNN_C4F16.onnx (7a643629…) d'Artoriuz/ArtCNN 0837e53 : 190 et 51 Ko, rangés dans le
# dépôt, à côté de leur licence MIT.
ARTCNN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../upscale/models" && pwd)"
ARTCNN=(
  "0f14a4cc990c5a00544fd8b939c35e2607d19ae872c4c4f31bc6e6b2dd643c12 2x_ArtCNN_C4F32_rgb_fp16.onnx"
  "0c9f39abf2d0a8cc899192771761cd3ad865aa3d0db1f95417e85c1f89c8b8fe 2x_ArtCNN_C4F16_rgb_fp16.onnx"
)
for entry in "${ARTCNN[@]}"; do
  read -r sha name <<<"$entry"
  cp "$ARTCNN_DIR/$name" "$DEST/onnx/"
  check "$sha" "$DEST/onnx/$name"
done
# La licence MIT demande son texte avec chaque copie : il voyage à côté des modèles.
cp "$ARTCNN_DIR/ArtCNN-LICENSE.txt" "$DEST/onnx/"

# Les autres fichiers de models-interp-1 ne sont pas pris : GMFSS est retiré du lecteur (0.10.0), et les graphes FP8
# de DistilDRBA, plus lents que le fusionné, restent facultatifs.
for entry in "${INTERP[@]}"; do
  read -r sha name <<<"$entry"
  curl -fsSL --retry 5 --retry-delay 5 -o "$DEST/rife/$name" "$INTERP_URL/$name"
  check "$sha" "$DEST/rife/$name"
done

cat > "$DEST/LICENCES.txt" <<'TXT'
Modèles livrés avec le lecteur

onnx/  AnimeJaNai HD V3.1 (SPAN), par the-database : Creative Commons BY-NC-SA 4.0
       (attribution, pas d'usage commercial, partage dans les mêmes conditions).
       https://github.com/the-database/mpv-upscale-2x_animejanai
       ArtCNN C4F32 et C4F16 (João Chrisóstomo), enveloppés en RGB par ajp : licence MIT
       (texte complet dans onnx/ArtCNN-LICENSE.txt).
       https://github.com/Artoriuz/ArtCNN
rife/  RIFE 4.26, 4.25, 4.22 lite et 4.25 lite (Practical-RIFE, hzwer), conversions ONNX de
       vs-mlrt passées en fp16 par animejanai-inference (4.25 lite : par ajp) : licence MIT.
       https://github.com/hzwer/Practical-RIFE
       DistilDRBA v2 lite (routineLife1, « neko »), converti en ONNX fp16 par ajp : licence MIT.
       https://github.com/routineLife1/DistilDRBA
TXT
echo "modèles : $DEST"
