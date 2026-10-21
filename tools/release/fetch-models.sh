#!/usr/bin/env bash
# Les modèles livrés avec le lecteur, téléchargés et vérifiés par leur SHA-256 :
#
#   <dest>/onnx/   SPAN AnimeJaNai HD V3.1 (Sharp1 et normal, Performance et Balanced), depuis le
#                  dépôt mpv-AnimeJaNai-HDR à un commit fixé ; licence CC BY-NC-SA 4.0
#   <dest>/rife/   RIFE 4.25 et 4.22 lite en fp16 (conversions de vs-mlrt), depuis la release
#                  models-rife-fp16-1 d'animejanai-inference ; RIFE est sous licence MIT
#                  DistilDRBA v2 lite en fp16, converti par tools/drba/export_distil_drba.py depuis
#                  routineLife1/DistilDRBA (db51bde, weights/v2_lite.pkl), depuis la release
#                  models-drba-1 d'ajp-releases, sinon de la branche assets/distil-drba-v2-lite du
#                  dépôt ; DistilDRBA est sous licence MIT
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
  "de0fb71ae50fe082eb3903fd5cf752063dea5e9fd61e8bbf3c4577e11b877c9c rife_v4.25.onnx"
  "8b30b93ef8ab0b3cd1bb8905ad8b911eb887ff4b2900558eb1914d99568e2ca5 rife_v4.22_lite.onnx"
)
DRBA_URL="https://github.com/Ta2b/ajp-releases/releases/download/models-drba-1"
# Le même fichier, tant que la release models-drba-1 n'existe pas (le SHA-256 vérifie les deux).
DRBA_MIRROR="https://raw.githubusercontent.com/Ta2b/Ultimate-Anime-VideoPlayer/305b4cd5de2d516d2e55bb5a12b299828374ca39"
DRBA=(
  "7048367e94f6c43d9296f71adec95ddba7ef4c4b8efdbd8d91b8021891dbc39a distil_drba_v2_lite.onnx"
)

check() {  # sha256 fichier
  echo "$1  $2" | "${SHA256[@]}" -c --status - || { echo "SHA-256 inattendu : $2" >&2; exit 1; }
}

for entry in "${SPAN[@]}"; do
  read -r sha name <<<"$entry"
  curl -fsSL -o "$DEST/onnx/$name" "$SPAN_URL/$name"
  check "$sha" "$DEST/onnx/$name"
done

archive="$CACHE/rife-fp16-1.7z"
if ! echo "$RIFE_ARCHIVE_SHA  $archive" | "${SHA256[@]}" -c --status - >/dev/null 2>&1; then  # pas encore en cache
  curl -fsSL -o "$archive" "$RIFE_URL"
  check "$RIFE_ARCHIVE_SHA" "$archive"
fi
for entry in "${RIFE[@]}"; do
  read -r sha name <<<"$entry"
  "$SEVENZ" e -y -bso0 -o"$DEST/rife" "$archive" "$name"
  check "$sha" "$DEST/rife/$name"
done

for entry in "${DRBA[@]}"; do
  read -r sha name <<<"$entry"
  curl -fsSL -o "$DEST/rife/$name" "$DRBA_URL/$name" || curl -fsSL -o "$DEST/rife/$name" "$DRBA_MIRROR/$name"
  check "$sha" "$DEST/rife/$name"
done

cat > "$DEST/LICENCES.txt" <<'TXT'
Modèles livrés avec le lecteur

onnx/  AnimeJaNai HD V3.1 (SPAN), par the-database : Creative Commons BY-NC-SA 4.0
       (attribution, pas d'usage commercial, partage dans les mêmes conditions).
       https://github.com/the-database/mpv-upscale-2x_animejanai
rife/  RIFE 4.25 et 4.22 lite (Practical-RIFE, hzwer), conversions ONNX de vs-mlrt passées en
       fp16 par animejanai-inference : licence MIT.
       https://github.com/hzwer/Practical-RIFE
       DistilDRBA v2 lite (routineLife1, « neko »), converti en ONNX fp16 par ajp : licence MIT.
       https://github.com/routineLife1/DistilDRBA
TXT
echo "modèles : $DEST"
