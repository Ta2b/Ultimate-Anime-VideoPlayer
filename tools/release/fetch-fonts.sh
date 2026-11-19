#!/usr/bin/env bash
# La police CJK que le paquet Linux joint en dernier recours (src/app/play/subtitle_fonts.h) : Noto Sans CJK JP
# Regular, de notofonts/noto-cjk à l'étiquette Sans2.004, vérifiée par son SHA-256 ; licence SIL OFL 1.1 (texte
# dans THIRD_PARTY_NOTICES.md). Elle sert aux sous-titres et à l'interface quand le système n'a aucune police
# japonaise (installation minimale). Windows et macOS en ont toujours : leurs paquets ne la joignent pas.
#
#   tools/release/fetch-fonts.sh <dest> [dossier-de-cache]
#
# Le fichier (16 Mo) reste dans le cache : un second appel ne le retélécharge pas.
set -euo pipefail

DEST="${1:?dossier de destination}"
CACHE="${2:-${XDG_CACHE_HOME:-$HOME/.cache}/kirari-release-fonts}"
if command -v sha256sum >/dev/null; then SHA256=(sha256sum); else SHA256=(shasum -a 256); fi
mkdir -p "$DEST" "$CACHE"

NAME=NotoSansCJKjp-Regular.otf
URL="https://raw.githubusercontent.com/notofonts/noto-cjk/Sans2.004/Sans/OTF/Japanese/$NAME"
SHA=68a3fc98800b2a27b371f2fb79991daf3633bd89309d4ffaa6946fd587f375b5

ok() { echo "$SHA  $1" | "${SHA256[@]}" -c --status -; }
if ! [ -f "$CACHE/$NAME" ] || ! ok "$CACHE/$NAME"; then
  curl -fsSL --retry 4 -o "$CACHE/$NAME.part" "$URL"
  ok "$CACHE/$NAME.part" || { echo "SHA-256 inattendu : $NAME" >&2; rm -f "$CACHE/$NAME.part"; exit 1; }
  mv "$CACHE/$NAME.part" "$CACHE/$NAME"
fi
cp "$CACHE/$NAME" "$DEST/$NAME"
echo "police CJK de repli : $DEST/$NAME"
