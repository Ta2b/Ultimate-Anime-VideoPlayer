#!/usr/bin/env bash
# Vérifie une archive téléchargée contre le manifeste signé de sa release (P7.1, signature gratuite des
# archives) : la signature Ed25519 de manifest.toml pour la clé publique du lecteur, puis le SHA-256 et la
# taille de l'archive tels que le manifeste les donne. Ne demande qu'openssl et sha256sum (ou shasum).
#
#   tools/release/verify-download.sh manifest.toml manifest.toml.sig ajp-linux-x64.tar.gz
#
# AJP_UPDATE_PUBKEY : la clé publique en hexadécimal (64 caractères) ; par défaut celle des releases
# (docs/signature-des-releases.md). Sortie 0 si tout concorde, 1 sinon.
set -euo pipefail

[ $# -eq 3 ] || { sed -n '2,10p' "$0"; exit 2; }
manifest=$1 sig=$2 archive=$3
pub=${AJP_UPDATE_PUBKEY:-c50eeb3e63c327a087487cd3f485e66fbbf8ef2340a0b90d7ec99d4c2d3f0d58}
[[ "$pub" =~ ^[0-9a-f]{64}$ ]] || { echo "clé publique invalide : $pub" >&2; exit 2; }

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
# SubjectPublicKeyInfo d'une clé Ed25519 : 12 octets d'en-tête fixes, puis les 32 octets de la clé.
# Hexadécimal vers octets sans xxd (absent d'une installation minimale) : \xHH lu par printf.
unhex() { printf '%b' "$(tr -d ' \n' | sed 's/../\\x&/g')"; }
printf '302a300506032b6570032100%s' "$pub" | unhex > "$work/pub.der"
unhex < "$sig" > "$work/sig.bin"
if ! openssl pkeyutl -verify -pubin -inkey "$work/pub.der" -keyform DER -rawin -in "$manifest" \
  -sigfile "$work/sig.bin" > /dev/null 2>&1; then
  echo "signature du manifeste invalide" >&2
  exit 1
fi

name=$(basename "$archive")
# La section du manifeste dont l'adresse finit par le nom de l'archive.
read -r want_sha want_size < <(awk -v name="$name" '
  /^\[/ { url = ""; sha = ""; size = "" }
  /^url = / { url = $3; gsub(/"/, "", url) }
  /^sha256 = / { sha = $3; gsub(/"/, "", sha) }
  /^size = / { size = $3 }
  url != "" && sha != "" && size != "" && url ~ ("/" name "$") { print sha, size; exit }
' "$manifest")
[ -n "${want_sha:-}" ] || { echo "$name absent du manifeste" >&2; exit 1; }

if command -v sha256sum > /dev/null; then got_sha=$(sha256sum "$archive" | cut -d' ' -f1)
else got_sha=$(shasum -a 256 "$archive" | cut -d' ' -f1); fi
got_size=$(wc -c < "$archive" | tr -d ' ')
if [ "$got_sha" != "$want_sha" ] || [ "$got_size" != "$want_size" ]; then
  echo "$name ne correspond pas au manifeste (SHA-256 ou taille)" >&2
  exit 1
fi
echo "$name : signature et empreinte vérifiées"
