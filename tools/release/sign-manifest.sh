#!/usr/bin/env bash
# Écrit et signe le manifeste des mises à jour d'une release (src/core/update_manifest.h).
#
#   AJP_UPDATE_KEY=/chemin/ajp-update-key.pem tools/release/sign-manifest.sh \
#       VERSION CANAL DOSSIER-DES-ARCHIVES URL-DE-BASE [NOTES-URL]
#
#   VERSION   0.5.0 (ou 0.5.0-preview.1)       CANAL   stable | preview
#   DOSSIER   contient ajp-linux-x64.tar.gz, ajp-windows-x64.zip, ajp-macos-arm64.zip (ceux qui existent)
#   URL-DE-BASE   https://github.com/Ta2b/ajp-releases/releases/download/v0.5.0
# Produit DOSSIER/manifest.toml et DOSSIER/manifest.toml.sig (Ed25519, hexadécimal), à joindre à la release.
#
# La clé privée n'est jamais dans le dépôt. Pour la créer une fois et en tirer la clé publique à compiler
# (-DAJP_UPDATE_PUBKEY=…) :
#   openssl genpkey -algorithm ed25519 -out ajp-update-key.pem
#   openssl pkey -in ajp-update-key.pem -pubout -outform DER | tail -c 32 | xxd -p -c 64
set -euo pipefail

[ $# -ge 4 ] || { sed -n '2,13p' "$0"; exit 2; }
version=$1 channel=$2 dir=$3 base=$4 notes=${5:-}
key=${AJP_UPDATE_KEY:?AJP_UPDATE_KEY : chemin de la clé privée Ed25519 (PEM)}

out="$dir/manifest.toml"
{
  echo "version = \"$version\""
  echo "channel = \"$channel\""
  echo "date = \"$(date -u +%F)\""
  [ -n "$notes" ] && echo "notes_url = \"$notes\""
  for entry in "linux-x64:ajp-linux-x64.tar.gz" "windows-x64:ajp-windows-x64.zip" "macos-arm64:ajp-macos-arm64.zip"; do
    section=${entry%%:*} file=${entry#*:}
    [ -f "$dir/$file" ] || continue
    echo "[$section]"
    echo "url = \"$base/$file\""
    echo "sha256 = \"$(sha256sum "$dir/$file" | cut -d' ' -f1)\""
    echo "size = $(stat -c %s "$dir/$file")"
  done
} > "$out"

openssl pkeyutl -sign -inkey "$key" -rawin -in "$out" -out "$out.sig.bin"
xxd -p -c 256 "$out.sig.bin" > "$out.sig"
rm -f "$out.sig.bin"
echo "manifeste : $out"
echo "signature : $out.sig"
