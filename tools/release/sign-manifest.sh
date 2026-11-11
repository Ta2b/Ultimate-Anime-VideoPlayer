#!/usr/bin/env bash
# Écrit et signe le manifeste des mises à jour d'une release (src/core/update_manifest.h).
#
#   AJP_UPDATE_KEY=/chemin/ajp-update-key.pem tools/release/sign-manifest.sh [--revoke CLÉ-PUBLIQUE]... \
#       VERSION CANAL DOSSIER-DES-ARCHIVES URL-DE-BASE [NOTES-URL]
#
#   --revoke  clé publique (64 chiffres hexadécimaux) écrite dans revoked_keys AVANT la signature : la clé
#             volée ou perdue que les lecteurs ne doivent plus croire (docs/signature-des-releases.md).
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

revoke=()
while [ $# -gt 0 ] && [ "$1" = --revoke ]; do
  [ $# -ge 2 ] || { echo "--revoke : clé publique attendue" >&2; exit 2; }
  hex=$(printf '%s' "$2" | tr 'A-F' 'a-f')
  [[ "$hex" =~ ^[0-9a-f]{64}$ ]] || { echo "--revoke : $2 n'est pas une clé publique de 64 chiffres hexadécimaux" >&2; exit 2; }
  revoke+=("$hex")
  shift 2
done
[ $# -ge 4 ] || { sed -n '2,16p' "$0"; exit 2; }
version=$1 channel=$2 dir=$3 base=$4 notes=${5:-}
key=${AJP_UPDATE_KEY:?AJP_UPDATE_KEY : chemin de la clé privée Ed25519 (PEM)}

out="$dir/manifest.toml"
{
  echo "version = \"$version\""
  echo "channel = \"$channel\""
  echo "date = \"$(date -u +%F)\""
  # Au-delà, le lecteur refuse le manifeste (un ancien ne se rejoue pas) : re-signer avant (signature-des-releases.md).
  echo "expires = \"$(date -u -d '+120 days' +%F)\""
  [ -n "$notes" ] && echo "notes_url = \"$notes\""
  # Avant la première section : une clé TOML écrite après [linux-x64] serait lue dans la section.
  if [ ${#revoke[@]} -gt 0 ]; then
    printf 'revoked_keys = ['
    sep=""
    for hex in "${revoke[@]}"; do printf '%s"%s"' "$sep" "$hex"; sep=", "; done
    printf ']\n'
  fi
  for entry in "linux-x64:ajp-linux-x64.tar.gz" "windows-x64:ajp-windows-x64.zip" "macos-arm64:ajp-macos-arm64.zip"; do
    section=${entry%%:*} file=${entry#*:}
    [ -f "$dir/$file" ] || continue
    echo "[$section]"
    echo "url = \"$base/$file\""
    echo "sha256 = \"$(sha256sum "$dir/$file" | cut -d' ' -f1)\""
    echo "size = $(stat -c %s "$dir/$file")"
  done
} > "$out"

"$(dirname "$0")/sign-file.sh" "$out"
echo "manifeste : $out"
echo "signature : $out.sig"
