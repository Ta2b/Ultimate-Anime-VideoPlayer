#!/usr/bin/env bash
# Signe un fichier avec la clé Ed25519 des mises à jour : écrit <fichier>.sig, la signature en hexadécimal sur une
# ligne, que le lecteur vérifie (manifest.toml, engines-nvidia.lock).
#
#   AJP_UPDATE_KEY=<clé privée PEM> tools/release/sign-file.sh <fichier>
set -euo pipefail

[ $# -eq 1 ] || { sed -n '2,5p' "$0"; exit 2; }
file=$1
key=${AJP_UPDATE_KEY:?AJP_UPDATE_KEY : chemin de la clé privée Ed25519 (PEM)}
openssl pkeyutl -sign -inkey "$key" -rawin -in "$file" -out "$file.sig.bin"
# Hexadécimal sans xxd (absent de certaines images) : od, puis une seule ligne.
od -An -tx1 -v "$file.sig.bin" | tr -d ' \n' > "$file.sig"
echo >> "$file.sig"
rm -f "$file.sig.bin"
