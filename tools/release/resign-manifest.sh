#!/usr/bin/env bash
# Re-signe le manifeste de la « latest » sans rien blanchir (C51-02). Le dossier contient le manifeste actuel
# (manifest.toml et manifest.toml.sig) et les archives téléchargées de la release. Avant de signer :
#   - chaque archive que le manifeste actuel nomme est vérifiée contre LUI (signature de l'ancien manifeste
#     comprise, avec la clé active ou celle de secours du lecteur) : une archive remplacée depuis sa signature fait
#     échouer la re-signature au lieu d'être signée à son tour ;
#   - une archive que l'ancien manifeste ne nomme pas (macOS, reste d'un essai) n'entre pas dans le nouveau ;
#   - ses revoked_keys sont reprises.
#
#   AJP_UPDATE_KEY=/chemin/clé.pem tools/release/resign-manifest.sh VERSION DOSSIER URL-DE-BASE [NOTES-URL]
set -euo pipefail

[ $# -ge 3 ] || { sed -n '2,12p' "$0"; exit 2; }
version=$1 dir=$2 base=$3 notes=${4:-}
here="$(cd "$(dirname "$0")" && pwd)"
cmake="$here/../../src/app/CMakeLists.txt"
old="$dir/manifest.old.toml" old_sig="$dir/manifest.old.toml.sig"
mv "$dir/manifest.toml" "$old"
mv "$dir/manifest.toml.sig" "$old_sig"

# Les clés que le lecteur croit : l'active, puis celle de secours si elle est compilée.
if [ -n "${AJP_UPDATE_PUBKEYS:-}" ]; then read -r -a keys <<<"$AJP_UPDATE_PUBKEYS"
else
  keys=()
  for var in AJP_UPDATE_PUBKEY AJP_UPDATE_PUBKEY_BACKUP; do
    k="$(sed -n "s/^set($var \"\\([0-9a-f]*\\)\".*/\\1/p" "$cmake")"
    [ -z "$k" ] || keys+=("$k")
  done
fi
[ ${#keys[@]} -gt 0 ] || { echo "resign-manifest : aucune clé publique du lecteur" >&2; exit 1; }

# Les archives nommées par l'ancien manifeste (adresse de chaque section).
named="$(sed -n 's/^url = "\(.*\)"$/\1/p' "$old" | sed 's|.*/||')"
[ -n "$named" ] || { echo "resign-manifest : l'ancien manifeste ne nomme aucune archive" >&2; exit 1; }

verified=0
for archive in "$dir"/ajp-*; do
  [ -f "$archive" ] || continue
  name="$(basename "$archive")"
  if ! grep -qxF "$name" <<<"$named"; then
    echo "$name : absente de l'ancien manifeste, laissée de côté"
    rm -f "$archive"
    continue
  fi
  ok=0
  for k in "${keys[@]}"; do
    if AJP_UPDATE_PUBKEY="$k" "$here/verify-download.sh" "$old" "$old_sig" "$archive" >/dev/null 2>&1; then ok=1; break; fi
  done
  [ "$ok" = 1 ] || { echo "::error::$name ne correspond pas au manifeste actuel (signature, SHA-256 ou taille) : archive remplacée ? Pas de re-signature." >&2; exit 1; }
  echo "$name : conforme au manifeste actuel"
  verified=$((verified + 1))
done
# Chaque archive du manifeste actuel doit être là : un manifeste qui en perdrait une l'enlèverait du parc.
while IFS= read -r name; do
  [ -f "$dir/$name" ] || { echo "::error::$name est nommée par le manifeste actuel mais absente de la release" >&2; exit 1; }
done <<<"$named"
[ "$verified" -gt 0 ] || { echo "resign-manifest : aucune archive vérifiée" >&2; exit 1; }

revoke=()
while read -r k; do [ -z "$k" ] || revoke+=(--revoke "$k"); done < <(sed -n 's/^revoked_keys *= *\[\(.*\)\].*/\1/p' "$old" | grep -oE '[0-9a-fA-F]{64}')

"$here/sign-manifest.sh" "${revoke[@]}" "$version" stable "$dir" "$base" ${notes:+"$notes"}
