#!/usr/bin/env bash
# La procédure de révocation de bout en bout, avec deux paires jetables (B51-03) : update-key.sh accepte la clé
# de secours, sign-manifest.sh --revoke écrit revoked_keys avant de signer, et la signature couvre la ligne.
# Lancé par la CI (rules.yml). Ne demande qu'openssl.
#
#   tools/release/test-signing.sh
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
fail() { echo "test-signing : $*" >&2; exit 1; }
pub_of() { openssl pkey -in "$1" -pubout -outform DER | tail -c 32 | od -An -tx1 -v | tr -d ' \n'; }

openssl genpkey -algorithm ed25519 -out "$work/active.pem" 2>/dev/null
openssl genpkey -algorithm ed25519 -out "$work/secours.pem" 2>/dev/null
active="$(pub_of "$work/active.pem")"
secours="$(pub_of "$work/secours.pem")"

# update-key.sh : la clé demandée passe, une autre non.
AJP_UPDATE_PUBKEY="$secours" UPDATE_KEY="$(cat "$work/secours.pem")" "$here/update-key.sh" "$work/key.pem" >/dev/null \
  || fail "update-key.sh refuse la clé de secours demandée"
if AJP_UPDATE_PUBKEY="$secours" UPDATE_KEY="$(cat "$work/active.pem")" "$here/update-key.sh" "$work/x.pem" >/dev/null 2>&1; then
  fail "update-key.sh accepte une clé qui n'est pas celle demandée"
fi

# Le manifeste de secours : signé par la secours, il révoque l'active.
mkdir -p "$work/rel"
printf 'archive' > "$work/rel/ajp-linux-x64.tar.gz"
AJP_UPDATE_KEY="$work/key.pem" "$here/sign-manifest.sh" --revoke "$(printf '%s' "$active" | tr 'a-f' 'A-F')" \
  9.9.9 stable "$work/rel" https://example.invalid/v9.9.9 >/dev/null
manifest="$work/rel/manifest.toml"
grep -qx "revoked_keys = \[\"$active\"\]" "$manifest" || fail "revoked_keys absent ou mal écrit : $(cat "$manifest")"
# Avant la première section, sinon le lecteur la lirait dans [linux-x64].
[ "$(grep -n '^revoked_keys' "$manifest" | cut -d: -f1)" -lt "$(grep -n '^\[' "$manifest" | head -1 | cut -d: -f1)" ] \
  || fail "revoked_keys après une section"
AJP_UPDATE_PUBKEY="$secours" "$here/verify-download.sh" "$manifest" "$manifest.sig" "$work/rel/ajp-linux-x64.tar.gz" \
  >/dev/null || fail "la signature de secours ne vérifie pas le manifeste"
if AJP_UPDATE_PUBKEY="$active" "$here/verify-download.sh" "$manifest" "$manifest.sig" \
  "$work/rel/ajp-linux-x64.tar.gz" >/dev/null 2>&1; then
  fail "le manifeste de secours se vérifie avec l'active"
fi
# La ligne ajoutée après coup invaliderait la signature : elle est bien couverte.
sed -i '/^revoked_keys/d' "$manifest"
if AJP_UPDATE_PUBKEY="$secours" "$here/verify-download.sh" "$manifest" "$manifest.sig" \
  "$work/rel/ajp-linux-x64.tar.gz" >/dev/null 2>&1; then
  fail "la signature ne couvre pas revoked_keys"
fi
# Une clé mal formée est refusée avant toute signature.
if AJP_UPDATE_KEY="$work/key.pem" "$here/sign-manifest.sh" --revoke abc 9.9.9 stable "$work/rel" x >/dev/null 2>&1; then
  fail "--revoke accepte une clé mal formée"
fi
echo "test-signing : révocation par la clé de secours vérifiée"

# C51-02 : la re-signature ne signe que ce que l'ancien manifeste nomme, avec les mêmes octets.
resign() {  # dossier
  AJP_UPDATE_PUBKEYS="$active" AJP_UPDATE_KEY="$work/active.pem" "$here/resign-manifest.sh" 9.9.9 "$1" https://example.invalid/v9.9.9
}
mkdir -p "$work/r1"
printf 'archive' > "$work/r1/ajp-linux-x64.tar.gz"
AJP_UPDATE_KEY="$work/active.pem" "$here/sign-manifest.sh" --revoke "$(printf '1%.0s' {1..64})" 9.9.9 stable "$work/r1" \
  https://example.invalid/v9.9.9 >/dev/null
cp -r "$work/r1" "$work/r2" && cp -r "$work/r1" "$work/r3"
printf 'macos' > "$work/r1/ajp-macos-arm64.zip"   # reste d'un essai, absent de l'ancien manifeste
resign "$work/r1" >/dev/null || fail "la re-signature d'archives conformes échoue"
grep -q 'macos' "$work/r1/manifest.toml" && fail "la re-signature a repris une archive que l'ancien manifeste ne nomme pas"
grep -q "revoked_keys = \[\"$(printf '1%.0s' {1..64})\"\]" "$work/r1/manifest.toml" || fail "revoked_keys perdues à la re-signature"
AJP_UPDATE_PUBKEY="$active" "$here/verify-download.sh" "$work/r1/manifest.toml" "$work/r1/manifest.toml.sig" \
  "$work/r1/ajp-linux-x64.tar.gz" >/dev/null || fail "le manifeste re-signé ne se vérifie pas"
printf 'remplacee' > "$work/r2/ajp-linux-x64.tar.gz"   # archive changée depuis la signature
if resign "$work/r2" >/dev/null 2>&1; then fail "la re-signature blanchit une archive remplacée"; fi
rm "$work/r3/ajp-linux-x64.tar.gz"
if resign "$work/r3" >/dev/null 2>&1; then fail "la re-signature accepte qu'une archive du manifeste manque"; fi
echo "test-signing : re-signature vérifiée"
