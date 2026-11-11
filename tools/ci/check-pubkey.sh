#!/usr/bin/env bash
# La clé publique des mises à jour a trois copies (B51-17) : CMakeLists.txt (celle compilée dans le lecteur),
# verify-download.sh (un script que des utilisateurs copient, qui doit rester autonome) et signature-des-releases.md.
# Elles disent la même chose, ou la CI échoue.
set -euo pipefail
cd "$(dirname "$0")/../.."
cmake="$(sed -n 's/^set(AJP_UPDATE_PUBKEY "\([0-9a-f]*\)".*/\1/p' src/app/CMakeLists.txt)"
[[ "$cmake" =~ ^[0-9a-f]{64}$ ]] || { echo "clé publique illisible dans src/app/CMakeLists.txt" >&2; exit 1; }
verify="$(sed -n 's/^pub=\${AJP_UPDATE_PUBKEY:-\([0-9a-f]*\)}.*/\1/p' tools/release/verify-download.sh)"
[ "$verify" = "$cmake" ] || { echo "verify-download.sh : clé « $verify » au lieu de celle du lecteur ($cmake)" >&2; exit 1; }
grep -q "$cmake" docs/signature-des-releases.md || { echo "docs/signature-des-releases.md ne cite pas la clé du lecteur ($cmake)" >&2; exit 1; }
echo "clé publique des mises à jour : trois copies d'accord"
