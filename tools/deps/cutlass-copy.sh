#!/usr/bin/env bash
# La copie partielle de CUTLASS (third_party/cutlass, PROVENANCE.md) : la vérifier, ou la refaire.
#
#   tools/deps/cutlass-copy.sh --check      chaque fichier a la somme de FILES.sha256, et rien de plus (sans réseau)
#   tools/deps/cutlass-copy.sh --upstream   chaque fichier est celui de l'amont au commit de PROVENANCE.md
#   tools/deps/cutlass-copy.sh <commit>     recopie les mêmes fichiers depuis ce commit et réécrit FILES.sha256
#                                           (PROVENANCE.md et THIRD_PARTY_NOTICES.md restent à mettre à jour à la main)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/third_party/cutlass"
URL=https://github.com/NVIDIA/cutlass
pinned="$(sed -n 's/.*`\([0-9a-f]\{40\}\)`.*/\1/p' "$DIR/PROVENANCE.md" | head -n 1)"

listed() { sed 's/^[0-9a-f]*  //' "$DIR/FILES.sha256"; }

check() {
  (cd "$DIR" && sha256sum --quiet -c FILES.sha256)
  local extra
  extra="$(comm -13 <(listed | LC_ALL=C sort) \
    <(cd "$DIR" && find . -type f ! -name PROVENANCE.md ! -name FILES.sha256 | sed 's|^\./||' | LC_ALL=C sort))"
  if [[ -n "$extra" ]]; then
    echo "third_party/cutlass : fichiers hors de FILES.sha256 :" >&2
    echo "$extra" >&2
    exit 1
  fi
  echo "third_party/cutlass : $(listed | wc -l) fichiers, sommes exactes"
}

clone_at() {  # commit dossier
  git init -q "$2"
  git -C "$2" fetch -q --depth 1 "$URL" "$1"
  git -C "$2" checkout -q FETCH_HEAD
}

case "${1:---check}" in
  --check) check ;;
  --upstream)
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    clone_at "$pinned" "$work/cutlass"
    while read -r file; do
      cmp -s "$DIR/$file" "$work/cutlass/$file" || { echo "$file diffère de l'amont ($pinned)" >&2; exit 1; }
    done < <(listed)
    echo "third_party/cutlass : identique à l'amont au commit $pinned"
    ;;
  *)
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    clone_at "$1" "$work/cutlass"
    while read -r file; do
      install -D -m 644 "$work/cutlass/$file" "$DIR/$file"
    done < <(listed)
    (cd "$DIR" && listed | xargs sha256sum > FILES.sha256.new && mv FILES.sha256.new FILES.sha256)
    echo "third_party/cutlass recopié depuis $1 : mettre à jour PROVENANCE.md et THIRD_PARTY_NOTICES.md"
    ;;
esac
