#!/usr/bin/env bash
# La copie partielle d'Anime4K (third_party/anime4k, PROVENANCE.md) : la vérifier, ou la refaire.
#
#   tools/deps/anime4k-copy.sh --check      chaque fichier a la somme de FILES.sha256, et rien de plus (sans réseau)
#   tools/deps/anime4k-copy.sh --upstream   chaque fichier est celui de l'amont au commit de PROVENANCE.md
#   tools/deps/anime4k-copy.sh <commit>     recopie les mêmes fichiers depuis ce commit et réécrit FILES.sha256
#                                           (PROVENANCE.md et THIRD_PARTY_NOTICES.md restent à mettre à jour à la main)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIR="$ROOT/third_party/anime4k"
URL=https://github.com/bloc97/Anime4K
pinned="$(sed -n 's/.*`\([0-9a-f]\{40\}\)`.*/\1/p' "$DIR/PROVENANCE.md" | head -n 1)"

listed() { sed 's/^[0-9a-f]*  //' "$DIR/FILES.sha256"; }

check() {
  (cd "$DIR" && sha256sum --quiet -c FILES.sha256)
  local extra
  extra="$(comm -13 <(listed | LC_ALL=C sort) \
    <(cd "$DIR" && find . -type f ! -name PROVENANCE.md ! -name FILES.sha256 | sed 's|^\./||' | LC_ALL=C sort))"
  if [[ -n "$extra" ]]; then
    echo "third_party/anime4k : fichiers hors de FILES.sha256 :" >&2
    echo "$extra" >&2
    exit 1
  fi
  echo "third_party/anime4k : $(listed | wc -l) fichiers, sommes exactes"
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
    clone_at "$pinned" "$work/anime4k"
    while read -r file; do
      cmp -s "$DIR/$file" "$work/anime4k/$file" || { echo "$file diffère de l'amont ($pinned)" >&2; exit 1; }
    done < <(listed)
    echo "third_party/anime4k : identique à l'amont au commit $pinned"
    ;;
  *)
    work="$(mktemp -d)"
    trap 'rm -rf "$work"' EXIT
    clone_at "$1" "$work/anime4k"
    while read -r file; do
      install -D -m 644 "$work/anime4k/$file" "$DIR/$file"
    done < <(listed)
    (cd "$DIR" && listed | xargs sha256sum > FILES.sha256.new && mv FILES.sha256.new FILES.sha256)
    echo "third_party/anime4k recopié depuis $1 : mettre à jour PROVENANCE.md et THIRD_PARTY_NOTICES.md"
    ;;
esac
