#!/usr/bin/env bash
# Les versions des dépendances se suivent à la main entre trois listes : le préfixe Linux
# (tools/deps/linux-from-source.sh), le préfixe macOS (tools/deps/macos-from-source.sh) et les ports vcpkg du
# dépôt (ports/, Windows). Ce contrôle refuse qu'elles s'écartent pour ce que le dépôt fixe lui-même : une
# dépendance commune aux deux scripts y a la même référence et le même commit, et libplacebo et libass ont
# le commit de leur port. Ce que les trois plateformes prennent ailleurs (apt, Homebrew, baseline de vcpkg :
# dav1d, lcms2, shaderc, FriBidi, FreeType, HarfBuzz) n'est pas vérifié ici : cmake/Dependencies.cmake le dit.
set -euo pipefail
cd "$(dirname "$0")/../.."
bad=0
deps_of() { sed -n 's/^  "\([a-z0-9+-]* https[^"]*\)"$/\1/p' "$1"; }
while read -r name url ref commit; do
  other="$(deps_of tools/deps/macos-from-source.sh | awk -v n="$name" '$1 == n {print $3, $4}')"
  [[ -z "$other" || "$other" == "$ref $commit" ]] && continue
  echo "$name : $ref $commit sous Linux, $other sous macOS" >&2
  bad=1
done < <(deps_of tools/deps/linux-from-source.sh)
for port in libplacebo libass; do
  commit="$(deps_of tools/deps/linux-from-source.sh | awk -v n="$port" '$1 == n {print $4}')"
  if ! grep -q "REF $commit" "ports/$port/portfile.cmake"; then
    echo "$port : le port vcpkg (ports/$port/portfile.cmake) n'est pas au commit $commit du préfixe Linux" >&2
    bad=1
  fi
done
# libplacebo : la version du port et le minimum exigé à la configuration suivent l'étiquette.
tag="$(deps_of tools/deps/linux-from-source.sh | awk '$1 == "libplacebo" {print $3}')"
version="${tag#v}"
grep -q "\"version\": \"$version\"" ports/libplacebo/vcpkg.json || {
  echo "libplacebo : ports/libplacebo/vcpkg.json n'est pas à la version $version" >&2
  bad=1
}
grep -q "libplacebo>=${version%.*}" cmake/Dependencies.cmake || {
  echo "libplacebo : cmake/Dependencies.cmake n'exige pas libplacebo>=${version%.*}" >&2
  bad=1
}
((bad == 0)) && echo "dépendances : Linux, macOS et les ports du dépôt aux mêmes références"
exit "$bad"
