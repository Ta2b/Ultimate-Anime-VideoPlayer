#!/usr/bin/env bash
# Les versions des dépendances se suivent à la main entre trois listes : le préfixe Linux
# (tools/deps/linux-from-source.sh), le préfixe macOS (tools/deps/macos-from-source.sh) et les ports vcpkg du
# dépôt (ports/ et vcpkg.json, Windows). Ce contrôle refuse qu'elles s'écartent pour ce que le dépôt fixe lui-même : une
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
  # macOS est délaissé (plus construit par la CI ni par les releases) : l'écart se signale sans bloquer (C28-16).
  echo "::warning::$name : $ref $commit sous Linux, $other sous macOS (macOS délaissé : non bloquant)" >&2
done < <(deps_of tools/deps/linux-from-source.sh)
for port in libplacebo libass; do
  commit="$(deps_of tools/deps/linux-from-source.sh | awk -v n="$port" '$1 == n {print $4}')"
  if ! grep -q "REF $commit" "ports/$port/portfile.cmake"; then
    echo "$port : le port vcpkg (ports/$port/portfile.cmake) n'est pas au commit $commit du préfixe Linux" >&2
    bad=1
  fi
done
# Windows : vcpkg.json fixe par « overrides » chaque dépendance que DEPS fixe, à la même version. Sans cela, un
# « builtin-baseline » rafraîchi changerait FFmpeg, SDL3 ou les en-têtes Vulkan de Windows sans que la CI Linux le
# voie (le HDR de Windows dépend de SDL, le décodage de FFmpeg).
if ! python3 - "$(deps_of tools/deps/linux-from-source.sh)" <<'PY'
import json, re, sys
deps = {}
for line in sys.argv[1].splitlines():
    name, _url, ref, _commit = line.split()
    deps[name] = re.sub(r"^(release-|v|n)", "", ref)
overrides = {o["name"]: o["version"] for o in json.load(open("vcpkg.json")).get("overrides", [])}
# Les dépendances de vcpkg.json que DEPS fixe aussi, avec le nom que vcpkg leur donne.
wanted = {"ffmpeg": "ffmpeg", "sdl3": "sdl3", "spdlog": "spdlog", "tomlplusplus": "tomlplusplus",
          "imgui": "imgui", "vulkan-headers": "vulkan-headers"}
bad = False
for vcpkg_name, deps_name in wanted.items():
    expected = deps[deps_name]
    got = overrides.get(vcpkg_name)
    # vcpkg écrit les en-têtes Vulkan en quatre champs (1.4.363.0) là où l'étiquette en a trois.
    if got is None or (got != expected and got != expected + ".0"):
        print(f"{vcpkg_name} : vcpkg.json overrides = {got}, tools/deps/linux-from-source.sh = {expected}", file=sys.stderr)
        bad = True
sys.exit(1 if bad else 0)
PY
then
  bad=1
fi
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
# cmake/Dependencies.cmake : les minimums de find_package et la version d'ImGui qu'il attend suivent DEPS (C27-03).
dep_version() { deps_of tools/deps/linux-from-source.sh | awk -v n="$1" '$1 == n {print $3}' | sed -E 's/^(release-|v|n)//'; }
for pair in "tomlplusplus:tomlplusplus" "spdlog:spdlog" "SDL3:sdl3"; do
  package="${pair%%:*}" dep="${pair##*:}"
  have="$(dep_version "$dep")"
  want="$(sed -n -E "s/^find_package\($package ([0-9.]+) .*/\1/p" cmake/Dependencies.cmake)"
  # Le minimum est un préfixe de la version fixée (3.4 pour 3.4.0, 3.4.20 pour 3.4.20).
  if [[ -z "$want" || "$have" != "$want"* ]]; then
    echo "$package : cmake/Dependencies.cmake exige '$want', DEPS fixe '$have'" >&2
    bad=1
  fi
done
imgui="$(sed -n 's/^set(AJP_IMGUI_VERSION_WANTED "\([0-9.]*\)").*/\1/p' cmake/Dependencies.cmake)"
[[ "$imgui" == "$(dep_version imgui)" ]] || {
  echo "imgui : cmake/Dependencies.cmake attend '$imgui', DEPS fixe '$(dep_version imgui)'" >&2
  bad=1
}
# meson : versions.env et les requirements à empreintes disent la même version (C28-14).
meson="$(sed -n 's/^MESON_VERSION=//p' tools/release/versions.env)"
grep -q "^meson==$meson " tools/ci/requirements/meson.txt || {
  echo "meson : tools/ci/requirements/meson.txt n'est pas à la version $meson de versions.env" >&2
  bad=1
}
((bad == 0)) && echo "dépendances : Linux, macOS et les ports du dépôt aux mêmes références"
exit "$bad"
