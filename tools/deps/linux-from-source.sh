#!/usr/bin/env bash
# Construit les dépendances du lecteur depuis leurs dépôts git, aux commits fixés
# dans DEPS ci-dessous, dans un préfixe local.
#
# C'est la source des dépendances Linux de la CI et du paquet Linux, et celle des
# postes de développement. vcpkg (vcpkg.json, ports/) ne sert qu'à Windows : les
# versions des deux se suivent à la main.
#
#   tools/deps/linux-from-source.sh [--core] <prefixe> [dossier-de-travail]
#   AJP_DEPS_PREFIX=<prefixe> cmake --preset linux-prefix
#
# --core ne construit que ce dont core, kernels et leurs tests ont besoin.
set -euo pipefail

CORE_ONLY=0
if [[ "${1:-}" == "--core" ]]; then CORE_ONLY=1; shift; fi
PREFIX="$(realpath -m "${1:?usage: $0 <prefixe> [dossier-de-travail]}")"
WORK="$(realpath -m "${2:-$PREFIX/../ajp-deps-src}")"
JOBS="$(nproc)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # le dépôt : ports/ et ses correctifs
mkdir -p "$PREFIX" "$WORK"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib/x86_64-linux-gnu/pkgconfig:$PREFIX/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$PREFIX/bin:$PATH"

# nom url référence commit (le SHA complet que la référence désignait, vérifié après le clonage)
DEPS=(
  "vulkan-headers https://github.com/KhronosGroup/Vulkan-Headers v1.4.363 6802bb4733b63ed5efd3adb308a6c885ef180ea1"
  "tomlplusplus https://github.com/marzer/tomlplusplus v3.4.0 30172438cee64926dc41fdd9c11fb3ba5b2ba9de"
  "catch2 https://github.com/catchorg/Catch2 v3.16.0 317ac1ed4c0bb6e6b91eafc817e05c488feffcb3"
  "spdlog https://github.com/gabime/spdlog v1.17.0 79524ddd08a4ec981b7fea76afd08ee05f83755d"
  "nanobench https://github.com/martinus/nanobench v4.6.0 b709728dba4df0356d467abfb0ba20781a8031c0"
  "imgui https://github.com/ocornut/imgui v1.92.9 01380c579715e62fb9a8d6ec0502c4ea83bfde6e"
  "nv-codec-headers https://github.com/FFmpeg/nv-codec-headers n13.0.19.0 e844e5b26f46bb77479f063029595293aa8f812d"
  "ffmpeg https://github.com/FFmpeg/FFmpeg n9.0.2 946fcce07b6dcd0331c8cc609192aeff5e1924f8"
  "sdl3 https://github.com/libsdl-org/SDL release-3.4.18 829a65d769d935c4852f8159e964312c0957260a"
  "libplacebo https://github.com/haasn/libplacebo v7.360.1 cee9b076f2c63104ccfd497fa79c39a867293ec4"
  "libunibreak https://github.com/adah1972/libunibreak libunibreak_6_1 304585d8e2d63187507368d612c3d5fff1486368"
  "libass https://github.com/the-database/libass 896614fa 896614fa263f9f64944f7b8650ff6806d5328c4d"
)

fetch() {  # nom url ref commit
  local dir="$WORK/$1"
  # Déjà à ce commit : rien à faire. Sinon (version changée), recloné. Le commit complet, pas
  # l'étiquette, qu'un dépôt peut déplacer (P1.23) ; GitHub sert un commit par son SHA.
  [[ -d "$dir/.git" && "$(git -C "$dir" rev-parse HEAD)" == "$4" ]] && return
  rm -rf "${dir:?}"
  git init -q "$dir"
  git -C "$dir" fetch -q --depth 1 "$2" "$4"
  git -C "$dir" checkout -q FETCH_HEAD
  [[ "$(git -C "$dir" rev-parse HEAD)" == "$4" ]] || { echo "$1 : HEAD n'est pas $4" >&2; exit 1; }
}

# Une dépendance est à jour quand son marqueur porte sa signature : sa référence, ses options et
# la référence de ce qu'elle compile d'une autre. Changer une version ou une option rebâtit cette
# dépendance seule ; le cache de la CI, restauré même d'une version plus ancienne du script,
# garde les autres.
ref_of() {  # nom : sa référence dans DEPS (bash 3 de macOS : pas de tableau associatif)
  local entry name url ref commit
  for entry in "${DEPS[@]}"; do
    read -r name url ref commit <<<"$entry"
    if [[ "$name" == "$1" ]]; then echo "$ref"; return; fi
  done
}
done_marker() { [[ -f "$PREFIX/.built-$1" && "$(<"$PREFIX/.built-$1")" == "$2" ]]; }
mark() { printf '%s\n' "$2" >"$PREFIX/.built-$1"; }

cmake_dep() {  # nom [options...]
  local name="$1"; shift
  local signature
  signature="$(ref_of "$name") $*"
  done_marker "$name" "$signature" && return
  cmake -S "$WORK/$name" -B "$WORK/$name/_build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" "$@"
  cmake --build "$WORK/$name/_build" -j "$JOBS"
  cmake --install "$WORK/$name/_build"
  mark "$name" "$signature"
}

CORE_DEPS=" tomlplusplus catch2 spdlog "
for entry in "${DEPS[@]}"; do
  read -r name url ref commit <<<"$entry"
  if (( CORE_ONLY )) && [[ "$CORE_DEPS" != *" $name "* ]]; then continue; fi
  echo "== $name $ref"
  fetch "$name" "$url" "$ref" "$commit"
done

cmake_dep tomlplusplus -DTOMLPLUSPLUS_BUILD_TESTS=OFF -DTOMLPLUSPLUS_BUILD_EXAMPLES=OFF
cmake_dep catch2 -DCATCH_INSTALL_DOCS=OFF -DCATCH_INSTALL_EXTRAS=ON
cmake_dep spdlog -DSPDLOG_USE_STD_FORMAT=ON -DSPDLOG_BUILD_SHARED=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
if (( CORE_ONLY )); then
  echo "Dépendances du cœur prêtes dans $PREFIX"
  exit 0
fi

cmake_dep vulkan-headers
cmake_dep sdl3 -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF

if ! done_marker nanobench "$(ref_of nanobench)"; then
  install -D -m 644 "$WORK/nanobench/src/include/nanobench.h" "$PREFIX/include/nanobench.h"
  mark nanobench "$(ref_of nanobench)"
fi

# imgui n'a pas de système de construction : on installe ses sources, que
# cmake/Dependencies.cmake compile avec les backends SDL3 et Vulkan.
if ! done_marker imgui "$(ref_of imgui)"; then
  mkdir -p "$PREFIX/share/imgui"
  cp -r "$WORK/imgui/"*.h "$WORK/imgui/"*.cpp "$WORK/imgui/backends" "$PREFIX/share/imgui/"
  # Le chargeur FreeType : stb_truetype ne lit pas les polices OpenType à contours CFF2 (Noto Sans CJK variable de Fedora).
  mkdir -p "$PREFIX/share/imgui/misc/freetype"
  cp "$WORK/imgui/misc/freetype/"imgui_freetype.* "$PREFIX/share/imgui/misc/freetype/"
  mark imgui "$(ref_of imgui)"
fi

if ! done_marker nv-codec-headers "$(ref_of nv-codec-headers)"; then
  make -C "$WORK/nv-codec-headers" PREFIX="$PREFIX" install
  mark nv-codec-headers "$(ref_of nv-codec-headers)"
fi

# FFmpeg en LGPL partagée, comme le triplet dynamique de vcpkg. VA-API demande les en-têtes de
# libva (libva-dev, libva-devel) ; la libva du système sert à l'exécution. Sans bzip2 : il ne
# sert qu'aux pistes MKV compressées en bzip2, et sa bibliothèque ne porte pas le même nom
# partout (libbz2.so.1.0 sous Debian et Ubuntu, libbz2.so.1 sous Fedora : package-linux.sh).
FFMPEG_OPTS=(--prefix="$PREFIX" --enable-shared --disable-static
  --disable-programs --disable-doc --disable-debug --disable-bzlib
  --enable-ffnvcodec --enable-nvdec --enable-cuvid
  --enable-vulkan --enable-vaapi --enable-libdav1d
  --extra-cflags="-I$PREFIX/include" --extra-ldflags="-L$PREFIX/lib")
signature="$(ref_of ffmpeg) ${FFMPEG_OPTS[*]} nv-codec-headers $(ref_of nv-codec-headers)"
if ! done_marker ffmpeg "$signature"; then
  (cd "$WORK/ffmpeg" && ./configure "${FFMPEG_OPTS[@]}" && make -j "$JOBS" && make install)
  mark ffmpeg "$signature"
fi

PLACEBO_OPTS=(--prefix="$PREFIX" --libdir=lib
  --buildtype=release -Dvulkan=enabled -Dshaderc=enabled -Dlcms=enabled
  -Dopengl=disabled -Dd3d11=disabled -Ddemos=false -Dtests=false
  -Dvulkan-registry="$PREFIX/share/vulkan/registry/vk.xml")
# Le registre Vulkan vient de vulkan-headers : une autre version rebâtit libplacebo.
# Le correctif de présentation (chantier 6.1), le même que le port vcpkg : sa somme entre dans la signature.
placebo_patch="$ROOT/ports/libplacebo/0001-ajp-present-timing.patch"
signature="$(ref_of libplacebo) ${PLACEBO_OPTS[*]} vulkan-headers $(ref_of vulkan-headers) $(sha256sum "$placebo_patch" | cut -c1-16)"
if ! done_marker libplacebo "$signature"; then
  git -C "$WORK/libplacebo" submodule update --init --depth 1 -q
  # Appliqué une fois : déjà là (une construction interrompue), rien à refaire.
  if ! git -C "$WORK/libplacebo" apply --reverse --check "$placebo_patch" 2>/dev/null; then
    git -C "$WORK/libplacebo" apply "$placebo_patch"
  fi
  rm -rf "$WORK/libplacebo/_build"
  meson setup "$WORK/libplacebo/_build" "$WORK/libplacebo" "${PLACEBO_OPTS[@]}"
  meson compile -C "$WORK/libplacebo/_build"
  meson install -C "$WORK/libplacebo/_build"
  mark libplacebo "$signature"
fi

# La coupure des lignes selon Unicode (UAX #14), que libass prend à libunibreak.
if ! done_marker libunibreak "$(ref_of libunibreak)"; then
  (cd "$WORK/libunibreak" && ./autogen.sh && ./configure --prefix="$PREFIX" --disable-static \
    && make -j "$JOBS" && make install)
  mark libunibreak "$(ref_of libunibreak)"
fi

# Assembleur et libunibreak exigés (P1.24) : sans eux, configure passe outre avec un simple
# avertissement, et le rendu des sous-titres perd ses routines SIMD ou la coupure Unicode.
signature="$(ref_of libass) asm libunibreak $(ref_of libunibreak)"
if ! done_marker libass "$signature"; then
  (cd "$WORK/libass" && ./autogen.sh && ./configure --prefix="$PREFIX" --disable-static \
    --enable-asm --enable-libunibreak && make -j "$JOBS" && make install)
  mark libass "$signature"
fi

echo "Dépendances prêtes dans $PREFIX"
