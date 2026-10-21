#!/usr/bin/env bash
# Construit les dépendances du lecteur depuis leurs dépôts git, aux versions
# fixées par vcpkg.json et ports/, dans un préfixe local.
#
# Sert quand vcpkg ne peut pas télécharger ses archives (réseau filtré, machine
# de dev sans cache binaire). La CI et les versions livrées passent par vcpkg.
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
mkdir -p "$PREFIX" "$WORK"
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib/x86_64-linux-gnu/pkgconfig:$PREFIX/share/pkgconfig:${PKG_CONFIG_PATH:-}"
export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export PATH="$PREFIX/bin:$PATH"

# nom url référence
DEPS=(
  "vulkan-headers https://github.com/KhronosGroup/Vulkan-Headers v1.4.363"
  "tomlplusplus https://github.com/marzer/tomlplusplus v3.4.0"
  "catch2 https://github.com/catchorg/Catch2 v3.16.0"
  "spdlog https://github.com/gabime/spdlog v1.17.0"
  "nanobench https://github.com/martinus/nanobench v4.6.0"
  "imgui https://github.com/ocornut/imgui v1.92.9"
  "nv-codec-headers https://github.com/FFmpeg/nv-codec-headers n13.0.19.0"
  "ffmpeg https://github.com/FFmpeg/FFmpeg n9.0.2"
  "sdl3 https://github.com/libsdl-org/SDL release-3.4.18"
  "libplacebo https://github.com/haasn/libplacebo v7.360.1"
  "libass https://github.com/the-database/libass 896614fa"
)

fetch() {  # nom url ref
  local dir="$WORK/$1"
  if [[ -d "$dir/.git" ]]; then
    # Déjà à cette référence : rien à faire. D'une autre (version changée) : recloné.
    [[ "$(git -C "$dir" rev-parse -q --verify "$3^{commit}" || true)" == "$(git -C "$dir" rev-parse HEAD)" ]] && return
    rm -rf "$dir"
  fi
  if git clone -q --depth 1 --branch "$3" "$2" "$dir" 2>/dev/null; then return; fi
  # Une référence qui n'est ni tag ni branche : un commit.
  git clone -q --filter=blob:none "$2" "$dir"
  git -C "$dir" checkout -q "$3"
}

# Une dépendance est à jour quand son marqueur porte sa signature : sa référence, ses options et
# la référence de ce qu'elle compile d'une autre. Changer une version ou une option rebâtit cette
# dépendance seule ; le cache de la CI, restauré même d'une version plus ancienne du script,
# garde les autres.
ref_of() {  # nom : sa référence dans DEPS (bash 3 de macOS : pas de tableau associatif)
  local entry name url ref
  for entry in "${DEPS[@]}"; do
    read -r name url ref <<<"$entry"
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
  read -r name url ref <<<"$entry"
  if (( CORE_ONLY )) && [[ "$CORE_DEPS" != *" $name "* ]]; then continue; fi
  echo "== $name $ref"
  fetch "$name" "$url" "$ref"
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
signature="$(ref_of libplacebo) ${PLACEBO_OPTS[*]} vulkan-headers $(ref_of vulkan-headers)"
if ! done_marker libplacebo "$signature"; then
  git -C "$WORK/libplacebo" submodule update --init --depth 1 -q
  rm -rf "$WORK/libplacebo/_build"
  meson setup "$WORK/libplacebo/_build" "$WORK/libplacebo" "${PLACEBO_OPTS[@]}"
  meson compile -C "$WORK/libplacebo/_build"
  meson install -C "$WORK/libplacebo/_build"
  mark libplacebo "$signature"
fi

if ! done_marker libass "$(ref_of libass)"; then
  (cd "$WORK/libass" && ./autogen.sh && ./configure --prefix="$PREFIX" --disable-static \
    && make -j "$JOBS" && make install)
  mark libass "$(ref_of libass)"
fi

echo "Dépendances prêtes dans $PREFIX"
