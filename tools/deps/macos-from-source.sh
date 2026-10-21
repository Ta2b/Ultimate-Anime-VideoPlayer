#!/usr/bin/env bash
# Dépendances du lecteur sur macOS (Apple Silicon), dans un préfixe local : les bibliothèques
# système de Homebrew (MoltenVK, chargeur Vulkan, shaderc, dav1d, freetype…), puis celles dont
# le lecteur fixe la version (FFmpeg, SDL3, libplacebo, libass du fork…), construites depuis
# leurs dépôts aux mêmes références que linux-from-source.sh.
#
#   tools/deps/macos-from-source.sh <prefixe> [dossier-de-travail]
#   AJP_DEPS_PREFIX=<prefixe> cmake --preset macos-release
set -euo pipefail

PREFIX="$(cd "$(dirname "${1:?usage: $0 <prefixe> [dossier-de-travail]}")" && pwd)/$(basename "$1")"
WORK="${2:-$PREFIX/../ajp-deps-src}"
JOBS="$(sysctl -n hw.ncpu)"
mkdir -p "$PREFIX" "$WORK"
WORK="$(cd "$WORK" && pwd)"

BREW_DEPS=(ninja meson pkgconf nasm autoconf automake libtool cmake
  molten-vk vulkan-loader shaderc glslang spirv-tools dav1d little-cms2
  freetype fribidi harfbuzz fontconfig)
brew install --quiet "${BREW_DEPS[@]}"
BREW="$(brew --prefix)"

export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig:$BREW/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PATH="$PREFIX/bin:$PATH"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"

# nom url référence : celles de linux-from-source.sh, sans nv-codec-headers (pas de NVDEC).
DEPS=(
  "vulkan-headers https://github.com/KhronosGroup/Vulkan-Headers v1.4.363"
  "tomlplusplus https://github.com/marzer/tomlplusplus v3.4.0"
  "catch2 https://github.com/catchorg/Catch2 v3.16.0"
  "spdlog https://github.com/gabime/spdlog v1.17.0"
  "nanobench https://github.com/martinus/nanobench v4.6.0"
  "imgui https://github.com/ocornut/imgui v1.92.9"
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
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" -DCMAKE_INSTALL_NAME_DIR="$PREFIX/lib" "$@"
  cmake --build "$WORK/$name/_build" -j "$JOBS"
  cmake --install "$WORK/$name/_build"
  mark "$name" "$signature"
}

for entry in "${DEPS[@]}"; do
  read -r name url ref <<<"$entry"
  echo "== $name $ref"
  fetch "$name" "$url" "$ref"
done

cmake_dep tomlplusplus -DTOMLPLUSPLUS_BUILD_TESTS=OFF -DTOMLPLUSPLUS_BUILD_EXAMPLES=OFF
# Catch2 en C++17 au moins, comme les tests : AppleClang compile par défaut en C++14 pour Catch2, et
# la bibliothèque n'a alors pas les StringMaker de std::string_view ni de std::byte qu'ils appellent.
cmake_dep catch2 -DCATCH_INSTALL_DOCS=OFF -DCATCH_INSTALL_EXTRAS=ON -DCMAKE_CXX_STANDARD=17
cmake_dep spdlog -DSPDLOG_USE_STD_FORMAT=ON -DSPDLOG_BUILD_SHARED=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake_dep vulkan-headers
cmake_dep sdl3 -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF

if ! done_marker nanobench "$(ref_of nanobench)"; then
  install -d "$PREFIX/include"
  install -m 644 "$WORK/nanobench/src/include/nanobench.h" "$PREFIX/include/nanobench.h"
  mark nanobench "$(ref_of nanobench)"
fi

if ! done_marker imgui "$(ref_of imgui)"; then
  mkdir -p "$PREFIX/share/imgui"
  cp -r "$WORK/imgui/"*.h "$WORK/imgui/"*.cpp "$WORK/imgui/backends" "$PREFIX/share/imgui/"
  mark imgui "$(ref_of imgui)"
fi

# FFmpeg en LGPL partagée ; VideoToolbox pour le décodage matériel, Vulkan pour libplacebo.
FFMPEG_OPTS=(--prefix="$PREFIX" --enable-shared --disable-static
  --disable-programs --disable-doc --disable-debug
  --enable-videotoolbox --enable-vulkan --enable-libdav1d
  --extra-cflags="-I$PREFIX/include -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
  --extra-ldflags="-L$PREFIX/lib -mmacosx-version-min=$MACOSX_DEPLOYMENT_TARGET"
  --install-name-dir="$PREFIX/lib")
signature="$(ref_of ffmpeg) ${FFMPEG_OPTS[*]}"
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

# ONNX Runtime (fournisseur CoreML) pour ajp_backend_coreml : la version publiée, vérifiée. Avant
# 1.31, le fournisseur refusait chaque nœud FP16 (tools/ci/coreml_report.py : SPAN 2 nœuds sur 56,
# RIFE 34 sur 247), qui tournaient alors sur le CPU ; en 1.31, 53 sur 56 et 247 sur 247.
ORT_VERSION=1.31.0
ORT_SHA256=1031b36dd3bdaa8d976a6130d9cf2a05e9be320b93ae804f8427b91dc3432afc
if ! done_marker onnxruntime "$ORT_VERSION $ORT_SHA256"; then
  ort="onnxruntime-osx-arm64-$ORT_VERSION"
  curl -fsSL -o "$WORK/$ort.tgz" \
    "https://github.com/microsoft/onnxruntime/releases/download/v$ORT_VERSION/$ort.tgz"
  echo "$ORT_SHA256  $WORK/$ort.tgz" | shasum -a 256 -c -
  rm -rf "$PREFIX/onnxruntime" "$WORK/$ort"
  tar -xzf "$WORK/$ort.tgz" -C "$WORK"
  mkdir -p "$PREFIX/onnxruntime"
  cp -R "$WORK/$ort/include" "$WORK/$ort/lib" "$WORK/$ort/LICENSE" "$PREFIX/onnxruntime/"
  mark onnxruntime "$ORT_VERSION $ORT_SHA256"
fi

echo "Dépendances prêtes dans $PREFIX"
