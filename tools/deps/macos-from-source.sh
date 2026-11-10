#!/usr/bin/env bash
# Dépendances du lecteur sur macOS (Apple Silicon), dans un préfixe local : les bibliothèques
# système de Homebrew (MoltenVK, chargeur Vulkan, shaderc, dav1d, freetype…), puis celles dont
# le lecteur fixe la version (FFmpeg, SDL3, libplacebo, libass…), construites depuis
# leurs dépôts aux mêmes références que linux-from-source.sh.
#
#   tools/deps/macos-from-source.sh [--locked] <prefixe> [dossier-de-travail]
#   AJP_DEPS_PREFIX=<prefixe> cmake --preset macos-release
set -euo pipefail

LOCKED=0
if [[ "${1:-}" == "--locked" ]]; then LOCKED=1; shift; fi
PREFIX="$(cd "$(dirname "${1:?usage: $0 [--locked] <prefixe> [dossier-de-travail]}")" && pwd)/$(basename "$1")"
WORK="${2:-$PREFIX/../ajp-deps-src}"
JOBS="$(sysctl -n hw.ncpu)"
mkdir -p "$PREFIX" "$WORK"
WORK="$(cd "$WORK" && pwd)"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # le dépôt : ports/ et ses correctifs

# Outils de construction, puis les bibliothèques que le paquet recopie (dylibbundler). Homebrew ne
# sert qu'une version de chaque formule : celles du paquet sont relevées dans BREW_LOCK et
# comparées après l'installation (P1.25). --locked (release-macos.yml) refuse un écart ; sans lui,
# un écart s'affiche seulement, la CI reste verte jusqu'à la mise à jour du relevé. Homebrew vérifie
# lui-même l'empreinte de chaque bouteille. MoltenVK et le chargeur Vulkan ne viennent plus de
# Homebrew : version et empreinte fixées plus bas.
BREW_TOOLS=(ninja meson pkgconf nasm autoconf automake libtool cmake)
BREW_LOCK=(
  "shaderc 2026.4" "glslang 16.6.0" "spirv-tools 1.4.363.0" "dav1d 1.5.4" "little-cms2 2.19.1"
  "freetype 2.15.0" "fribidi 1.0.18" "harfbuzz 15.1.0" "fontconfig 2.18.3"
)
brew_libs=()
for entry in "${BREW_LOCK[@]}"; do brew_libs+=("${entry%% *}"); done
# L'index de l'image du runner peut retarder sur le relevé (0.10.1 : harfbuzz 15.0.0 contre 15.1.0) :
# le mettre à jour d'abord.
brew update --quiet
brew install --quiet "${BREW_TOOLS[@]}" "${brew_libs[@]}"
# Une formule déjà dans l'image du runner n'est pas mise à jour par brew install (0.10.0 : harfbuzz 15.0.0 de
# l'image contre 15.1.0 relevé) : la même que celle du dépôt Homebrew, celle du relevé.
brew upgrade --quiet "${brew_libs[@]}"
BREW="$(brew --prefix)"
drift=0
for entry in "${BREW_LOCK[@]}"; do
  read -r formula want <<<"$entry"
  # brew upgrade garde l'ancienne version à côté (« harfbuzz 15.0.0_1 15.1.0 ») : la plus récente compte.
  got="$(brew list --versions "$formula" | tr ' ' '\n' | tail -n +2 | sort -V | tail -1)"
  got="${got%_*}"  # « 2.15.0_1 » : la révision de la formule, même source
  if [[ "$got" != "$want" ]]; then
    echo "Homebrew : $formula $got, relevé $want (BREW_LOCK de $0)" >&2
    drift=1
  fi
done
if (( drift && LOCKED )); then exit 1; fi

export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/share/pkgconfig:$BREW/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
export PATH="$PREFIX/bin:$PATH"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"

# nom url référence commit : ceux de linux-from-source.sh, sans nv-codec-headers (pas de NVDEC).
DEPS=(
  "vulkan-headers https://github.com/KhronosGroup/Vulkan-Headers v1.4.363 6802bb4733b63ed5efd3adb308a6c885ef180ea1"
  "vulkan-loader https://github.com/KhronosGroup/Vulkan-Loader v1.4.363 9e9518d2cf00054a220ae1a40f4406804b420ef5"
  "tomlplusplus https://github.com/marzer/tomlplusplus v3.4.0 30172438cee64926dc41fdd9c11fb3ba5b2ba9de"
  "catch2 https://github.com/catchorg/Catch2 v3.16.0 317ac1ed4c0bb6e6b91eafc817e05c488feffcb3"
  "spdlog https://github.com/gabime/spdlog v1.17.0 79524ddd08a4ec981b7fea76afd08ee05f83755d"
  "nanobench https://github.com/martinus/nanobench v4.6.0 b709728dba4df0356d467abfb0ba20781a8031c0"
  "imgui https://github.com/ocornut/imgui v1.93.0 c1aaac65e896a45f81f47a4d68f9959e0f86c859"
  "ffmpeg https://github.com/FFmpeg/FFmpeg n9.0.2 946fcce07b6dcd0331c8cc609192aeff5e1924f8"
  "sdl3 https://github.com/libsdl-org/SDL release-3.4.20 f00dc919ca35b9756316e86c4777392de442f9b2"
  "libplacebo https://github.com/haasn/libplacebo v7.387.0 275f7dea27924cdd34ab2e6c20e448b6bc4fab65"
  "libunibreak https://github.com/adah1972/libunibreak libunibreak_7_0 3ce4bfa3129ff3738046a44a6db533d2ce25af2b"
  "libass https://github.com/libass/libass 0.17.5 4a05d8127f525943ebf45fdc6497c9e665947f0d"
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
    -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_PREFIX_PATH="$PREFIX" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOSX_DEPLOYMENT_TARGET" -DCMAKE_INSTALL_NAME_DIR="$PREFIX/lib" "$@"
  cmake --build "$WORK/$name/_build" -j "$JOBS"
  cmake --install "$WORK/$name/_build"
  mark "$name" "$signature"
}

for entry in "${DEPS[@]}"; do
  read -r name url ref commit <<<"$entry"
  echo "== $name $ref"
  fetch "$name" "$url" "$ref" "$commit"
done

cmake_dep tomlplusplus -DTOMLPLUSPLUS_BUILD_TESTS=OFF -DTOMLPLUSPLUS_BUILD_EXAMPLES=OFF
# Catch2 en C++17 au moins, comme les tests : AppleClang compile par défaut en C++14 pour Catch2, et
# la bibliothèque n'a alors pas les StringMaker de std::string_view ni de std::byte qu'ils appellent.
cmake_dep catch2 -DCATCH_INSTALL_DOCS=OFF -DCATCH_INSTALL_EXTRAS=ON -DCMAKE_CXX_STANDARD=17
cmake_dep spdlog -DSPDLOG_USE_STD_FORMAT=ON -DSPDLOG_BUILD_SHARED=OFF -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake_dep vulkan-headers
cmake_dep vulkan-loader -DBUILD_TESTS=OFF

# MoltenVK : l'archive publiée par Khronos, vérifiée. Son manifeste d'ICD va dans share/vulkan/icd.d,
# où le chargeur du préfixe le trouve ; le paquet écrit le sien.
MOLTENVK_VERSION=1.4.2
MOLTENVK_SHA256=f95765a6229cb7b915990a2890ce12ebe36a730b021545d3d52ae69ce4c4024e
if ! done_marker moltenvk "$MOLTENVK_VERSION $MOLTENVK_SHA256"; then
  curl -fsSL -o "$WORK/MoltenVK-macos.tar" \
    "https://github.com/KhronosGroup/MoltenVK/releases/download/v$MOLTENVK_VERSION/MoltenVK-macos.tar"
  echo "$MOLTENVK_SHA256  $WORK/MoltenVK-macos.tar" | shasum -a 256 -c -
  rm -rf "$WORK/MoltenVK"
  tar -xf "$WORK/MoltenVK-macos.tar" -C "$WORK"
  install -d "$PREFIX/lib" "$PREFIX/share/vulkan/icd.d"
  install -m 755 "$WORK/MoltenVK/MoltenVK/dynamic/dylib/macOS/libMoltenVK.dylib" "$PREFIX/lib/"
  sed 's|"library_path": *"[^"]*"|"library_path": "../../../lib/libMoltenVK.dylib"|' \
    "$WORK/MoltenVK/MoltenVK/dynamic/dylib/macOS/MoltenVK_icd.json" >"$PREFIX/share/vulkan/icd.d/MoltenVK_icd.json"
  mark moltenvk "$MOLTENVK_VERSION $MOLTENVK_SHA256"
fi
cmake_dep sdl3 -DSDL_TESTS=OFF -DSDL_EXAMPLES=OFF

if ! done_marker nanobench "$(ref_of nanobench)"; then
  install -d "$PREFIX/include"
  install -m 644 "$WORK/nanobench/src/include/nanobench.h" "$PREFIX/include/nanobench.h"
  mark nanobench "$(ref_of nanobench)"
fi

if ! done_marker imgui "$(ref_of imgui)"; then
  mkdir -p "$PREFIX/share/imgui"
  cp "$WORK/imgui/"*.h "$WORK/imgui/"*.cpp "$PREFIX/share/imgui/"
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
  --buildtype=release -Dvulkan=enabled -Dshaderc=enabled -Dglslang=disabled -Dlcms=enabled
  -Dopengl=disabled -Dd3d11=disabled -Dlibdovi=disabled -Dunwind=disabled -Dxxhash=disabled
  -Ddemos=false -Dtests=false
  -Dvulkan-registry="$PREFIX/share/vulkan/registry/vk.xml")
# Le registre Vulkan vient de vulkan-headers, la libvulkan liée de vulkan-loader : une autre version de l'un
# ou de l'autre rebâtit libplacebo (une copie liée à celle de Homebrew ne se charge plus).
# Le correctif de présentation (chantier 6.1), le même que sous Linux et le port vcpkg : le lecteur lit ses types
# (pl_vulkan_present, placebo_feedback.h) ; sa somme entre dans la signature.
placebo_patch="$ROOT/ports/libplacebo/0001-ajp-present-timing.patch"
signature="$(ref_of libplacebo) ${PLACEBO_OPTS[*]} vulkan-headers $(ref_of vulkan-headers) vulkan-loader $(ref_of vulkan-loader) $(shasum -a 256 "$placebo_patch" | cut -c1-16)"
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

# libass de l'amont, plus le rendu sur plusieurs fils (ports/libass/rendu-sur-plusieurs-fils.patch, comme
# Linux et vcpkg), appliqué une fois sur le clone. Assembleur, libunibreak et fils exigés (P1.24) : sans eux,
# configure passe outre avec un simple avertissement.
LIBASS_PATCH="$ROOT/ports/libass/rendu-sur-plusieurs-fils.patch"
signature="$(ref_of libass) $(shasum -a 256 "$LIBASS_PATCH" | cut -c1-16) asm libunibreak threads $(ref_of libunibreak)"
if ! done_marker libass "$signature"; then
  git -C "$WORK/libass" apply --reverse --check "$LIBASS_PATCH" 2>/dev/null \
    || git -C "$WORK/libass" apply "$LIBASS_PATCH"
  (cd "$WORK/libass" && ./autogen.sh && ./configure --prefix="$PREFIX" --disable-static \
    --enable-asm --enable-libunibreak --enable-threads && make -j "$JOBS" && make install)
  mark libass "$signature"
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
