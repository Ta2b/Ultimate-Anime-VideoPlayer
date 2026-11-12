#!/usr/bin/env bash
# Construit les dépendances du lecteur depuis leurs dépôts git, aux commits fixés
# dans DEPS ci-dessous, dans un préfixe local.
#
# C'est la source des dépendances Linux de la CI et du paquet Linux, et celle des
# postes de développement. vcpkg (vcpkg.json, ports/) ne sert qu'à Windows : les
# versions des deux se suivent à la main, et tools/ci/check-deps-versions.sh refuse un écart
# pour ce que DEPS fixe. Ce qui vient d'apt (dav1d, lcms2, shaderc, FriBidi, FreeType,
# HarfBuzz) suit la distribution, pas la baseline de vcpkg.
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
  "imgui https://github.com/ocornut/imgui v1.93.0 c1aaac65e896a45f81f47a4d68f9959e0f86c859"
  "nv-codec-headers https://github.com/FFmpeg/nv-codec-headers n13.0.19.0 e844e5b26f46bb77479f063029595293aa8f812d"
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
# Toute signature porte aussi le compilateur : un préfixe du cache construit par un autre GCC se refait.
# system_libs donne la version des bibliothèques du système qu'une dépendance lie (dav1d, shaderc, lcms2…) :
# quand l'image du runner ou la distribution les monte, la dépendance se relie au lieu de garder l'ancienne.
COMPILER="$("${CC:-cc}" -dumpfullversion 2>/dev/null || echo inconnu)"
system_libs() {  # modules pkg-config...
  local module out=""
  for module in "$@"; do out+=" $module=$(pkg-config --modversion "$module" 2>/dev/null || echo absent)"; done
  echo "$out"
}
done_marker() { [[ -f "$PREFIX/.built-$1" && "$(<"$PREFIX/.built-$1")" == "$2 cc $COMPILER" ]]; }
mark() { printf '%s\n' "$2 cc $COMPILER" >"$PREFIX/.built-$1"; }

# Remet le clone d'une dépendance construite dans l'arbre (FFmpeg, libass, libunibreak, libplacebo) à son commit,
# sans reste d'un correctif ou d'une configuration plus anciens : quand sa signature change (correctif réécrit,
# option modifiée), le dossier de travail d'un poste garde sinon l'ancien correctif appliqué, `git apply` du
# nouveau échoue et `set -e` laisse le préfixe à moitié mis à jour. -ff retire aussi les dépôts imbriqués.
pristine() {  # nom
  git -C "$WORK/$1" reset --hard -q
  git -C "$WORK/$1" clean -ffdxq
}

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

# imgui n'a pas de système de construction : on installe ses sources, que
# cmake/Dependencies.cmake compile, sans backend (le lecteur n'en appelle aucun).
if ! done_marker imgui "$(ref_of imgui) sans-backends"; then
  rm -rf "$PREFIX/share/imgui"
  mkdir -p "$PREFIX/share/imgui"
  cp "$WORK/imgui/"*.h "$WORK/imgui/"*.cpp "$PREFIX/share/imgui/"
  # Le chargeur FreeType : stb_truetype ne lit pas les polices OpenType à contours CFF2 (Noto Sans CJK variable de Fedora).
  mkdir -p "$PREFIX/share/imgui/misc/freetype"
  cp "$WORK/imgui/misc/freetype/"imgui_freetype.* "$PREFIX/share/imgui/misc/freetype/"
  mark imgui "$(ref_of imgui) sans-backends"
fi

if ! done_marker nv-codec-headers "$(ref_of nv-codec-headers)"; then
  make -C "$WORK/nv-codec-headers" PREFIX="$PREFIX" install
  mark nv-codec-headers "$(ref_of nv-codec-headers)"
fi

# FFmpeg en LGPL partagée, comme le triplet dynamique de vcpkg. VA-API demande les en-têtes de
# libva (libva-dev, libva-devel) ; la libva du système sert à l'exécution. Sans bzip2 : il ne
# sert qu'aux pistes MKV compressées en bzip2, et sa bibliothèque ne porte pas le même nom
# partout (libbz2.so.1.0 sous Debian et Ubuntu, libbz2.so.1 sous Fedora : package-linux.sh).
# Sans détection automatique, comme le port vcpkg (--disable-autodetect) : seules les options écrites ici entrent
# dans les bibliothèques, pas ce que la machine de construction a d'installé (lzma, alsa, xcb…). libdrm reste voulu
# (export DRM-PRIME de VA-API vers Vulkan, libdrm-dev dans APT_PLAYER). zlib est
# demandé des deux côtés : il ouvre les pistes Matroska dont les en-têtes sont compressés (ContentCompression).
FFMPEG_OPTS=(--prefix="$PREFIX" --enable-shared --disable-static
  --disable-programs --disable-doc --disable-debug --disable-autodetect --disable-bzlib --enable-zlib
  --enable-ffnvcodec --enable-nvdec --enable-cuvid
  --enable-vulkan --enable-vaapi --enable-libdrm --enable-libdav1d --enable-libsvtav1
  --extra-cflags="-I$PREFIX/include" --extra-ldflags="-L$PREFIX/lib")
signature="$(ref_of ffmpeg) ${FFMPEG_OPTS[*]} nv-codec-headers $(ref_of nv-codec-headers)$(system_libs dav1d SvtAv1Enc libva)"
if ! done_marker ffmpeg "$signature"; then
  pristine ffmpeg
  (cd "$WORK/ffmpeg" && ./configure "${FFMPEG_OPTS[@]}" && make -j "$JOBS" && make install)
  mark ffmpeg "$signature"
fi

# Les mêmes options que le port vcpkg (ports/libplacebo/portfile.cmake), fixées : laissées en « auto », elles
# suivaient ce que la machine avait d'installé (glslang, xxhash, libunwind, libdovi), et le libplacebo d'un
# poste n'était pas celui du paquet. Une différence voulue : vk-proc-addr reste allumé ici, les tests de rendu
# ouvrent Vulkan par le chargeur de libplacebo (sans lui ils se sautent tous, « pas d'appareil Vulkan »).
PLACEBO_OPTS=(--prefix="$PREFIX" --libdir=lib
  --buildtype=release -Dvulkan=enabled -Dvk-proc-addr=enabled -Dshaderc=enabled -Dglslang=disabled
  -Dlcms=enabled -Dopengl=disabled -Dd3d11=disabled -Dlibdovi=disabled -Dunwind=disabled
  -Dxxhash=disabled -Ddemos=false -Dtests=false
  -Dvulkan-registry="$PREFIX/share/vulkan/registry/vk.xml" -Dc_args="-I$PREFIX/include")
# Les en-têtes et le registre Vulkan viennent de vulkan-headers, comme sous Windows (C27-01) : le sous-module
# 3rdparty/Vulkan-Headers de libplacebo, s'il est initialisé, passe devant et force son propre vk.xml (meson.build de
# src/vulkan : « Prefer our Vulkan headers »), d'une autre version que celle du correctif de présentation. Seuls les
# trois sous-modules que le port Windows prend sont donc initialisés, et -I place les en-têtes du préfixe devant
# ceux du système. Une autre version de vulkan-headers rebâtit libplacebo (sa version entre dans la signature).
# Le correctif de présentation (chantier 6.1), le même que le port vcpkg : sa somme entre dans la signature.
placebo_patch="$ROOT/ports/libplacebo/0001-ajp-present-timing.patch"
signature="$(ref_of libplacebo) ${PLACEBO_OPTS[*]} vulkan-headers $(ref_of vulkan-headers) $(sha256sum "$placebo_patch" | cut -c1-16)$(system_libs shaderc lcms2)"
if ! done_marker libplacebo "$signature"; then
  pristine libplacebo
  # Un clone gardé d'avant ce réglage peut avoir le sous-module des en-têtes Vulkan : il le ferait préférer à ceux du préfixe.
  git -C "$WORK/libplacebo" submodule deinit -f -q -- 3rdparty/Vulkan-Headers 2>/dev/null || true
  rm -rf "$WORK/libplacebo/3rdparty/Vulkan-Headers/include" "$WORK/libplacebo/3rdparty/Vulkan-Headers/registry"
  git -C "$WORK/libplacebo" submodule update --init --depth 1 -q 3rdparty/jinja 3rdparty/markupsafe 3rdparty/fast_float
  git -C "$WORK/libplacebo" apply "$placebo_patch"
  rm -rf "$WORK/libplacebo/_build"
  meson setup "$WORK/libplacebo/_build" "$WORK/libplacebo" "${PLACEBO_OPTS[@]}"
  # Aucune compilation ne doit lire les en-têtes d'un sous-module (C27-01) : le correctif de présentation est écrit
  # pour ceux de vulkan-headers.
  if grep -q '3rdparty/Vulkan-Headers' "$WORK/libplacebo/_build/compile_commands.json"; then
    echo "libplacebo : construit avec les en-têtes de 3rdparty/Vulkan-Headers au lieu de ceux de $PREFIX" >&2
    exit 1
  fi
  meson compile -C "$WORK/libplacebo/_build"
  meson install -C "$WORK/libplacebo/_build"
  mark libplacebo "$signature"
fi

# La coupure des lignes selon Unicode (UAX #14), que libass prend à libunibreak.
if ! done_marker libunibreak "$(ref_of libunibreak)"; then
  pristine libunibreak
  (cd "$WORK/libunibreak" && ./autogen.sh && ./configure --prefix="$PREFIX" --disable-static \
    && make -j "$JOBS" && make install)
  mark libunibreak "$(ref_of libunibreak)"
fi

# libass de l'amont, plus le rendu sur plusieurs fils (ports/libass/rendu-sur-plusieurs-fils.patch, le même
# que le port de vcpkg), appliqué une fois sur le clone. Assembleur, libunibreak et fils exigés (P1.24) : sans
# eux, configure passe outre avec un simple avertissement, et le rendu des sous-titres perd ses routines SIMD, la
# coupure Unicode ou ses fils.
LIBASS_PATCH="$ROOT/ports/libass/rendu-sur-plusieurs-fils.patch"
# AJP_LIBASS_CFLAGS : les options de compilation de libass seul, pour l'instrumenter (« -fsanitize=thread -g -O1 »
# dans la tâche TSan de ci.yml, « -fsanitize=address -g -O1 » dans la campagne de fuzz). Sans elles, le code multi-fils
# du correctif, qui lit des scripts ASS venus de l'extérieur, échappait à TSan et à ASan (B35-02) : l'instrumentation
# du lecteur s'arrête à la frontière de la bibliothèque. Elles entrent dans la signature : un préfixe du cache construit
# autrement se refait.
LIBASS_CFLAGS="${AJP_LIBASS_CFLAGS:--O2 -g}"
signature="$(ref_of libass) $(sha256sum "$LIBASS_PATCH" | cut -c1-16) asm libunibreak threads cflags $LIBASS_CFLAGS $(ref_of libunibreak)$(system_libs freetype2 fribidi harfbuzz fontconfig)"
if ! done_marker libass "$signature"; then
  pristine libass
  git -C "$WORK/libass" apply "$LIBASS_PATCH"
  (cd "$WORK/libass" && ./autogen.sh && ./configure --prefix="$PREFIX" --disable-static \
    --enable-asm --enable-libunibreak --enable-threads CFLAGS="$LIBASS_CFLAGS" && make -j "$JOBS" && make install)
  mark libass "$signature"
fi

echo "Dépendances prêtes dans $PREFIX"
