#!/usr/bin/env bash
# Paquets Linux d'une version : un seul lecteur pour tous les GPU, et ses moteurs à la demande.
#
#   ajp-linux-x64.tar.gz                  le lecteur, le backend Vulkan, leurs bibliothèques et
#                                         engines.lock (l'index haché des paquets ci-dessous)
#   ajp-engine-tensorrt-linux-x64.tar.gz  backend TensorRT, TensorRT et trtexec (NVIDIA)
#   ajp-engine-migraphx-linux-x64.tar.gz  backend MIGraphX (AMD ; ROCm du système)
#   ajp-engine-openvino-linux-x64.tar.gz  backend OpenVINO, son runtime et son greffon GPU (Intel)
#
#   ajp-engine-maxine_vfx-linux-x64.tar.gz  RTX HDR : les bibliothèques TrueHDR du SDK NVIDIA, intactes
#
# Une release peut réunir deux constructions : la CI (OpenVINO, MIGraphX) et un poste NVIDIA
# (TensorRT, RTX HDR). AJP_ENGINES_LOCK=<fichier ou lien de l'engines.lock de l'autre> reprend ses
# moteurs non construits ici, dont les paquets restent ceux qu'elle a joints à la release. Le
# poste NVIDIA joint les siens avant l'étiquette, son index renommé engines-nvidia.lock, que
# release-linux.yml reprend.
#
# L'écran de bienvenue propose les moteurs qui conviennent au GPU ; les cochés se téléchargent
# depuis la release (engine_install.h). Chaque moteur n'est empaqueté que s'il a été construit :
#
#   AJP_DEPS_PREFIX=<prefixe> [AJP_WITH_CUDA=ON [AJP_TENSORRT_ROOT=<racine>]] [AJP_WITH_MIGRAPHX=ON] \
#     [AJP_WITH_OPENVINO=ON] [AJP_RELEASE_TAG=v0.2.0] tools/release/package-linux.sh [dossier-de-sortie]
#
# patchelf range chaque binaire sur $ORIGIN : rien ne dépend du préfixe de construction.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/build/package}" && cd "${1:-$ROOT/build/package}" && pwd)"
: "${AJP_DEPS_PREFIX:?AJP_DEPS_PREFIX : le préfixe des dépendances}"
command -v patchelf >/dev/null || { echo "patchelf manquant (apt install patchelf)" >&2; exit 1; }
VERSION="$(sed -n 's/^  VERSION \([0-9.]*\)$/\1/p' "$ROOT/CMakeLists.txt")"
TAG="${AJP_RELEASE_TAG:-v$VERSION}"
# Archives reproductibles : mêmes fichiers, mêmes octets. Sans cela, remplacer un paquet d'une
# release publiée (gh release upload --clobber) change son SHA-256 et invalide les engines.lock des
# lecteurs déjà installés. Date : celle du commit, ou SOURCE_DATE_EPOCH.
COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo inconnu)"
EPOCH="${SOURCE_DATE_EPOCH:-$(git -C "$ROOT" log -1 --format=%ct 2>/dev/null || echo 0)}"
pack() {  # dossier-parent entrée archive.tar.gz : tar trié, propriétaire neutre, gzip sans nom ni date
  tar --sort=name --owner=0 --group=0 --numeric-owner --mtime="@$EPOCH" -C "$1" -cf - "$2" | gzip -9n > "$3"
}
# Le dépôt dont la release porte les paquets : il doit être public pour que le lecteur les télécharge.
BASE_URL="https://github.com/${AJP_RELEASE_REPO:-Ta2b/Ultimate-Anime-VideoPlayer}/releases/download/$TAG"

cmake --preset linux-release -S "$ROOT" \
  -DAJP_WITH_CUDA="${AJP_WITH_CUDA:-OFF}" \
  ${AJP_TENSORRT_ROOT:+-DAJP_TENSORRT_ROOT="$AJP_TENSORRT_ROOT"} \
  -DAJP_WITH_MIGRAPHX="${AJP_WITH_MIGRAPHX:-OFF}" \
  -DAJP_WITH_OPENVINO="${AJP_WITH_OPENVINO:-OFF}"
cmake --build --preset linux-release
BUILD="$ROOT/build/linux-release"

# Bibliothèques du système, jamais livrées : glibc, pilotes et serveurs d'affichage, chargeur
# Vulkan, son, ROCm.
system_lib() {
  case "$(basename "$1")" in
    ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|libutil.so*) return 0 ;;
    libvulkan.so*|libGL*|libEGL*|libGLX*|libOpenGL*|libnvidia*|libcuda.so*|libdrm*|libgbm*) return 0 ;;
    libX*|libxcb*|libxkbcommon*|libwayland*|libdecor*|libasound*|libpulse*|libpipewire*) return 0 ;;
    libdbus*|libsystemd*|libudev*|libcap*|libgcrypt*|libgpg-error*|liblzma*|libz.so*|libzstd*) return 0 ;;
    libffi*|libexpat*|libbsd*|libmd.so*) return 0 ;;
    libva.so*|libva-*) return 0 ;;  # VA-API : la libva du système trouve ses pilotes (iHD, radeonsi)
    libOpenCL.so*|libze_loader.so*) return 0 ;;  # chargeurs OpenCL et Level Zero (pilotes Intel)
    # Chargées aussi par les pilotes (Mesa, VA-API) : une copie plus ancienne que celle du système
    # passe avant elle et les casse (symbole manquant, configuration de /etc/fonts illisible).
    # La glibc 2.38 qu'exige la construction (Ubuntu 24.04) les garantit assez récentes.
    libstdc++.so*|libgcc_s.so*) return 0 ;;
    libglib-2.0*|libgobject-2.0*|libgio-2.0*|libgmodule-2.0*|libpcre2*) return 0 ;;
    libfontconfig*|libfreetype*|libharfbuzz*|libgraphite2*|libpng16*|libbrotli*|libbz2*) return 0 ;;
  esac
  # Celles du runtime HIP, que tout système ROCm installe avec lui.
  case "$(basename "$1")" in libelf.so*|libnuma.so*) return 0 ;; esac
  case "$1" in /opt/rocm*) return 0 ;; esac
  return 1
}

# Ce que le paquet demande au système doit porter le même nom sur Debian et Ubuntu, Fedora et Arch.
# libbz2.so.1.0, par exemple, n'existe que sous Debian et Ubuntu (Fedora n'a que libbz2.so.1) : le
# FFmpeg livré la demandait et le lecteur ne démarrait pas sous Fedora. Une bibliothèque hors de
# cette liste fait refuser le paquet : la livrer (system_lib), ou s'en passer.
PORTABLE_SONAMES=" ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1
  libresolv.so.2 libutil.so.1 libgcc_s.so.1 libstdc++.so.6 libX11.so.6 libxcb.so.1 libdrm.so.2
  libvulkan.so.1 libva.so.2 libva-drm.so.2 libva-x11.so.2 libOpenCL.so.1 libze_loader.so.1
  libcuda.so.1 libfontconfig.so.1 libfreetype.so.6 libharfbuzz.so.0 libz.so.1 liblzma.so.5
  libzstd.so.1 "
PORTABLE_SONAMES="${PORTABLE_SONAMES//$'\n'/ }"

check_portable() {  # dossier des binaires vérifiés, [autres dossiers livrés à côté…]
  local bad=0 f soname dir
  while read -r f; do
    while read -r soname; do
      for dir in "$@"; do [ -e "$dir/$soname" ] && continue 2; done
      case "$PORTABLE_SONAMES" in *" $soname "*) continue ;; esac
      echo "$f demande $soname au système, un nom qui n'est pas le même partout : à livrer ou à retirer" >&2
      bad=1
    done < <(readelf -d "$f" | sed -n 's/.*(NEEDED).*\[\(.*\)\]/\1/p')
  done < <(find "$1" -maxdepth 1 -type f -exec sh -c 'file -b "$1" | grep -q ELF' _ {} \; -print)
  return "$bad"
}

# Les dépendances hors système de chaque binaire donné, copiées dans $1 (sans écraser).
copy_deps() {
  local dest="$1"; shift
  local bin lib
  for bin in "$@"; do
    ldd "$bin" | awk '/=> \// {print $3}' | while read -r lib; do
      system_lib "$lib" && continue
      [ -e "$dest/$(basename "$lib")" ] || cp -L "$lib" "$dest/"
    done
  done
}

# ldd ne voit pas ce qu'une bibliothèque ouvre à l'exécution : copiés par motif depuis le
# dossier d'une dépendance connue.
copy_beside() {  # dossier binaire nom-de-la-dépendance motifs…
  local dest="$1" known="$2" name="$3"; shift 3
  local dir pattern f
  dir="$(dirname "$(ldd "$known" | awk -v n="$name" 'index($1, n) == 1 {print $3; exit}')")"
  for pattern in "$@"; do
    for f in "$dir"/$pattern; do [ -e "$f" ] && cp -L "$f" "$dest/"; done
  done
  return 0
}

origin() {  # rpath $ORIGIN (et $ORIGIN/$2) sur chaque fichier ELF du dossier $1
  local f
  for f in "$1"/*; do
    file -b "$f" | grep -q ELF || continue
    patchelf --set-rpath "\$ORIGIN${2:+:\$ORIGIN/$2}" "$f"
  done
}

# --- Le lecteur. ---
STAGE="$OUT/ajp-linux-x64"
rm -rf "$STAGE"
mkdir -p "$STAGE/lib"
cp "$BUILD/src/app/ajp" "$STAGE/"
cp "$BUILD/src/gpu/vulkan/backend/ajp_backend_vulkan.so" "$STAGE/"
copy_deps "$STAGE/lib" "$STAGE/ajp" "$STAGE/ajp_backend_vulkan.so"
patchelf --set-rpath '$ORIGIN/lib' "$STAGE/ajp" "$STAGE/ajp_backend_vulkan.so"
origin "$STAGE/lib"
check_portable "$STAGE/lib"
check_portable "$STAGE" "$STAGE/lib"
cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"
# De quel commit vient ce paquet : l'étiquette seule ne le garantit pas.
printf 'version %s\ncommit %s\n' "$TAG" "$COMMIT" > "$STAGE/BUILD_INFO.txt"
cp "$ROOT/tools/release/linux/LISEZMOI.txt" "$STAGE/"
# Les modèles SPAN et RIFE, pour lire dès l'extraction (AJP_WITHOUT_MODELS=1 pour s'en passer).
[ -n "${AJP_WITHOUT_MODELS:-}" ] || "$ROOT/tools/release/fetch-models.sh" "$STAGE/models"

# --- Les moteurs. ---
LOCK="$STAGE/engines.lock"
echo "version = 1" > "$LOCK"

engine_package() {  # id version licence dossier [intact : binaires laissés tels quels]
  local id="$1" version="$2" license="$3" dir="$4"
  local archive="ajp-engine-$id-linux-x64.tar.gz"
  [ -n "${5:-}" ] || origin "$dir"
  pack "$dir" . "$OUT/$archive"
  {
    echo
    echo "[[engine]]"
    echo "id = \"$id\""
    echo "platform = \"linux\""
    echo "url = \"$BASE_URL/$archive\""
    echo "sha256 = \"$(sha256sum "$OUT/$archive" | cut -d' ' -f1)\""
    echo "size = $(stat -c %s "$OUT/$archive")"
    echo "version = \"$version\""
    [ -n "$license" ] && echo "license = \"$license\""
  } >> "$LOCK"
  echo "moteur $id : $OUT/$archive"
}

TRT_MODULE="$BUILD/src/gpu/cuda/backend/ajp_backend_tensorrt.so"
if [ -f "$TRT_MODULE" ]; then
  dir="$OUT/engine-tensorrt"; rm -rf "$dir"; mkdir -p "$dir"
  cp "$TRT_MODULE" "$dir/"
  TRTEXEC="$(sed -n 's/^AJP_TRTEXEC:[A-Z]*=//p' "$BUILD/CMakeCache.txt")"
  [ -n "$TRTEXEC" ] && [ -x "$TRTEXEC" ] && cp "$TRTEXEC" "$dir/trtexec"
  copy_deps "$dir" "$dir"/*
  # Les ressources du constructeur, ouvertes par libnvinfer selon l'architecture du GPU, et les
  # bibliothèques que trtexec ouvre lui-même (dlopen, invisibles pour ldd) : greffons et lecteur ONNX.
  copy_beside "$dir" "$dir/ajp_backend_tensorrt.so" libnvinfer 'libnvinfer_builder_resource*' \
    'libnvinfer_plugin.so.[0-9]*' 'libnvonnxparser.so.[0-9]*'
  for f in "$dir"/*; do [ -e "$STAGE/lib/$(basename "$f")" ] && [ "$f" != "$dir/ajp_backend_tensorrt.so" ] && rm "$f"; done
  check_portable "$dir" "$STAGE/lib"
  TRT_VERSION="$(basename "$(ls "$dir"/libnvinfer.so.* | sort | tail -1)" | sed 's/^libnvinfer\.so\.//')"
  engine_package tensorrt "${TRT_VERSION:-$VERSION}" \
    "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html" "$dir"
fi

MX_MODULE="$BUILD/src/gpu/migraphx/ajp_backend_migraphx.so"
if [ -f "$MX_MODULE" ]; then
  dir="$OUT/engine-migraphx"; rm -rf "$dir"; mkdir -p "$dir"
  cp "$MX_MODULE" "$dir/"
  copy_deps "$dir" "$MX_MODULE"
  for f in "$dir"/*; do [ -e "$STAGE/lib/$(basename "$f")" ] && [ "$f" != "$dir/ajp_backend_migraphx.so" ] && rm "$f"; done
  # ROCm reste celui du système, dans /opt/rocm/lib, que ld.so.conf ne liste pas toujours.
  origin "$dir"
  patchelf --set-rpath '$ORIGIN:/opt/rocm/lib' "$dir/ajp_backend_migraphx.so"
  engine_package migraphx "$VERSION" "" "$dir" intact
fi

OV_MODULE="$BUILD/src/gpu/openvino/ajp_backend_openvino.so"
if [ -f "$OV_MODULE" ]; then
  dir="$OUT/engine-openvino"; rm -rf "$dir"; mkdir -p "$dir"
  cp "$OV_MODULE" "$dir/"
  copy_deps "$dir" "$OV_MODULE"
  # Les greffons GPU et NPU (Core Ultra) et le lecteur ONNX, ouverts par libopenvino depuis son
  # propre dossier, avec cache.json, les noyaux réglés que le greffon GPU choisit par appareil.
  copy_beside "$dir" "$OV_MODULE" libopenvino 'libopenvino_intel_gpu_plugin.so' 'libopenvino_intel_npu_plugin.so' \
    'libopenvino_onnx_frontend.so*' 'cache.json'
  copy_deps "$dir" "$dir"/libopenvino_*
  for f in "$dir"/*; do [ -e "$STAGE/lib/$(basename "$f")" ] && [ "$f" != "$dir/ajp_backend_openvino.so" ] && rm "$f"; done
  check_portable "$dir" "$STAGE/lib"
  engine_package openvino "$VERSION" "" "$dir"
fi

# RTX HDR : un paquet fermé à part, sous la licence de NVIDIA. Elle permet de livrer le SDK comme
# partie du lecteur (§1.1.3), jamais sous sa licence libre (§8.8) : le paquet ne contient que ce
# que le magasin des SDK copie pour TrueHDR (vendor_hdr/sdk_layout.cpp, kMaxineCopied), sans
# modification (pas de patchelf), et le lecteur le télécharge sans clé ni compte NVIDIA puis
# l'importe dans son magasin, qui le vérifie (maxine_install.h). Depuis le SDK installé :
#   AJP_MAXINE_SDK=/usr/local/VideoFX
if [ -n "${AJP_MAXINE_SDK:-}" ]; then
  dir="$OUT/engine-maxine"; rm -rf "$dir"; mkdir -p "$dir/external/cuda/lib"
  for entry in lib include share features/nvvfxtruehdr; do
    [ -e "$AJP_MAXINE_SDK/$entry" ] || { echo "$AJP_MAXINE_SDK/$entry manquant" >&2; exit 1; }
    mkdir -p "$dir/$(dirname "$entry")"
    cp -a "$AJP_MAXINE_SDK/$entry" "$dir/$entry"
  done
  for lib in libcudart.so.12 libnppc.so.12 libnppial.so.12 libnppicc.so.12 libnppidei.so.12 libnppig.so.12 \
    libcudnn.so.9; do
    cp -L "$AJP_MAXINE_SDK/external/cuda/lib/$lib" "$dir/external/cuda/lib/"
  done
  cp "$ROOT/tools/release/linux/LISEZMOI-NVIDIA.txt" "$dir/"
  header="$AJP_MAXINE_SDK/share/videofx_version.h"
  part() { sed -n "s/^#define NVIDIA_VIDEOEFFECTS_SDK_VERSION_$1 *\([0-9]*\).*/\1/p" "$header"; }
  sdk_version="$VERSION"
  if [ -f "$header" ]; then
    parsed="$(part MAJOR).$(part MINOR).$(part RELEASE).$(part BUILD)"
    [[ "$parsed" =~ ^[0-9]+\.[0-9]+ ]] && sdk_version="$parsed"
  fi
  engine_package maxine_vfx "$sdk_version" \
    "https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/" "$dir" intact
fi

# À défaut du paquet, engines.lock peut nommer la ressource NGC du SDK, que le lecteur télécharge
# avec la clé d'API de l'utilisateur (maxine_install.h). Les noms
# se lisent sur la page de la ressource « Video Effects SDK Core » du catalogue NGC, connecté :
#   AJP_MAXINE_RESOURCE=<nom> AJP_MAXINE_VERSION=<version> AJP_MAXINE_FILE=<archive .tgz>
#   AJP_MAXINE_LICENSE=<lien https de sa licence>
if [ -n "${AJP_MAXINE_RESOURCE:-}" ]; then
  {
    echo
    echo "[maxine_vfx]"
    echo "org = \"${AJP_MAXINE_ORG:-nvidia}\""
    echo "team = \"${AJP_MAXINE_TEAM:-maxine}\""
    echo "resource = \"$AJP_MAXINE_RESOURCE\""
    echo "version = \"${AJP_MAXINE_VERSION:?AJP_MAXINE_VERSION}\""
    echo "file = \"${AJP_MAXINE_FILE:?AJP_MAXINE_FILE}\""
    echo "feature = \"${AJP_MAXINE_FEATURE:-nvvfxtruehdr}\""
    echo "license = \"${AJP_MAXINE_LICENSE:?AJP_MAXINE_LICENSE : lien https de la licence du SDK}\""
  } >> "$LOCK"
fi

if [ -n "${AJP_ENGINES_LOCK:-}" ]; then
  other="$AJP_ENGINES_LOCK"
  case "$other" in
    http://*|https://*) other="$OUT/engines-autre.lock"; curl -fsSL "$AJP_ENGINES_LOCK" -o "$other" ;;
  esac
  have="$(sed -n 's/^id = "\(.*\)"$/\1/p' "$LOCK" | tr '\n' ' ')"
  # Les blocs [[engine]] de l'autre dont l'id manque ici ; ses autres tables ne se reprennent pas.
  awk -v have="$have" '
    BEGIN { n = split(have, ids, " "); for (i = 1; i <= n; i++) skip[ids[i]] = 1 }
    function flush() { if (block != "" && !(id in skip)) printf "\n%s", block; block = ""; id = "" }
    /^\[\[engine\]\]$/ { flush(); block = $0 "\n"; inside = 1; next }
    /^\[/ { flush(); inside = 0; next }
    inside && /^id = / { id = $3; gsub(/"/, "", id) }
    inside && NF { block = block $0 "\n" }
    END { flush() }' "$other" >> "$LOCK"
  echo "moteurs d'engines.lock : $(sed -n 's/^id = "\(.*\)"$/\1/p' "$LOCK" | tr '\n' ' ')"
fi
cp "$LOCK" "$OUT/engines.lock"

pack "$OUT" ajp-linux-x64 "$OUT/ajp-linux-x64.tar.gz"
( cd "$OUT" && sha256sum ajp-*.tar.gz engines.lock > SHA256SUMS )
echo "lecteur : $OUT/ajp-linux-x64.tar.gz"
echo "à joindre à la release $TAG : $OUT/ajp-*.tar.gz, engines.lock et SHA256SUMS"
