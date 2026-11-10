#!/usr/bin/env bash
# Paquets Linux d'une version : un seul lecteur pour tous les GPU, et ses moteurs à la demande.
#
#   ajp-linux-x64.tar.gz                  le lecteur, le backend Vulkan, leurs bibliothèques et
#                                         engines.lock (l'index haché des paquets ci-dessous)
#   ajp-engine-tensorrt-linux-x64.tar.gz  backend TensorRT, TensorRT et trtexec (NVIDIA)
#
# Une release peut réunir deux constructions : la CI (le lecteur) et un poste NVIDIA ou le paquet
# TensorRT de la CI (trt-package.yml). AJP_ENGINES_LOCK=<fichier ou lien de l'engines.lock de l'autre> reprend ses
# moteurs non construits ici, dont les paquets restent ceux qu'elle a joints à la release. Le
# poste NVIDIA joint les siens avant l'étiquette, son index renommé engines-nvidia.lock, que
# release-linux.yml reprend.
#
# L'écran de bienvenue propose les moteurs qui conviennent au GPU ; les cochés se téléchargent
# depuis la release (engine_install.h). Chaque moteur n'est empaqueté que s'il a été construit :
#
#   AJP_DEPS_PREFIX=<prefixe> [AJP_WITH_CUDA=ON [AJP_TENSORRT_ROOT=<racine>]] \
#     [AJP_RELEASE_TAG=v0.2.0] tools/release/package-linux.sh [dossier-de-sortie]
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
# Le dépôt par défaut vient de versions.env, la seule source (A28-05) ; AJP_RELEASE_REPO posé l'emporte.
AJP_RELEASE_REPO="${AJP_RELEASE_REPO:-$(sed -n 's/^AJP_RELEASE_REPO=//p' "$ROOT/tools/release/versions.env")}"
BASE_URL="https://github.com/${AJP_RELEASE_REPO}/releases/download/$TAG"

# Une étiquette de préversion (v0.11.0-rc.1) donne son suffixe au binaire : il dit la version de l'étiquette (A26-04).
SUFFIX=""
case "$TAG" in "v$VERSION"-*) SUFFIX="${TAG#"v$VERSION"}" ;; esac
cmake --preset linux-release -S "$ROOT" \
  -DAJP_WITH_CUDA="${AJP_WITH_CUDA:-OFF}" -DAJP_VERSION_SUFFIX="$SUFFIX" \
  ${AJP_TENSORRT_ROOT:+-DAJP_TENSORRT_ROOT="$AJP_TENSORRT_ROOT"}
cmake --build --preset linux-release
BUILD="$ROOT/build/linux-release"

# Bibliothèques du système, jamais livrées : glibc, pilotes et serveurs d'affichage, chargeur
# Vulkan, son.
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
  return 1
}

# Ce que le paquet demande au système doit porter le même nom sur Debian et Ubuntu, Fedora et Arch.
# libbz2.so.1.0, par exemple, n'existe que sous Debian et Ubuntu (Fedora n'a que libbz2.so.1) : le
# FFmpeg livré la demandait et le lecteur ne démarrait pas sous Fedora. Une bibliothèque hors de
# cette liste fait refuser le paquet : la livrer (system_lib), ou s'en passer.
# libwayland-client.so.0 : lue par le retour de présentation du compositeur (wp_presentation, libajp_wayland_presentation) ;
# même nom sur Debian, Ubuntu, Fedora et Arch, et installée avec Mesa et SDL3 sous Wayland.
PORTABLE_SONAMES=" ld-linux-x86-64.so.2 libc.so.6 libm.so.6 libdl.so.2 libpthread.so.0 librt.so.1
  libresolv.so.2 libutil.so.1 libgcc_s.so.1 libstdc++.so.6 libX11.so.6 libxcb.so.1 libdrm.so.2
  libvulkan.so.1 libva.so.2 libva-drm.so.2 libva-x11.so.2 libOpenCL.so.1 libze_loader.so.1
  libcuda.so.1 libfontconfig.so.1 libfreetype.so.6 libharfbuzz.so.0 libz.so.1 liblzma.so.5
  libzstd.so.1 libwayland-client.so.0 "
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
    # Aucun RUNPATH dans le paquet (voir origin) ; trtexec reste tel que la configuration l'a rangé.
    if [ "$(basename "$f")" != trtexec ] && readelf -d "$f" | grep -q '(RUNPATH)'; then
      echo "$f porte un RUNPATH, que LD_LIBRARY_PATH devance : patchelf --force-rpath" >&2
      bad=1
    fi
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

# --force-rpath partout : sans lui patchelf pose un DT_RUNPATH, que LD_LIBRARY_PATH devance. Le paquet
# chargeait alors le libass ou le libplacebo d'un autre logiciel (sans nos correctifs) dès qu'un environnement
# en exportait un. Un DT_RPATH passe avant LD_LIBRARY_PATH et vaut aussi pour les dépendances indirectes.
origin() {  # rpath $ORIGIN (et $ORIGIN/$2) sur chaque fichier ELF du dossier $1
  local f
  for f in "$1"/*; do
    file -b "$f" | grep -q ELF || continue
    # trtexec est déjà rangé par la configuration, et le module garde son empreinte : un octet changé et le
    # moteur le refuse (« trtexec absent du dossier du moteur, ou d'empreinte inattendue », 0.10.0).
    [ "$(basename "$f")" = trtexec ] && continue
    patchelf --force-rpath --set-rpath "\$ORIGIN${2:+:\$ORIGIN/$2}" "$f"
  done
}

# --- Le lecteur. ---
STAGE="$OUT/ajp-linux-x64"
rm -rf "$STAGE"
mkdir -p "$STAGE/lib"
cp "$BUILD/src/app/ajp" "$STAGE/"
cp "$BUILD/src/gpu/vulkan/backend/ajp_backend_vulkan.so" "$STAGE/"
copy_deps "$STAGE/lib" "$STAGE/ajp" "$STAGE/ajp_backend_vulkan.so"
patchelf --force-rpath --set-rpath '$ORIGIN/lib' "$STAGE/ajp" "$STAGE/ajp_backend_vulkan.so"
origin "$STAGE/lib"
check_portable "$STAGE/lib"
check_portable "$STAGE" "$STAGE/lib"
cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$STAGE/"
# De quel commit vient ce paquet : l'étiquette seule ne le garantit pas.
printf 'version %s\ncommit %s\n' "$TAG" "$COMMIT" > "$STAGE/BUILD_INFO.txt"
cp "$ROOT/tools/release/linux/LISEZMOI.txt" "$STAGE/"
# Les LUT d'HDR (data/luts), lues par le lecteur dans luts/ à côté de lui.
cp -R "$ROOT/data/luts" "$STAGE/luts"
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
# TensorRT et RTX HDR ne sont pas publiés (licence NVIDIA) : ils ne s'empaquettent que sur demande
# expresse (AJP_PACKAGE_NVIDIA=1), pour qu'un poste NVIDIA qui construit le paquet ne les joigne pas
# par mégarde. Le chemin normal est la construction sur la machine (docs/nvidia-linux.md).
if [ -f "$TRT_MODULE" ] && [ -z "${AJP_PACKAGE_NVIDIA:-}" ]; then
  echo "TensorRT trouvé, non empaqueté (AJP_PACKAGE_NVIDIA=1 pour le joindre)" >&2
fi
if [ -f "$TRT_MODULE" ] && [ -n "${AJP_PACKAGE_NVIDIA:-}" ]; then
  dir="$OUT/engine-tensorrt"; rm -rf "$dir"; mkdir -p "$dir"
  cp "$TRT_MODULE" "$dir/"
  TRTEXEC="$(sed -n 's/^AJP_TRTEXEC_PACKAGED:[A-Z]*=//p' "$BUILD/CMakeCache.txt")"
  # Sans trtexec, le moteur ne construit aucun réseau : le paquet ne part pas (TRT-04).
  [ -n "$TRTEXEC" ] && [ -x "$TRTEXEC" ] || { echo "trtexec introuvable (AJP_TRTEXEC_PACKAGED : « $TRTEXEC »)" >&2; exit 1; }
  cp "$TRTEXEC" "$dir/trtexec"
  # Le module ne lance que le trtexec dont il garde l'empreinte : un paquet où elle manque ne construirait rien.
  if [ -f "$dir/trtexec" ] && ! strings "$dir/ajp_backend_tensorrt.so" |
      grep -qx "$(sha256sum "$dir/trtexec" | cut -d' ' -f1)"; then
    echo "trtexec : son SHA-256 n'est pas celui que le module attend (reconfigurer la construction)" >&2
    exit 1
  fi
  copy_deps "$dir" "$dir"/*
  # La version entière de TensorRT (11.3.0.99), celle de versions.env que les [[vendor_file]] tirent : jamais le
  # seul numéro majeur, ni la version du lecteur par défaut (A28-09). Elle nomme le dossier d'installation.
  TRT_VERSION="$(sed -n 's/^TENSORRT_LIBS_VERSION=//p' "$ROOT/tools/release/versions.env")"
  [ -n "$TRT_VERSION" ] || { echo "TENSORRT_LIBS_VERSION absent de versions.env" >&2; exit 1; }
  VENDOR="$ROOT/tools/release/nvidia-vendor/tensorrt-linux-x64.lock"
  if [ -z "${AJP_NVIDIA_BUNDLE:-}" ] && [ -f "$VENDOR" ]; then
    # Lien direct : le paquet ne porte que ce qui est à nous (module, trtexec). Les bibliothèques de
    # NVIDIA n'y entrent pas : le lecteur les tire des archives de NVIDIA elles-mêmes, après
    # l'acceptation de sa licence à l'écran (entrées [[vendor_file]], tools/release/vendor-index.py).
    rm -f "$dir"/libnvinfer* "$dir"/libnvonnxparser* "$dir"/libnvrtc* "$dir"/libcu*
    engine_package tensorrt "$TRT_VERSION" \
      "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html" "$dir"
    { echo; cat "$VENDOR"; } >> "$LOCK"
  else
    # Les ressources du constructeur, ouvertes par libnvinfer selon l'architecture du GPU, et les
    # bibliothèques que trtexec ouvre lui-même (dlopen, invisibles pour ldd) : greffons et lecteur ONNX.
    copy_beside "$dir" "$dir/ajp_backend_tensorrt.so" libnvinfer 'libnvinfer_builder_resource*' \
      'libnvinfer_plugin.so.[0-9]*' 'libnvonnxparser.so.[0-9]*'
    for f in "$dir"/*; do [ -e "$STAGE/lib/$(basename "$f")" ] && [ "$f" != "$dir/ajp_backend_tensorrt.so" ] && rm "$f"; done
    check_portable "$dir" "$STAGE/lib"
    engine_package tensorrt "$TRT_VERSION" \
      "https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/sla.html" "$dir"
  fi
fi

if [ -n "${AJP_ENGINES_LOCK:-}" ]; then
  other="$AJP_ENGINES_LOCK"
  case "$other" in
    http://*|https://*) other="$OUT/engines-autre.lock"; curl -fsSL "$AJP_ENGINES_LOCK" -o "$other" ;;
  esac
  have="$(sed -n 's/^id = "\(.*\)"$/\1/p' "$LOCK" | tr '\n' ' ')"
  # Les blocs [[engine]] de l'autre dont l'id manque ici, et les blocs [[vendor_file]] des mêmes moteurs
  # (les fichiers de NVIDIA que le lecteur tire de chez NVIDIA pour TensorRT : sans eux le moteur
  # s'installerait sans bibliothèques) ; ses autres tables ne se reprennent pas.
  awk -v have="$have" '
    BEGIN { n = split(have, ids, " "); for (i = 1; i <= n; i++) skip[ids[i]] = 1 }
    function flush() { if (block != "" && !(id in skip)) printf "\n%s", block; block = ""; id = "" }
    /^\[\[(engine|vendor_file)\]\]$/ { flush(); block = $0 "\n"; inside = 1; next }
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
