#!/usr/bin/env bash
# AppImage du lecteur Linux : un seul fichier à rendre exécutable et à lancer, sans installation.
#
#   tools/release/appimage.sh <ajp-linux-x64.tar.gz> [fichier-de-sortie]
#
# Prend l'archive de package-linux.sh (le lecteur, le backend Vulkan, leurs bibliothèques) et en fait
# ajp-linux-x86_64.AppImage. Les moteurs (TensorRT, OpenVINO, MIGraphX) se téléchargent comme avant dans
# le dossier de données de l'utilisateur ; aucun fichier de NVIDIA n'est dedans. La mise à jour
# automatique ne s'applique pas à un AppImage (sa racine est en lecture seule) : le lecteur le dit et
# renvoie vers la page des releases. La fiche « Ouvrir avec » pointe sur le fichier .AppImage
# (variable APPIMAGE, posée par le lanceur). appimagetool est téléchargé à la version épinglée.
set -euo pipefail

ARCHIVE="${1:?archive ajp-linux-x64.tar.gz}"
OUT="${2:-$(dirname "$ARCHIVE")/ajp-linux-x86_64.AppImage}"
HERE="$(cd "$(dirname "$0")" && pwd)"
TOOL_VERSION="${AJP_APPIMAGETOOL_VERSION:-1.9.0}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

APPDIR="$WORK/AJP.AppDir"
mkdir -p "$APPDIR/usr/lib/ajp"
tar -xzf "$ARCHIVE" -C "$WORK"
STAGE="$WORK/ajp-linux-x64"
[ -x "$STAGE/ajp" ] || { echo "$ARCHIVE : pas de ajp exécutable à la racine" >&2; exit 1; }
cp -a "$STAGE"/. "$APPDIR/usr/lib/ajp/"

cat > "$APPDIR/AppRun" <<'RUN'
#!/bin/sh
HERE="$(dirname "$(readlink -f "$0")")"
export AJP_PACKAGED=appimage
exec "$HERE/usr/lib/ajp/ajp" "$@"
RUN
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/ajp.desktop" <<'DESK'
[Desktop Entry]
Type=Application
Name=Ultimate Anime VideoPlayer
GenericName=Lecteur vidéo
Comment=Anime agrandi et interpolé en temps réel
Icon=ajp
Exec=ajp %f
Terminal=false
Categories=AudioVideo;Video;Player;
MimeType=video/mp4;video/x-matroska;video/webm;video/x-msvideo;video/quicktime;video/mpeg;video/x-flv;video/ogg;
DESK
cp "$HERE/linux/ajp.png" "$APPDIR/ajp.png"

TOOL="$WORK/appimagetool"
curl -fsSL -o "$TOOL" \
  "https://github.com/AppImage/appimagetool/releases/download/$TOOL_VERSION/appimagetool-x86_64.AppImage"
chmod +x "$TOOL"
# Sans FUSE (poste de CI) : l'outil s'extrait lui-même avant de s'exécuter.
ARCH=x86_64 "$TOOL" --appimage-extract-and-run --no-appstream "$APPDIR" "$OUT"
chmod +x "$OUT"
echo "$OUT"
