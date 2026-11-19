#!/usr/bin/env bash
# Paquet macOS du lecteur : Kirari.app (le lecteur, les backends Vulkan et Core ML, leurs
# bibliothèques, ONNX Runtime, MoltenVK et les modèles), signé ad hoc, dans kirari-macos-arm64.zip
# et kirari-macos-arm64.dmg (image disque), chacun avec sa somme SHA-256.
#
#   KIRARI_DEPS_PREFIX=<prefixe> tools/release/package-macos.sh [dossier-de-sortie]
#
# Le préfixe vient de tools/deps/macos-from-source.sh. dylibbundler (Homebrew) recopie dans
# Contents/Frameworks chaque bibliothèque hors du système et réécrit les chemins d'installation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/build/package}" && cd "${1:-$ROOT/build/package}" && pwd)"
: "${KIRARI_DEPS_PREFIX:?KIRARI_DEPS_PREFIX : le préfixe des dépendances}"
BREW="$(brew --prefix)"
brew install --quiet dylibbundler

# Une étiquette de préversion (KIRARI_RELEASE_TAG=v0.11.0-rc.1) donne son suffixe au binaire (A26-04).
VERSION="$(sed -n 's/^  VERSION \([0-9.]*\)$/\1/p' "$ROOT/CMakeLists.txt")"
SUFFIX=""
case "${KIRARI_RELEASE_TAG:-}" in "v$VERSION"-*) SUFFIX="${KIRARI_RELEASE_TAG#"v$VERSION"}" ;; esac
cmake --preset macos-release -S "$ROOT" -DKIRARI_VERSION_SUFFIX="$SUFFIX"
cmake --build --preset macos-release
BUILD="$ROOT/build/macos-release"

APP="$OUT/Kirari.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/PlugIns" "$APP/Contents/Frameworks" \
  "$APP/Contents/Resources/vulkan/icd.d"
cp "$BUILD/src/app/kirari" "$APP/Contents/MacOS/kirari"
cp "$BUILD/src/gpu/vulkan/backend/kirari_backend_vulkan.so" "$APP/Contents/PlugIns/"
# Core ML (le moteur Metal de l'écran de bienvenue) : la chaîne Vulkan, les réseaux par ONNX Runtime.
cp "$BUILD/src/gpu/coreml/kirari_backend_coreml.so" "$APP/Contents/PlugIns/"
mkdir -p "$APP/Contents/Resources/licenses"
cp "$KIRARI_DEPS_PREFIX/onnxruntime/LICENSE" "$APP/Contents/Resources/licenses/onnxruntime.txt"
cp "$ROOT/LICENSE" "$ROOT/THIRD_PARTY_NOTICES.md" "$APP/Contents/Resources/licenses/"
# Les LUT d'HDR (data/luts), lues dans luts/ à côté de SDL_GetBasePath, Contents/Resources ici.
cp -R "$ROOT/data/luts" "$APP/Contents/Resources/luts"
# Les modèles, comme sous Linux et Windows : le lecteur les cherche dans models/ à côté de
# SDL_GetBasePath, Contents/Resources dans un paquet .app (sauf KIRARI_WITHOUT_MODELS).
if [ -z "${KIRARI_WITHOUT_MODELS:-}" ]; then
  command -v 7z >/dev/null || command -v 7zz >/dev/null || brew install --quiet sevenzip
  "$ROOT/tools/release/fetch-models.sh" "$APP/Contents/Resources/models"
fi

# Le chargeur Vulkan et MoltenVK, ouverts à l'exécution : copiés à la main depuis le préfixe, aux
# versions de macos-from-source.sh. Le chargeur lit les pilotes de Contents/Resources/vulkan/icd.d.
cp -L "$KIRARI_DEPS_PREFIX/lib/libvulkan.1.dylib" "$APP/Contents/Frameworks/"
cp "$KIRARI_DEPS_PREFIX/lib/libMoltenVK.dylib" "$APP/Contents/Frameworks/"
cat > "$APP/Contents/Resources/vulkan/icd.d/MoltenVK_icd.json" <<'JSON'
{
  "file_format_version": "1.0.0",
  "ICD": {
    "library_path": "../../../Frameworks/libMoltenVK.dylib",
    "api_version": "1.4.0",
    "is_portability_driver": true
  }
}
JSON

dylibbundler -of -b -cd -d "$APP/Contents/Frameworks" -p @executable_path/../Frameworks/ \
  -s "$KIRARI_DEPS_PREFIX/lib" -s "$KIRARI_DEPS_PREFIX/onnxruntime/lib" -s "$BREW/lib" \
  -x "$APP/Contents/MacOS/kirari" -x "$APP/Contents/PlugIns/kirari_backend_vulkan.so" \
  -x "$APP/Contents/PlugIns/kirari_backend_coreml.so" \
  -x "$APP/Contents/Frameworks/libvulkan.1.dylib" -x "$APP/Contents/Frameworks/libMoltenVK.dylib"
# Le module du backend vit dans PlugIns : ses dépendances se résolvent depuis l'exécutable.
install_name_tool -add_rpath @executable_path/../Frameworks "$APP/Contents/MacOS/kirari" 2>/dev/null || true

# Les vidéos que Kirari.app ouvre depuis le Finder (« Ouvrir avec », puis « Tout modifier » pour en
# faire le lecteur par défaut ; rang Alternate : il ne prend la place de personne de lui-même).
# macOS ne déclare ni .mkv ni .webm : l'app importe leurs types. Le fichier arrive par un Apple
# Event, que SDL3 rend en SDL_EVENT_DROP_FILE.
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Kirari</string>
  <key>CFBundleDisplayName</key><string>Kirari Player</string>
  <key>CFBundleIdentifier</key><string>io.github.ta2b.kirari</string>
  <key>CFBundleExecutable</key><string>kirari</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION:-0.2.0}</string>
  <key>CFBundleVersion</key><string>${VERSION:-0.2.0}</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeName</key><string>Vidéo</string>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSHandlerRank</key><string>Alternate</string>
      <key>LSItemContentTypes</key>
      <array><string>public.movie</string><string>org.matroska.mkv</string><string>org.webmproject.webm</string></array>
    </dict>
  </array>
  <key>UTImportedTypeDeclarations</key>
  <array>
    <dict>
      <key>UTTypeIdentifier</key><string>org.matroska.mkv</string>
      <key>UTTypeDescription</key><string>Vidéo Matroska</string>
      <key>UTTypeConformsTo</key><array><string>public.movie</string></array>
      <key>UTTypeTagSpecification</key>
      <dict>
        <key>public.filename-extension</key><array><string>mkv</string></array>
        <key>public.mime-type</key><array><string>video/x-matroska</string></array>
      </dict>
    </dict>
    <dict>
      <key>UTTypeIdentifier</key><string>org.webmproject.webm</string>
      <key>UTTypeDescription</key><string>Vidéo WebM</string>
      <key>UTTypeConformsTo</key><array><string>public.movie</string></array>
      <key>UTTypeTagSpecification</key>
      <dict>
        <key>public.filename-extension</key><array><string>webm</string></array>
        <key>public.mime-type</key><array><string>video/webm</string></array>
      </dict>
    </dict>
  </array>
</dict>
</plist>
PLIST

# Signature (P7.1). Sans identité, ad hoc : Gatekeeper demande une ouverture par clic droit.
# AJP_MACOS_IDENTITY (« Developer ID Application: … », trousseau déjà importé) : signature Developer ID
# avec le runtime durci et l'horodatage d'Apple. KIRARI_NOTARY_KEY, AJP_NOTARY_KEY_ID et AJP_NOTARY_ISSUER
# (clé d'API App Store Connect, fichier .p8) : notarisation, puis le ticket agrafé à l'app et à l'image.
if [[ -n "${AJP_MACOS_IDENTITY:-}" ]]; then
  codesign --force --deep --options runtime --timestamp --sign "$AJP_MACOS_IDENTITY" "$APP"
  codesign --verify --deep --strict "$APP"
else
  codesign --force --deep --sign - "$APP"
fi
notarize() {
  [[ -n "${AJP_MACOS_IDENTITY:-}" && -n "${KIRARI_NOTARY_KEY:-}" ]] || return 0
  xcrun notarytool submit "$1" --key "$KIRARI_NOTARY_KEY" --key-id "$AJP_NOTARY_KEY_ID" \
    --issuer "$AJP_NOTARY_ISSUER" --wait
}
# L'app est notarisée par un zip de passage, puis agrafée : le zip livré porte le ticket.
if [[ -n "${AJP_MACOS_IDENTITY:-}" && -n "${KIRARI_NOTARY_KEY:-}" ]]; then
  (cd "$OUT" && rm -f notarize.zip && ditto -c -k --keepParent Kirari.app notarize.zip)
  notarize "$OUT/notarize.zip"
  rm -f "$OUT/notarize.zip"
  xcrun stapler staple "$APP"
fi
(cd "$OUT" && rm -f kirari-macos-arm64.zip && ditto -c -k --keepParent Kirari.app kirari-macos-arm64.zip \
  && shasum -a 256 kirari-macos-arm64.zip > kirari-macos-arm64.zip.sha256)
# Image disque : Kirari.app et un raccourci vers Applications, à glisser (même signature que le zip).
DMG_STAGE="$(mktemp -d)"
cp -R "$APP" "$DMG_STAGE/"
ln -s /Applications "$DMG_STAGE/Applications"
(cd "$OUT" && rm -f kirari-macos-arm64.dmg \
  && hdiutil create -volname "Kirari Player" -srcfolder "$DMG_STAGE" -ov -format UDZO kirari-macos-arm64.dmg)
rm -rf "$DMG_STAGE"
if [[ -n "${AJP_MACOS_IDENTITY:-}" ]]; then
  codesign --force --timestamp --sign "$AJP_MACOS_IDENTITY" "$OUT/kirari-macos-arm64.dmg"
  if [[ -n "${KIRARI_NOTARY_KEY:-}" ]]; then
    notarize "$OUT/kirari-macos-arm64.dmg"
    xcrun stapler staple "$OUT/kirari-macos-arm64.dmg"
  fi
fi
(cd "$OUT" && shasum -a 256 kirari-macos-arm64.dmg > kirari-macos-arm64.dmg.sha256)
echo "Paquet : $OUT/kirari-macos-arm64.zip et kirari-macos-arm64.dmg"
