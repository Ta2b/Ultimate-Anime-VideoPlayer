#!/usr/bin/env bash
# Paquet macOS du lecteur : AJP.app (le lecteur, les backends Vulkan et Core ML, leurs
# bibliothèques, ONNX Runtime, MoltenVK et les modèles), signé ad hoc, dans ajp-macos-arm64.zip
# et sa somme SHA-256.
#
#   AJP_DEPS_PREFIX=<prefixe> tools/release/package-macos.sh [dossier-de-sortie]
#
# Le préfixe vient de tools/deps/macos-from-source.sh. dylibbundler (Homebrew) recopie dans
# Contents/Frameworks chaque bibliothèque hors du système et réécrit les chemins d'installation.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OUT="$(mkdir -p "${1:-$ROOT/build/package}" && cd "${1:-$ROOT/build/package}" && pwd)"
: "${AJP_DEPS_PREFIX:?AJP_DEPS_PREFIX : le préfixe des dépendances}"
BREW="$(brew --prefix)"
brew install --quiet dylibbundler

cmake --preset macos-release -S "$ROOT"
cmake --build --preset macos-release
BUILD="$ROOT/build/macos-release"

APP="$OUT/AJP.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/PlugIns" "$APP/Contents/Frameworks" \
  "$APP/Contents/Resources/vulkan/icd.d"
cp "$BUILD/src/app/ajp" "$APP/Contents/MacOS/ajp"
cp "$BUILD/src/gpu/vulkan/backend/ajp_backend_vulkan.so" "$APP/Contents/PlugIns/"
# Core ML (le moteur Metal de l'écran de bienvenue) : la chaîne Vulkan, les réseaux par ONNX Runtime.
cp "$BUILD/src/gpu/coreml/ajp_backend_coreml.so" "$APP/Contents/PlugIns/"
mkdir -p "$APP/Contents/Resources/licenses"
cp "$AJP_DEPS_PREFIX/onnxruntime/LICENSE" "$APP/Contents/Resources/licenses/onnxruntime.txt"
# Les modèles, comme sous Linux et Windows : le lecteur les cherche dans models/ à côté de
# SDL_GetBasePath, Contents/Resources dans un paquet .app (sauf AJP_WITHOUT_MODELS).
if [ -z "${AJP_WITHOUT_MODELS:-}" ]; then
  command -v 7z >/dev/null || command -v 7zz >/dev/null || brew install --quiet sevenzip
  "$ROOT/tools/release/fetch-models.sh" "$APP/Contents/Resources/models"
fi

# Le chargeur Vulkan et MoltenVK, ouverts à l'exécution : copiés à la main. Le chargeur lit les
# pilotes de Contents/Resources/vulkan/icd.d du paquet.
cp "$BREW/lib/libvulkan.1.dylib" "$APP/Contents/Frameworks/"
cp "$BREW/lib/libMoltenVK.dylib" "$APP/Contents/Frameworks/"
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
  -s "$AJP_DEPS_PREFIX/lib" -s "$AJP_DEPS_PREFIX/onnxruntime/lib" -s "$BREW/lib" \
  -x "$APP/Contents/MacOS/ajp" -x "$APP/Contents/PlugIns/ajp_backend_vulkan.so" \
  -x "$APP/Contents/PlugIns/ajp_backend_coreml.so" \
  -x "$APP/Contents/Frameworks/libvulkan.1.dylib" -x "$APP/Contents/Frameworks/libMoltenVK.dylib"
# Le module du backend vit dans PlugIns : ses dépendances se résolvent depuis l'exécutable.
install_name_tool -add_rpath @executable_path/../Frameworks "$APP/Contents/MacOS/ajp" 2>/dev/null || true

VERSION="$(sed -n 's/^  VERSION \([0-9.]*\)$/\1/p' "$ROOT/CMakeLists.txt")"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>AJP</string>
  <key>CFBundleDisplayName</key><string>Ultimate Anime Video Player</string>
  <key>CFBundleIdentifier</key><string>io.github.ta2b.ajp</string>
  <key>CFBundleExecutable</key><string>ajp</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>${VERSION:-0.1.0}</string>
  <key>CFBundleVersion</key><string>${VERSION:-0.1.0}</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>CFBundleDocumentTypes</key>
  <array>
    <dict>
      <key>CFBundleTypeRole</key><string>Viewer</string>
      <key>LSItemContentTypes</key>
      <array><string>public.movie</string><string>org.matroska.mkv</string></array>
    </dict>
  </array>
</dict>
</plist>
PLIST

# Signature ad hoc : Gatekeeper demande encore une ouverture par clic droit (pas de notarisation).
codesign --force --deep --sign - "$APP"
(cd "$OUT" && rm -f ajp-macos-arm64.zip && ditto -c -k --keepParent AJP.app ajp-macos-arm64.zip \
  && shasum -a 256 ajp-macos-arm64.zip > ajp-macos-arm64.zip.sha256)
echo "Paquet : $OUT/ajp-macos-arm64.zip"
