#!/usr/bin/env bash
# Chaque module SPIR-V du lecteur (noyaux et réseaux du backend Vulkan) traduit en Metal comme
# MoltenVK le fait (SPIRV-Cross, sous « using namespace metal; »), puis compilé par le compilateur
# Metal d'Apple. spirv-val ne voit pas ce qui ne passe qu'en Metal : une fonction GLSL nommée
# comme un type de Metal (mesh, metal::mesh) rend son appel ambigu, et MoltenVK refuse alors le
# pipeline (VK_ERROR_INITIALIZATION_FAILED). macOS seulement ; spirv-cross et spirv-tools de Homebrew.
#
# Les modules à matrices coopératives sont écartés, nommés : MoltenVK n'offre pas
# VK_KHR_cooperative_matrix, et net_lower_conv.cpp ne les choisit que sur un appareil qui l'offre.
#   tools/ci/check-msl.sh build/macos-release
set -euo pipefail
build="${1:?dossier de construction}"
command -v spirv-cross >/dev/null || brew install --quiet spirv-cross
command -v spirv-dis >/dev/null || brew install --quiet spirv-tools
# Xcode 26 livre le compilateur Metal à part.
xcrun -sdk macosx metal --version >/dev/null 2>&1 || xcodebuild -downloadComponent MetalToolchain
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

status=0
count=0
while IFS= read -r -d '' spv; do
  name="$(basename "$spv" .spv)"
  spirv-dis -o "$work/$name.spvasm" "$spv"
  if grep -q 'OpCapability CooperativeMatrixKHR' "$work/$name.spvasm"; then
    echo "$name : matrices coopératives, absentes sous MoltenVK : écarté"
    continue
  fi
  if ! spirv-cross --msl --msl-version 30100 --output "$work/$name.metal" "$spv" 2>"$work/$name.log"; then
    echo "::error::$name : SPIRV-Cross ne le traduit pas en Metal"
    cat "$work/$name.log"
    status=1
    continue
  fi
  if ! xcrun -sdk macosx metal -std=metal3.1 -c "$work/$name.metal" -o "$work/$name.air" 2>"$work/$name.log"; then
    echo "::error::$name : le compilateur Metal le refuse"
    cat "$work/$name.log"
    status=1
    continue
  fi
  count=$((count + 1))
done < <(find "$build/src/gpu/vulkan" -name '*.spv' -print0)
if [[ $count -eq 0 && $status -eq 0 ]]; then
  echo "::error::aucun module SPIR-V sous $build/src/gpu/vulkan"
  exit 1
fi
echo "$count modules SPIR-V traduits en Metal et compilés"
exit $status
