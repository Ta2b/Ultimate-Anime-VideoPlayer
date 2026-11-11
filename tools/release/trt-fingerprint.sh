#!/usr/bin/env bash
# L'empreinte des entrées du paquet TensorRT (notre module + trtexec) : douze caractères qui ne changent que si ce
# qui entre dans le module change. Le paquet se joint alors à une release fixe, « trt-<empreinte> » pour Linux
# (tools/release/trt-package.sh, .github/workflows/trt-package.yml), « trt-win-<empreinte> » pour Windows
# (release-windows.yml), que les releases du lecteur lisent pour TOUTES les étiquettes : le paquet ne se reconstruit
# que quand l'empreinte change, et la CI de release le construit elle-même s'il manque.
#
# Entrées communes : le code que le module lie (src/core, gpu, kernels, prepare), CUTLASS (noyaux SPAN FP8), les
# définitions CMake (cmake/, le CMakeLists.txt racine sans sa ligne VERSION, qui change à chaque release sans
# toucher le module, et CMakePresets.json, qui porte les options du préréglage de livraison) et versions.env.
# Linux : la recette des dépendances, l'extraction de trtexec et des en-têtes TensorRT (linux-tensorrt.sh), le workflow
# qui construit le paquet (architectures passées à CMake), l'image du préflight, l'empaquetage et l'index des fichiers
# NVIDIA (B51-09). Windows :
# vcpkg, l'installation de CUDA et TensorRT, l'empaquetage et son index NVIDIA (A28-18). Conservateur : une
# modification de src/core refait le paquet ; src/app, src/audio, src/render, la documentation et les tests ne le
# refont pas.
#
#   tools/release/trt-fingerprint.sh [--windows] [dépôt]      # défaut : Linux, le dépôt de ce script
set -euo pipefail

platform=linux
[ "${1:-}" = "--windows" ] && { platform=windows; shift; }
root="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
paths=(src/core src/gpu src/kernels src/prepare third_party/cutlass cmake CMakeLists.txt CMakePresets.json
  tools/release/versions.env)
if [ "$platform" = linux ]; then
  paths+=(tools/deps/linux-from-source.sh tools/deps/linux-tensorrt.sh tools/release/package-linux.sh
    tools/release/preflight/Dockerfile tools/release/nvidia-vendor/tensorrt-linux-x64.lock
    .github/workflows/trt-package.yml)
else
  paths+=(vcpkg.json ports tools/deps/windows-tensorrt.ps1 tools/release/package-windows.ps1
    tools/release/nvidia-vendor/tensorrt-windows-x64.lock)
fi
{
  git ls-files -- "${paths[@]}" | LC_ALL=C sort | while IFS= read -r file; do
    # Fins de ligne ramenées à LF : l'empreinte est la même quel que soit le poste qui l'a extraite.
    if [ "$file" = CMakeLists.txt ]; then
      sed '/^  VERSION [0-9.]*$/d' "$file" | tr -d '\r' | sha256sum
    else
      tr -d '\r' < "$file" | sha256sum
    fi | cut -d' ' -f1 | sed "s|\$|  $file|"
  done
} | sha256sum | cut -c1-12
