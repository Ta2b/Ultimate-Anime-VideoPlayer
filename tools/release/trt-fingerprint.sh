#!/usr/bin/env bash
# L'empreinte des entrées du paquet TensorRT Linux (notre module + trtexec) : douze caractères qui ne changent que si
# ce qui entre dans le module change. Le paquet se joint alors à une release fixe « trt-<empreinte> » (tools/release/
# trt-package.sh) que release-linux.yml lit pour TOUTES les étiquettes : plus de paquet à reconstruire, ni de Release
# Linux à relancer, tant que l'empreinte est la même.
#
# Entrées : le code que le module lie (src/core, gpu, kernels, prepare, vendor_hdr), les définitions CMake (cmake/), la
# recette des dépendances et de l'image (tools/deps/linux-from-source.sh, versions.env), l'empaquetage
# (package-linux.sh) et l'index des fichiers NVIDIA. Le CMakeLists.txt racine n'y est pas : sa VERSION change à
# chaque release sans toucher le module. Conservateur : une modification de src/core refait le paquet ; src/app,
# src/audio, src/render, la documentation et les tests ne le refont pas.
#
#   tools/release/trt-fingerprint.sh [dépôt]      # défaut : le dépôt de ce script
set -euo pipefail

root="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$root"
paths=(src/core src/gpu src/kernels src/prepare src/vendor_hdr cmake
  tools/deps/linux-from-source.sh tools/release/versions.env tools/release/package-linux.sh
  tools/release/nvidia-vendor/tensorrt-linux-x64.lock)
{
  git ls-files -- "${paths[@]}" | LC_ALL=C sort | while IFS= read -r file; do
    printf '%s  %s\n' "$(sha256sum < "$file" | cut -d' ' -f1)" "$file"
  done
} | sha256sum | cut -c1-12
