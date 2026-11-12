#!/usr/bin/env bash
# Construit dans un dossier de build Ninja les seuls fichiers générés que lit le compilateur (en-têtes, .inc,
# en-tête GLSL des noyaux, code de wayland-scanner), sans compiler ni lier le lecteur. clang-tidy n'a besoin que
# d'eux et de compile_commands.json : sa tâche de ci.yml construisait le lecteur entier (7 min) pour analyser
# quelques fichiers. Les petits outils qui produisent ces fichiers (ajp_kernels_glsl) se construisent avec eux.
# Usage : tools/ci/build-generated.sh <dossier de build>
set -euo pipefail

build=${1:?dossier de build attendu}
[[ -f "$build/build.ninja" ]] || { echo "$build/build.ninja absent (générateur Ninja attendu)" >&2; exit 2; }

# Les sorties de toutes les règles du graphe, sauf les objets et les liens : ce que les règles de commande
# (add_custom_command) produisent et que les sources incluent.
list=$(mktemp)
trap 'rm -f "$list"' EXIT
ninja -C "$build" -t targets all | cut -d: -f1 \
  | grep -E '\.(h|hpp|inc|glsl|c|cpp)$' | grep -vE '(^|/)CMakeFiles/' > "$list" || true
echo "fichiers générés : $(wc -l < "$list")"
[[ -s "$list" ]] || exit 0
xargs -r ninja -C "$build" < "$list"
