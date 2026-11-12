#!/usr/bin/env bash
# clang-tidy sur les fichiers de src/ que compile la configuration donnée (son
# compile_commands.json). Les backends dont le SDK manque à cette configuration (CUDA,
# Core ML) et les sources d'un autre système ne s'analysent pas sans leurs en-têtes :
# ils sont nommés en tête de sortie au lieu d'échouer sur « file not found ».
# AJP_TIDY_BASE (une référence git, « origin/main ») : n'analyse que les fichiers de src/ changés
# depuis son point de départ commun avec HEAD, et ceux qui incluent (même par un autre en-tête) un en-tête
# changé ; tout src/ si .clang-tidy, cmake/, CMakePresets.json ou le CMakeLists.txt racine change, ou si la
# différence ne se calcule pas. Les en-têtes se suivent par leur #include "…", relatif à src/ ou au dossier
# de l'incluant : un en-tête de src/core changé n'envoyait plus tout src/ (13 min) à l'analyse. La nuit
# analyse tout, ce qu'une macro ou un chemin d'inclusion inhabituel aurait caché.
# AJP_TIDY_ONLY (une expression régulière sur le chemin, « ^src/gpu/cuda/ ») : seulement ces fichiers. Les tâches
# TensorRT et macOS (P5.8) n'analysent que ce que la tâche Linux ne compile pas.
# Usage : tools/ci/clang-tidy-src.sh <dossier de build> [exécutable clang-tidy]
set -euo pipefail

build=${1:?dossier de build attendu}
tidy=${2:-clang-tidy-20}
root=$(git rev-parse --show-toplevel)
cd "$root"
[[ -f "$build/compile_commands.json" ]] || { echo "$build/compile_commands.json absent" >&2; exit 2; }

list=$(mktemp)
only=$(mktemp)
trap 'rm -f "$list" "$only"' EXIT
if [[ -n "${AJP_TIDY_BASE:-}" ]] && git diff --name-only --diff-filter=d "$AJP_TIDY_BASE...HEAD" > "$only" 2>/dev/null; then
  if grep -qE '^\.clang-tidy$|^CMakeLists\.txt$|^cmake/|^CMakePresets\.json$' "$only"; then
    echo "clang-tidy : règles ou CMake racine changés, tout src/"
    : > "$only"
  else
    git ls-files -z -- 'src/*.h' 'src/*.hpp' 'src/*.inc' 'src/*.cuh' 'src/*.cpp' \
      | python3 "$root/tools/ci/tidy-includers.py" "$root" "$only" > "$only.cpp"
    mv "$only.cpp" "$only"
    echo "clang-tidy : $(wc -l < "$only") fichiers changés ou qui incluent un en-tête changé depuis $AJP_TIDY_BASE"
    [[ -s "$only" ]] || { echo "clang-tidy : rien à analyser"; exit 0; }
  fi
else
  # La différence ne se calcule pas (clone superficiel, référence absente) : tout src/, et on le dit (B28-05).
  [[ -z "${AJP_TIDY_BASE:-}" ]] || echo "clang-tidy : ${AJP_TIDY_BASE}...HEAD introuvable (clone superficiel ?), tout src/" >&2
  : > "$only"
fi
git ls-files -z -- 'src/*.cpp' | python3 -c '
import json, os, sys
root, db_path, out, only_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
compiled = {os.path.relpath(os.path.join(e["directory"], e["file"]), root)
            for e in json.load(open(db_path))}
tracked = [f for f in sys.stdin.read().split("\0") if f]
only = {l.strip() for l in open(only_path) if l.strip()}
if only:
    tracked = [f for f in tracked if f in only]
import re
pattern = os.environ.get("AJP_TIDY_ONLY", "")
if pattern:
    tracked = [f for f in tracked if re.search(pattern, f)]
skipped = [f for f in tracked if f not in compiled]
with open(out, "wb") as o:
    o.write(b"".join(f.encode() + b"\0" for f in tracked if f in compiled))
print(f"clang-tidy : {len(tracked) - len(skipped)} fichiers, {len(skipped)} hors de cette configuration")
for f in skipped:
    print(f"  ignoré : {f}")
' "$root" "$build/compile_commands.json" "$list" "$only"

# Un clang-tidy de Homebrew ne connaît pas le SDK de Xcode : sans lui, <cstddef> manque et chaque
# déclaration est lue de travers (fonctions prises pour des variables).
extra=()
if [[ "$(uname -s)" == Darwin ]]; then
  extra=(--extra-arg="-isysroot$(xcrun --show-sdk-path)")
fi
xargs -0 -r -n1 -P"$(getconf _NPROCESSORS_ONLN)" "$tidy" -p "$build" --quiet ${extra[@]+"${extra[@]}"} < "$list"
