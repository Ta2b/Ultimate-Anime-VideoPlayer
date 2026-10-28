#!/usr/bin/env bash
# clang-tidy sur les fichiers de src/ que compile la configuration donnée (son
# compile_commands.json). Les backends dont le SDK manque à cette configuration (CUDA, OpenVINO,
# MIGraphX, Core ML) et les sources d'un autre système ne s'analysent pas sans leurs en-têtes :
# ils sont nommés en tête de sortie au lieu d'échouer sur « file not found ».
# AJP_TIDY_BASE (une référence git, « origin/main ») : n'analyse que les fichiers de src/ changés
# depuis son point de départ commun avec HEAD ; tout src/ si un en-tête ou un CMakeLists.txt
# change, ou si la différence ne se calcule pas.
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
  if grep -qE '^src/.*\.(h|hpp|inc|cuh)$|(^|/)CMakeLists\.txt$|^cmake/|^CMakePresets\.json$' "$only"; then
    echo "clang-tidy : en-tête ou CMake changé, tout src/"
    : > "$only"
  else
    grep -E '^src/.*\.cpp$' "$only" > "$only.cpp" || true
    mv "$only.cpp" "$only"
    echo "clang-tidy : $(wc -l < "$only") fichiers changés depuis $AJP_TIDY_BASE"
    [[ -s "$only" ]] || { echo "clang-tidy : rien à analyser"; exit 0; }
  fi
else
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
skipped = [f for f in tracked if f not in compiled]
with open(out, "wb") as o:
    o.write(b"".join(f.encode() + b"\0" for f in tracked if f in compiled))
print(f"clang-tidy : {len(tracked) - len(skipped)} fichiers, {len(skipped)} hors de cette configuration")
for f in skipped:
    print(f"  ignoré : {f}")
' "$root" "$build/compile_commands.json" "$list" "$only"

xargs -0 -r -n1 -P"$(nproc)" "$tidy" -p "$build" --quiet < "$list"
