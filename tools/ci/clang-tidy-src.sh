#!/usr/bin/env bash
# clang-tidy sur les fichiers de src/ que compile la configuration donnée (son
# compile_commands.json). Les backends dont le SDK manque à cette configuration (CUDA, OpenVINO,
# MIGraphX, Core ML) et les sources d'un autre système ne s'analysent pas sans leurs en-têtes :
# ils sont nommés en tête de sortie au lieu d'échouer sur « file not found ».
# Usage : tools/ci/clang-tidy-src.sh <dossier de build> [exécutable clang-tidy]
set -euo pipefail

build=${1:?dossier de build attendu}
tidy=${2:-clang-tidy-20}
root=$(git rev-parse --show-toplevel)
cd "$root"
[[ -f "$build/compile_commands.json" ]] || { echo "$build/compile_commands.json absent" >&2; exit 2; }

list=$(mktemp)
trap 'rm -f "$list"' EXIT
git ls-files -z -- 'src/*.cpp' | python3 -c '
import json, os, sys
root, db_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
compiled = {os.path.relpath(os.path.join(e["directory"], e["file"]), root)
            for e in json.load(open(db_path))}
tracked = [f for f in sys.stdin.read().split("\0") if f]
skipped = [f for f in tracked if f not in compiled]
with open(out, "wb") as o:
    o.write(b"".join(f.encode() + b"\0" for f in tracked if f in compiled))
print(f"clang-tidy : {len(tracked) - len(skipped)} fichiers, {len(skipped)} hors de cette configuration")
for f in skipped:
    print(f"  ignoré : {f}")
' "$root" "$build/compile_commands.json" "$list"

xargs -0 -r -n1 -P"$(nproc)" "$tidy" -p "$build" --quiet < "$list"
