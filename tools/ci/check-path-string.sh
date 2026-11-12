#!/usr/bin/env bash
# path::string() (P5.12) : sous Windows, la page de code ANSI, et une exception devant un nom en japonais.
# Un chemin s'écrit par core::to_utf8 et se lit par core::path_from_utf8 (src/core/utf8_path.h). Les
# appels qui restent dans src/ ne peuvent que diminuer : baisser kMax quand on en retire.
set -euo pipefail
cd "$(dirname "$0")/../.."
# Les appels, pas les lignes (deux sur une ligne comptent deux), et generic_string() avec string() : sous Windows, les
# deux passent par la page de code ANSI (C28-09). native() reste libre : il rend le wstring que prepare/utf8.cpp
# convertit lui-même.
readonly kMax=42
readonly pattern='\.(generic_)?string\(\)'
count="$({ git grep -h -o -E "$pattern" -- 'src/*.cpp' 'src/*.h' || true; } | wc -l)"
if ((count > kMax)); then
  echo "path::string() : $count appels dans src/, $kMax au plus ; écrire core::to_utf8(chemin) (src/core/utf8_path.h)" >&2
  git grep -n -E "$pattern" -- 'src/*.cpp' 'src/*.h' | tail -n "$((count - kMax))" >&2
  exit 1
fi
echo "path::string() : $count appels (au plus $kMax)"
