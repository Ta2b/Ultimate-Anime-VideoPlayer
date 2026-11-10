#!/usr/bin/env bash
# path::string() (P5.12) : sous Windows, la page de code ANSI, et une exception devant un nom en japonais.
# Un chemin s'écrit par core::to_utf8 et se lit par core::path_from_utf8 (src/core/utf8_path.h). Les
# appels qui restent dans src/ ne peuvent que diminuer : baisser kMax quand on en retire.
set -euo pipefail
cd "$(dirname "$0")/../.."
readonly kMax=170
count="$(git grep -h '\.string()' -- 'src/*.cpp' 'src/*.h' | wc -l)"
if ((count > kMax)); then
  echo "path::string() : $count lignes dans src/, $kMax au plus ; écrire core::to_utf8(chemin) (src/core/utf8_path.h)" >&2
  git grep -n '\.string()' -- 'src/*.cpp' 'src/*.h' | tail -n "$((count - kMax))" >&2
  exit 1
fi
echo "path::string() : $count lignes (au plus $kMax)"
