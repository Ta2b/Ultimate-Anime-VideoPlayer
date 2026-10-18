#!/usr/bin/env bash
# Un fichier vise moins de 500 lignes ; au-delà de 800, la CI échoue.
set -euo pipefail
status=0
while IFS= read -r -d '' file; do
  lines=$(wc -l < "$file")
  if (( lines > 800 )); then
    echo "::error file=$file::$lines lignes, au-delà de la limite de 800"
    status=1
  elif (( lines > 500 )); then
    echo "::warning file=$file::$lines lignes, au-delà des 500 visées"
  fi
done < <(git ls-files -z -- 'src/*' 'tests/*' 'tools/*' ':!:*.json' ':!:*.toml')
exit $status
