#!/usr/bin/env bash
# Un fichier vise moins de 500 lignes ; au-delà de 800, la CI échoue. Seuls les fichiers de code et de script
# comptent : un modèle ONNX, une LUT, une image ou un corpus de fuzz en octets (des 0x0A en nombre) n'ont pas de
# lignes. Les .txt, .md, .tsv et .toml (données, notes) en sont aussi.
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
done < <(git ls-files -z -- 'src/*.cpp' 'src/*.h' 'src/*.cu' 'src/*.cuh' 'src/*.mm' 'src/*.hook' 'src/*.comp' \
  'src/*.glsl' 'src/*.py' 'src/*.sh' 'src/*.cmake' 'src/*.ps1' \
  'tests/*.cpp' 'tests/*.h' 'tests/*.cu' 'tests/*.cuh' 'tests/*.mm' 'tests/*.py' 'tests/*.sh' 'tests/*.cmake' \
  'tools/*.cpp' 'tools/*.h' 'tools/*.cu' 'tools/*.cuh' 'tools/*.mm' 'tools/*.hook' 'tools/*.comp' 'tools/*.glsl' \
  'tools/*.py' 'tools/*.sh' 'tools/*.cmake' 'tools/*.ps1')
exit $status
