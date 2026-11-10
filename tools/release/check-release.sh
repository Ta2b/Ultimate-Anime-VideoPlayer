#!/usr/bin/env bash
# Vérifie les fichiers d'une release du lecteur avant de la publier (job « publier » de release.yml) :
#   - les archives des trois plateformes sont là ;
#   - pour une release finale, l'engines.lock de Linux et celui du zip Windows nomment chacun un moteur tensorrt de
#     leur plateforme (décision de Tatoub : chaque release a un paquet TensorRT Windows et Linux, A28-01) ;
#   - chaque lien de ces index vise le dépôt des releases ou les archives de NVIDIA (A28-05).
#
#   AJP_RELEASE_REPO=… tools/release/check-release.sh <étiquette> <dossier des fichiers téléchargés>
set -euo pipefail

[ $# -eq 2 ] || { sed -n '2,8p' "$0"; exit 2; }
tag=$1 dir=$2
root="$(cd "$(dirname "$0")/../.." && pwd)"
repo="${AJP_RELEASE_REPO:-$(sed -n 's/^AJP_RELEASE_REPO=//p' "$root/tools/release/versions.env")}"
final=true
[[ "$tag" == *-* ]] && final=false
fail=0
error() { echo "::error::$*" >&2; fail=1; }

for f in ajp-linux-x64.tar.gz ajp-windows-x64.zip ajp-macos-arm64.zip engines.lock; do
  [ -s "$dir/$f" ] || error "$tag : $f absent de la release"
done
[ "$fail" -eq 0 ] || exit 1

# L'engines.lock livré dans le zip Windows (ajp/engines.lock ; séparateur / ou \ selon Compress-Archive).
python3 -I - "$dir/ajp-windows-x64.zip" "$dir/engines-windows.lock" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    names = [n for n in z.namelist() if n.replace("\\", "/").endswith("engines.lock")]
    if not names:
        sys.exit("ajp-windows-x64.zip sans engines.lock")
    open(sys.argv[2], "wb").write(z.read(names[0]))
PY

has_tensorrt() {  # index plateforme
  awk -v want="$2" '
    /^\[\[/ { e = ($0 == "[[engine]]"); t = 0; p = 0 }
    e && $0 == "id = \"tensorrt\"" { t = 1 }
    e && $0 == "platform = \"" want "\"" { p = 1 }
    t && p { found = 1 }
    END { exit !found }' "$1"
}
for pair in "engines.lock:linux" "engines-windows.lock:windows"; do
  lock="$dir/${pair%%:*}" platform="${pair#*:}"
  tr -d '\r' < "$lock" > "$lock.lf"
  bad="$(sed -n 's/^url = "\(.*\)"$/\1/p' "$lock.lf" \
    | grep -v "^https://github.com/$repo/releases/download/" | grep -v '^https://pypi\.nvidia\.com/' || true)"
  [ -z "$bad" ] || error "$platform : liens hors du dépôt des releases : $bad"
  if has_tensorrt "$lock.lf" "$platform"; then
    echo "$platform : paquet TensorRT présent"
  elif [ "$final" = true ]; then
    error "$platform : aucun paquet TensorRT dans l'index d'une release finale"
  else
    echo "::warning::$platform : préversion sans paquet TensorRT"
  fi
done
[ "$fail" -eq 0 ] || exit 1
echo "$tag : release complète"
