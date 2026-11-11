#!/usr/bin/env bash
# Vérifie les fichiers d'une release du lecteur avant de la publier (job « publier » de release.yml) :
#   - les archives de Linux et de Windows sont là (celle de macOS, hors cible, est facultative : B51-11) ;
#   - pour une release finale, l'engines.lock de Linux et celui du zip Windows nomment chacun un moteur tensorrt de
#     leur plateforme (décision de Tatoub : chaque release a un paquet TensorRT Windows et Linux, A28-01) ;
#   - chaque lien de ces index vise le dépôt des releases ou les archives de NVIDIA (A28-05) ;
#   - chaque moteur TensorRT d'une release finale a ses [[vendor_file]] (sans eux, il s'installerait sans les
#     bibliothèques de NVIDIA) et son archive existe vraiment, avec la taille et le SHA-256 que l'index annonce : un
#     paquet trt-* réécrit après coup n'est plus signé « latest » sans qu'on le voie (B51-02, B51-16).
#     AJP_CHECK_ARCHIVES=0 saute le téléchargement des archives (essais hors ligne).
#
#   AJP_RELEASE_REPO=… tools/release/check-release.sh <étiquette> <dossier des fichiers téléchargés>
set -euo pipefail

[ $# -eq 2 ] || { sed -n '2,13p' "$0"; exit 2; }
tag=$1 dir=$2
root="$(cd "$(dirname "$0")/../.." && pwd)"
repo="${AJP_RELEASE_REPO:-$(sed -n 's/^AJP_RELEASE_REPO=//p' "$root/tools/release/versions.env")}"
final=true
[[ "$tag" == *-* ]] && final=false
fail=0
error() { echo "::error::$*" >&2; fail=1; }

for f in ajp-linux-x64.tar.gz ajp-windows-x64.zip engines.lock; do
  [ -s "$dir/$f" ] || error "$tag : $f absent de la release"
done
[ -s "$dir/ajp-macos-arm64.zip" ] || echo "::warning::$tag : pas d'archive macOS (hors cible) : publiée sans elle"
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
# Les archives des moteurs TensorRT d'un index : adresse, taille et SHA-256 relus sur le serveur.
check_archives() {  # index plateforme
  [ "${AJP_CHECK_ARCHIVES:-1}" = 0 ] && return 0
  local line url sha size tmp
  while IFS='|' read -r url sha size; do
    tmp="$(mktemp)"
    if ! curl -fsSL --proto =https --retry 5 --retry-delay 5 -o "$tmp" "$url"; then
      error "$2 : archive introuvable : $url"
    elif [ "$(wc -c < "$tmp" | tr -d ' ')" != "$size" ] || [ "$(sha256sum "$tmp" | cut -d' ' -f1)" != "$sha" ]; then
      error "$2 : $url n'a pas la taille ou le SHA-256 de l'index (paquet réécrit après coup ?)"
    else
      echo "$2 : archive vérifiée ($url)"
    fi
    rm -f "$tmp"
  done < <(awk -v want="$2" '
    function flush() { if (inside && id == "tensorrt" && platform == want) print url "|" sha "|" size; id = platform = url = sha = size = "" }
    /^\[\[/ { flush(); inside = ($0 == "[[engine]]"); next }
    /^\[/ { flush(); inside = 0; next }
    inside && /^id = / { id = $3; gsub(/"/, "", id) }
    inside && /^platform = / { platform = $3; gsub(/"/, "", platform) }
    inside && /^url = / { url = $3; gsub(/"/, "", url) }
    inside && /^sha256 = / { sha = $3; gsub(/"/, "", sha) }
    inside && /^size = / { size = $3 }
    END { flush() }' "$1")
}
for pair in "engines.lock:linux" "engines-windows.lock:windows"; do
  lock="$dir/${pair%%:*}" platform="${pair#*:}"
  tr -d '\r' < "$lock" > "$lock.lf"
  bad="$(sed -n 's/^url = "\(.*\)"$/\1/p' "$lock.lf" \
    | grep -v "^https://github.com/$repo/releases/download/" | grep -v '^https://pypi\.nvidia\.com/' || true)"
  [ -z "$bad" ] || error "$platform : liens hors du dépôt des releases : $bad"
  if has_tensorrt "$lock.lf" "$platform"; then
    echo "$platform : paquet TensorRT présent"
    # Des fichiers de NVIDIA à tirer pour cette plateforme (B51-16).
    if ! awk -v want="$platform" '/^\[\[/ { v = ($0 == "[[vendor_file]]"); p = 0 } v && $0 == "platform = \"" want "\"" { found = 1 } END { exit !found }' "$lock.lf"; then
      if [ "$final" = true ]; then error "$platform : aucun [[vendor_file]] : le moteur s'installerait sans les bibliothèques de NVIDIA"
      else echo "::warning::$platform : aucun [[vendor_file]]"; fi
    fi
    check_archives "$lock.lf" "$platform"
  elif [ "$final" = true ]; then
    error "$platform : aucun paquet TensorRT dans l'index d'une release finale"
  else
    echo "::warning::$platform : préversion sans paquet TensorRT"
  fi
done
[ "$fail" -eq 0 ] || exit 1
echo "$tag : release complète"
