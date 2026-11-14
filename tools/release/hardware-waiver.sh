#!/usr/bin/env bash
# Dérogation écrite aux rapports des tests matériels d'une étiquette finale (release-guard.yml, C28-02). Elle se donne
# dans tools/release/hardware-waivers.txt, une ligne par étiquette : « <étiquette> <préversion de référence> <raison> ».
# Elle ne vaut que si rien de ce que les tests matériels jugent n'a bougé depuis la préversion de référence :
# l'empreinte TensorRT (tools/release/trt-fingerprint.sh, Linux et Windows) doit être la même aux deux commits.
#
#   tools/release/hardware-waiver.sh <étiquette> [dépôt]
#   0 : dérogation acceptée (la raison est affichée) ; 3 : aucune dérogation pour cette étiquette ; 1 : refusée.
set -euo pipefail

[ $# -ge 1 ] || { sed -n '2,8p' "$0"; exit 2; }
tag=$1
root="${2:-$(cd "$(dirname "$0")/../.." && pwd)}"
here="$(cd "$(dirname "$0")" && pwd)"
waivers="$root/tools/release/hardware-waivers.txt"
[ -f "$waivers" ] || exit 3
line="$(grep -E "^${tag//./\\.}[[:space:]]" "$waivers" | head -n 1 || true)"
[ -n "$line" ] || exit 3
read -r _ reference reason <<<"$line"
if [ -z "${reference:-}" ] || [ -z "${reason:-}" ]; then
  echo "::error::dérogation de $tag incomplète : « <étiquette> <préversion de référence> <raison> »" >&2
  exit 1
fi
if [[ "$reference" != *-* ]]; then
  echo "::error::dérogation de $tag : la référence $reference n'est pas une préversion" >&2
  exit 1
fi
if [ "${reference%%-*}" != "$tag" ]; then
  echo "::error::dérogation de $tag : la référence $reference est d'une autre version" >&2
  exit 1
fi
git -C "$root" rev-parse -q --verify "refs/tags/$reference^{commit}" >/dev/null \
  || { echo "::error::dérogation de $tag : l'étiquette $reference est introuvable" >&2; exit 1; }
old="$(mktemp -d)"
trap 'git -C "$root" worktree remove --force "$old" >/dev/null 2>&1 || rm -rf "$old"' EXIT
git -C "$root" worktree add -q --detach "$old" "$reference" >/dev/null 2>&1
for platform in linux windows; do
  flag=()
  [ "$platform" = windows ] && flag=(--windows)
  now="$("$here/trt-fingerprint.sh" "${flag[@]}" "$root")"
  then_="$("$here/trt-fingerprint.sh" "${flag[@]}" "$old")"
  if [ "$now" != "$then_" ]; then
    echo "::error::dérogation de $tag refusée : l'empreinte TensorRT $platform a changé depuis $reference ($then_ -> $now), les tests matériels sont à refaire" >&2
    exit 1
  fi
done
echo "::warning::$tag sans rapports des tests matériels, par dérogation : empreinte TensorRT identique à $reference. Raison : $reason"
