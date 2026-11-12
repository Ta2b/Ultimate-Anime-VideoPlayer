#!/bin/bash
# La porte du grade de l'anneau (chantier 2.11) : chaque plan de shots30.tsv capturé deux fois en PQ 16 bits
# (sortie HDR10 réelle), avec les options A puis B, puis comparé en ΔE_ITP (de_itp.py).
#   compare_shots.sh "A" "B" SORTIE        par ex. compare_shots.sh "" "--grade-ring" ~/ajp-reports/porte
# Environnement : L (le lanceur, lancer.sh), CAL (calibration.mkv), STILLS (les extraits de make_stills.sh),
# BASE (dossier de réglages copié pour chaque capture, grade et looks de l'utilisateur), WINDOW (taille des
# instantanés, défaut 1920x1080 ; la taille de l'anneau pour « à taille égale », 3840x2160 pour la 4K),
# SNAP_AT (seconde de la capture après le début du plan, 4 : le grade adaptatif a une constante de temps ; la
# capture reste dans le plan nommé, SNAP_AT est ramené à la durée du plan moins 1 s, C46-01), SHOTS
# (défaut : le shots30.tsv à côté). Un plan dont l'extrait manque est sauté et dit.
# Critères (docs/adr/0009-grade-dans-l-anneau.md) : par plan p95 ≤ MAX_P95 (1, 2,5 en 4K) et
# p99,9 ≤ MAX_P999 (3) ; code retour 1 si un plan les dépasse, si une capture manque, si la mesure échoue ou
# rend un verdict vide (B46-01). WAIT_MAX (secondes, défaut 600) borne l'attente d'une autre instance d'ajp.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
a_opts=${1?options A attendues (chaîne vide possible)}
b_opts=${2?options B attendues}
out=${3:?dossier de sortie attendu}
L=${L:-$HOME/ajp-test/lancer.sh}
CAL=${CAL:-$HOME/ajp-test/ajp-linux-x64/calibration/calibration.mkv}
STILLS=${STILLS:-$HOME/ajp-test/stills}
BASE=${BASE:-}
WINDOW=${WINDOW:-1920x1080}
SNAP_AT=${SNAP_AT:-4}
SHOTS=${SHOTS:-$here/shots30.tsv}
if [[ "$WINDOW" == 3840x2160 ]]; then MAX_P95=${MAX_P95:-2.5}; else MAX_P95=${MAX_P95:-1}; fi
MAX_P999=${MAX_P999:-3}
WAIT_MAX=${WAIT_MAX:-600}
mkdir -p "$out/frames"
summary="$out/resume.tsv"
printf 'plan\tgenre\tmoyenne\tp95\tp99\tp99.9\tmax\tverdict\n' > "$summary"

# Durée du plan (s) : jusqu'au plan suivant du même fichier de SHOTS, sinon jusqu'à la fin du fichier (ffprobe).
plan_duration() {  # plan_duration FICHIER DÉBUT SOURCE
  local next total
  next=$(awk -F'\t' -v src="$3" -v t="$2" '$0 !~ /^#/ && $2 == src && $3 + 0 > t + 0 { if (m == "" || $3 + 0 < m) m = $3 + 0 } END { print m }' "$SHOTS")
  if [[ -n "$next" ]]; then
    echo $((next - $2))
    return
  fi
  total=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$1" 2> /dev/null | cut -d. -f1)
  [[ "$total" =~ ^[0-9]+$ ]] && echo $((total - $2)) || echo 0
}

capture() {  # capture NOM FICHIER DÉBUT OPTIONS SORTIE
  local d snap=$SNAP_AT
  d=$(mktemp -d)
  # La capture reste dans le plan : au plus (durée − 1) s après son début (0 si la durée est inconnue : on garde SNAP_AT).
  local dur
  dur=$(plan_duration "$2" "$3" "$source")
  if ((dur > 1 && snap > dur - 1)); then snap=$((dur - 1)); fi
  [[ -n "$BASE" ]] && cp "$BASE"/*.toml "$d/" 2>/dev/null
  local waited=0
  while pgrep -x ajp > /dev/null; do  # une instance à la fois
    if ((waited >= WAIT_MAX)); then
      rm -rf "$d"
      echo "$1 : une autre instance d'ajp tourne encore après ${WAIT_MAX} s, capture abandonnée" >&2
      return 1
    fi
    sleep 10
    waited=$((waited + 10))
  done
  # shellcheck disable=SC2086
  timeout 180 "$L" "$2" --start="$3" --no-resume --mute --window="$WINDOW" --data-dir="$d" \
    --snapshot="$5" --snapshot-at="$snap" --snapshot-frames=1 --seconds=$((snap + 3)) $4 \
    > "${5%.pq.ppm}.log" 2>&1
  local rc=$?
  rm -rf "$d"
  return $rc
}

failed=0
while IFS=$'\t' read -r name source start kind; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  case $source in
    cal) file=$CAL ;;
    ep:*) file="$STILLS/${source#ep:}.mkv" ;;
    *) file=$source ;;
  esac
  if [[ ! -f "$file" ]]; then
    echo "$name : $file absent, sauté"
    printf '%s\t%s\t\t\t\t\t\tsauté\n' "$name" "$kind" >> "$summary"
    continue
  fi
  fa="$out/frames/$name-A.pq.ppm"
  fb="$out/frames/$name-B.pq.ppm"
  capture "$name" "$file" "$start" "$a_opts" "$fa" || echo "$name : capture A en échec (voir ${fa%.pq.ppm}.log)"
  capture "$name" "$file" "$start" "$b_opts" "$fb" || echo "$name : capture B en échec (voir ${fb%.pq.ppm}.log)"
  if [[ ! -f "$fa" || ! -f "$fb" ]]; then
    printf '%s\t%s\t\t\t\t\t\tpas de capture\n' "$name" "$kind" >> "$summary"
    failed=1
    continue
  fi
  verdict=
  if line=$(python3 "$here/de_itp.py" "$fa" "$fb" --heatmap "$out/frames/$name-carte.ppm" --json); then
    verdict=$(python3 - "$line" "$MAX_P95" "$MAX_P999" <<'PY'
import json, sys
s = json.loads(sys.argv[1])
ok = s["p95"] <= float(sys.argv[2]) and s["p99.9"] <= float(sys.argv[3])
print("\t".join(f"{s[k]:.3f}" for k in ("moyenne", "p95", "p99", "p99.9", "max")) + ("\tpasse" if ok else "\tÉCHEC"))
PY
)
  fi
  if [[ "$verdict" != *passe && "$verdict" != *ÉCHEC ]]; then
    # La mesure n'a rien rendu (PPM tronqué, de_itp.py en erreur) : un plan sans verdict est un échec.
    echo "$name : mesure en échec, pas de verdict"
    printf '%s\t%s\t\t\t\t\t\tmesure en échec\n' "$name" "$kind" >> "$summary"
    failed=1
    continue
  fi
  printf '%s\t%s\t%s\n' "$name" "$kind" "$verdict" >> "$summary"
  echo "$name ($kind) : $verdict"
  [[ "$verdict" == *ÉCHEC ]] && failed=1
done < "$SHOTS"
echo "résumé : $summary (p95 ≤ $MAX_P95, p99,9 ≤ $MAX_P999 par plan)"
exit $failed
