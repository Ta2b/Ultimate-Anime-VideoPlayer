#!/usr/bin/env bash
# Banc A/B du lecteur (chantier 0.8) : rejoue le même extrait en alternant deux variantes (A, B, A, B puis B, A, B, A...) × tours,
# un processus neuf à chaque passage, et résume avec banc_resume.py.
#
#   flock ~/kirari-reports/gpu-bench.lock tools/bench/banc.sh \
#     --a="OPTIONS" --b="OPTIONS" --extraits=extraits.tsv --sortie=DOSSIER [--tours=3] [--secondes=60] [--kirari=CHEMIN]
#
# extraits.tsv : « fichier<TAB>début en secondes » par ligne (tools/bench/extraits.exemple.tsv).
# Lancé sur le poste NVIDIA, GPU sans autre charge ; les horloges sont relevées (nvidia-smi), jamais forcées.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
A=""; B=""; EXTRAITS=""; OUT=""; ROUNDS=3; SECONDS_PLAYED=60; Kirari="${Kirari:-kirari}"
for arg in "$@"; do
  case "$arg" in
    --a=*) A="${arg#--a=}" ;;
    --b=*) B="${arg#--b=}" ;;
    --extraits=*) EXTRAITS="${arg#--extraits=}" ;;
    --sortie=*) OUT="${arg#--sortie=}" ;;
    --tours=*) ROUNDS="${arg#--tours=}" ;;
    --secondes=*) SECONDS_PLAYED="${arg#--secondes=}" ;;
    --kirari=*) Kirari="${arg#--kirari=}" ;;
    *) echo "option inconnue : $arg" >&2; exit 2 ;;
  esac
done
[ -n "$EXTRAITS" ] && [ -n "$OUT" ] || { echo "il faut --extraits et --sortie" >&2; exit 2; }
mkdir -p "$OUT"
{
  echo "date : $(date -Is)"
  echo "kirari : $("$Kirari" --version 2>&1 | head -1)"
  echo "noyau : $(uname -sr)"
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "télémétrie NVIDIA indisponible"
  fi
  echo "A : $A"
  echo "B : $B"
} >"$OUT/contexte.txt"

play() { # variante options fichier début étiquette
  local label="$1" options="$2" file="$3" start="$4" tag="$5"
  local base="$OUT/$tag"
  # shellcheck disable=SC2086
  "$Kirari" play "$file" --start="$start" --seconds="$SECONDS_PLAYED" --fullscreen --mute --no-resume \
    --data-dir="$OUT/data-$label" --stats="$base.tsv" $options </dev/null >"$base.log" 2>&1 || {
    echo "échec : $tag — voir $base.log" >&2
    return 1
  }
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,utilization.gpu --format=csv,noheader >"$base.smi" || true
  fi
}

n=0
passages=()
while IFS=$'\t' read -r file start; do
  [ -n "$file" ] && [[ "$file" != \#* ]] || continue
  n=$((n + 1))
  # préchauffage : un passage de 10 s par variante, non compté (moteurs, caches de plans)
  for label in A B; do
    options="$A"; [ "$label" = B ] && options="$B"
    SECONDS_PLAYED_SAVE=$SECONDS_PLAYED; SECONDS_PLAYED=10
    play "$label" "$options" "$file" "$start" "prechauffage-$n-$label"
    SECONDS_PLAYED=$SECONDS_PLAYED_SAVE
  done
  for round in $(seq 1 "$ROUNDS"); do
    # (A, B, A, B) aux tours impairs, (B, A, B, A) aux pairs : aucune variante n'est toujours première, une dérive
    # d'horloge ou thermique ne pèse pas toujours contre B (C46-05).
    order=(A B A B); ((round % 2)) || order=(B A B A)
    for label in "${order[@]}"; do
      options="$A"; [ "$label" = B ] && options="$B"
      tag="extrait$n-tour$round-$label-$(date +%s)-${#passages[@]}"
      play "$label" "$options" "$file" "$start" "$tag"
      passages+=("$OUT/$tag.tsv")
    done
  done
done <"$EXTRAITS"

(( ${#passages[@]} )) || { echo "aucun passage demandé" >&2; exit 2; }
python3 -I "$HERE/banc_resume.py" "$OUT" "${passages[@]}" | tee "$OUT/resume.txt"
