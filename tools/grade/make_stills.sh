#!/bin/bash
# Les extraits des 12 plans d'épisodes de la porte du grade de l'anneau (chantier 2.11) : pour chaque ligne
# « nom, épisode, seconde » de EPISODES, 8 s de l'épisode à partir de 2 s avant le plan, vers DOSSIER/nom.mkv.
# Les extraits sont recodés sans perte (FFV1) : une copie (-c copy) couperait sur l'image clé d'avant et le plan ne
# serait plus à 2 s dans l'extrait (C46-02) ; le recodage sans perte garde chaque pixel et coupe à l'image près.
# compare_shots.sh les lit par STILLS=DOSSIER.
#   make_stills.sh episodes.tsv DOSSIER
set -euo pipefail
episodes=${1:?episodes.tsv attendu (voir episodes.exemple.tsv)}
out=${2:?dossier de sortie attendu}
mkdir -p "$out"
count=0
while IFS=$'\t' read -r name file start; do
  [[ -z "$name" || "$name" == \#* ]] && continue
  if [[ ! -f "$file" ]]; then
    echo "$name : $file introuvable" >&2
    exit 1
  fi
  if ((start < 2)); then
    echo "$name : le plan doit commencer à 2 s ou plus (il est à ${start} s), shots30.tsv le place à 2 s de l'extrait" >&2
    exit 1
  fi
  from=$((start - 2))
  ffmpeg -nostdin -loglevel error -y -ss "$from" -i "$file" -t 8 -map 0:v:0 -c:v ffv1 -level 3 "$out/$name.mkv"
  echo "$name : $out/$name.mkv ($file à ${from} s)"
  count=$((count + 1))
done < "$episodes"
echo "$count extraits dans $out"
