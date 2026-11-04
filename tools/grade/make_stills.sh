#!/bin/bash
# Les extraits des 12 plans d'épisodes de la porte du grade de l'anneau (chantier 2.11) : pour chaque ligne
# « nom, épisode, seconde » de EPISODES, 8 s de l'épisode à partir de 2 s avant le plan, copiées sans
# réencodage (ffmpeg -c copy, coupe sur l'image clé d'avant : le plan est à 2 s environ dans l'extrait) vers
# DOSSIER/nom.mkv. compare_shots.sh les lit par STILLS=DOSSIER.
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
  from=$((start > 2 ? start - 2 : 0))
  ffmpeg -nostdin -loglevel error -y -ss "$from" -i "$file" -t 8 -map 0:v:0 -c copy "$out/$name.mkv"
  echo "$name : $out/$name.mkv ($file à ${from} s)"
  count=$((count + 1))
done < "$episodes"
echo "$count extraits dans $out"
