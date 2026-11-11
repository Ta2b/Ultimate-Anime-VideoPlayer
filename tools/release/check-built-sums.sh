#!/usr/bin/env bash
# Le job « publier » de release.yml signe les octets qu'il télécharge du dépôt des releases : ils doivent être ceux
# que les jobs de construction ont produits. Chaque job de plateforme sort le SHA-256 de son archive (sortie de job,
# jamais relue dans la release) ; ce script les compare à ce qui a été téléchargé, avant toute signature (B51-07).
# Une archive dont le SHA-256 attendu est vide n'est pas contrôlée si elle manque aussi dans le dossier (macOS,
# facultative) ; absente d'un côté seulement, c'est une erreur.
#
#   tools/release/check-built-sums.sh <dossier téléchargé> <nom de l'archive>=<sha256 construit>...
set -euo pipefail

[ $# -ge 2 ] || { sed -n '2,10p' "$0"; exit 2; }
dir=$1; shift
fail=0
for pair in "$@"; do
  name=${pair%%=*} want=${pair#*=}
  file="$dir/$name"
  if [ -z "$want" ]; then
    if [ -e "$file" ]; then
      echo "::error::$name est dans la release mais aucun job de construction n'en a donné l'empreinte" >&2
      fail=1
    else
      echo "$name : non construite, ignorée"
    fi
    continue
  fi
  [ -s "$file" ] || { echo "::error::$name absent de la release" >&2; fail=1; continue; }
  got="$(sha256sum "$file" | cut -d' ' -f1)"
  if [ "$got" != "$want" ]; then
    echo "::error::$name : le SHA-256 téléchargé ($got) n'est pas celui que le job de construction a produit ($want) : archive remplacée dans la release ?" >&2
    fail=1
  else
    echo "$name : identique à la construction"
  fi
done
exit "$fail"
