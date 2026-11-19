#!/usr/bin/env bash
# Paquets Ubuntu d'une tâche de la CI : « apt-get update » puis « apt-get install », chacun chronométré au journal.
#
#   tools/ci/apt-install.sh [options d'apt-get install] <paquet>…
#
# Les dépôts que l'image du runner ajoute (Microsoft, Azure CLI, etc.) ne servent à aucune tâche : ils sont retirés
# avant la mise à jour des index, qui n'attend plus qu'Ubuntu. Novembre 2026 : l'étape « Paquets système » passait
# de moins d'une minute à 4, 12 puis 35 minutes ; les deux temps séparés disent laquelle des deux attend.
set -euo pipefail

sudo find /etc/apt/sources.list.d -type f ! -name 'ubuntu.sources' -print -delete
retries=(-o Acquire::Retries=3 -o Acquire::http::Timeout=30)
start=$SECONDS
sudo apt-get "${retries[@]}" update -qq
echo "apt-get update : $((SECONDS - start)) s"
start=$SECONDS
sudo apt-get "${retries[@]}" install -y -qq "$@"
echo "apt-get install : $((SECONDS - start)) s"
