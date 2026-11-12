#!/bin/sh
# ABAB dans la même séance : A = chaîne fusionnée (graphe CUDA), B = réseau entier TensorRT FP16
# (trtexec --loadEngine, graphe CUDA, vraies entrées ; B' = le même moteur par le banc, méthode identique à A).
# 3 tours alternés, 200 itérations après 200 de chauffe. À lancer sous ajp-gpu.
# Usage : abab.sh <binaire> <dossier_moteurs> <in01.bin> <in12.bin>
export LD_LIBRARY_PATH=$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64
bin=$1
moteurs=$2
in01=$3
in12=$4
journal="$(mktemp)"
trap 'rm -f "$journal"' EXIT
(
for tour in 1 2 3; do
  echo "== tour $tour : A = chaîne fusionnée"
  "$bin" chaine_mesure "$moteurs" "$in01" "$in12"
  echo "== tour $tour : B = TensorRT FP16 réseau entier (trtexec)"
  "$HOME/ajp-sdk/tensorrt/bin/trtexec" --loadEngine="$moteurs/entier.engine" --warmUp=500 --iterations=200 \
    --useCudaGraph --noDataTransfers --loadInputs=pair01:"$in01",pair12:"$in12" 2>&1 | grep "GPU Compute Time" | head -1 | sed 's/percentile.*//'
  echo "== tour $tour : B' = TensorRT FP16 réseau entier (banc, même méthode que A)"
  "$bin" plein_mesure "$moteurs/entier.engine" "$in01" "$in12"
done
) | tee "$journal"
# Le seuil de la fiche (gain d'au moins 30 % sur le réseau entier, même méthode : A contre B'), appliqué ici et non
# lu à l'œil (C46-24). Code 1 si la chaîne n'est pas assez plus rapide ou si une mesure manque.
awk -F'médiane ' '
  /^CHAINE graphe CUDA/ { split($2, x, " "); a[++na] = x[1] }
  /^PLEIN graphe CUDA/ { split($2, x, " "); b[++nb] = x[1] }
  function median(v, n,   i, j, t) { for (i = 1; i <= n; i++) for (j = i + 1; j <= n; j++) if (v[j] < v[i]) { t = v[i]; v[i] = v[j]; v[j] = t }; return v[int((n + 1) / 2)] }
  END {
    if (na == 0 || nb == 0) { print "mesures manquantes : pas de verdict" > "/dev/stderr"; exit 1 }
    ma = median(a, na); mb = median(b, nb); g = (ma / mb - 1) * 100
    printf "chaîne %.4f ms contre réseau entier %.4f ms : %+.1f %% (seuil -30 %%) : %s\n", ma, mb, g, (g <= -30 ? "TENU" : "NON TENU")
    exit (g <= -30 ? 0 : 1)
  }' "$journal"
