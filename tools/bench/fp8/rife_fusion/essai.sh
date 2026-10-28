#!/bin/sh
# Qualité : sorties de la chaîne fusionnée et du réseau entier TensorRT sur les 6 cas, puis mesure des noyaux.
# Usage : essai.sh <binaire> <dossier_moteurs> <dossier_bornes> <dossier_sorties>
# (à lancer sous ajp-gpu ; les comparaisons PSNR/SSIM se font ensuite avec qualite.py)
export LD_LIBRARY_PATH=$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64
bin=$1
moteurs=$2
bornes=$3
sorties=$4
mkdir -p "$sorties"
for c in s5_6 s5_20 s9_6 s9_20 s14_6 s14_20; do
  "$bin" executer "$moteurs" "$bornes/$c/in.bin" "$sorties/chaine_$c.bin"
  "$bin" plein "$moteurs/rife_fixe.engine" "$bornes/$c/in.bin" "$sorties/plein_$c.bin"
done
"$bin" mesurer "$moteurs" "$bornes/s9_20/in.bin"
