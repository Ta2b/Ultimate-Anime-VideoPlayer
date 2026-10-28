#!/bin/sh
# Qualité : sorties de la chaîne fusionnée et du réseau entier TensorRT sur les 7 cas (à lancer sous ajp-gpu).
# Usage : essai.sh <binaire> <dossier_moteurs> <dossier_refs> <dossier_sorties>
# Les comparaisons PSNR/SSIM se font ensuite avec qualite.py.
export LD_LIBRARY_PATH=$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64
bin=$1
moteurs=$2
refs=$3
sorties=$4
mkdir -p "$sorties"
for c in s5_6 s5_20 s9_6 s9_20 s14_6 s14_20 s9_20_t1.25; do
  "$bin" executer "$moteurs" "$refs/$c/in01.bin" "$refs/$c/in12.bin" "$sorties/chaine_$c.bin"
  "$bin" plein "$moteurs/entier.engine" "$refs/$c/in01.bin" "$refs/$c/in12.bin" "$sorties/plein_$c.bin"
done
