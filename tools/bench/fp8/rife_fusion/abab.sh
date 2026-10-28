#!/bin/sh
# ABAB dans la même séance : A = chaîne fusionnée (graphe CUDA), B = réseau entier TensorRT FP16
# (trtexec --loadEngine, graphe CUDA ; et le même moteur par le banc, méthode identique à A).
# 3 tours alternés, 200 itérations après 200 de chauffe. Usage : abab.sh <binaire> <dossier_moteurs> <in.bin>
export LD_LIBRARY_PATH=$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64
bin=$1
moteurs=$2
in=$3  # entrée réelle (in.bin de ref_bornes.py)
for tour in 1 2 3; do
  echo "== tour $tour : A = chaîne fusionnée"
  "$bin" chaine_mesure "$moteurs" "$in"
  echo "== tour $tour : B = TensorRT FP16 réseau entier (trtexec)"
  "$HOME/ajp-sdk/tensorrt/bin/trtexec" --loadEngine="$moteurs/rife_fixe.engine" --warmUp=500 --iterations=200 \
    --useCudaGraph --noDataTransfers --loadInputs=input:"$in" 2>&1 | grep "GPU Compute Time" | head -1 | sed 's/percentile.*//'
  echo "== tour $tour : B' = TensorRT FP16 réseau entier (banc, même méthode que A)"
  "$bin" plein_mesure "$moteurs/rife_fixe.engine" "$in"
done
