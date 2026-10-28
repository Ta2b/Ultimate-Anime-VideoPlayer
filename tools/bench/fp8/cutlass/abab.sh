#!/bin/sh
# Mesure ABAB dans la même séance : TensorRT FP16 (même ONNX, entrée 1x3x1088x1920) contre span_net (CUTLASS).
#   abab.sh MODELE.onnx DOSSIER_SPAN_NET BIN_SPAN_NET [BIN_SPAN_NET_ACC16] > sortie.txt
# Chaque commande GPU passe par le verrou ajp-gpu, une à la fois. Le moteur TensorRT est construit une fois
# (niveau d'optimisation 5), puis chargé à chaque tour.
export LD_LIBRARY_PATH="$HOME/ajp-sdk/tensorrt/lib:/opt/cuda/lib64"
GPU="$HOME/ajp-tools/bin/ajp-gpu"
TRT="$HOME/ajp-sdk/tensorrt/bin/trtexec"
MODEL="$1"; DIR="$2"; A="$3"; A16="$4"
ENG="${DIR}/span_fp16.engine"
[ -f "$ENG" ] || "$GPU" "$TRT" --onnx="$MODEL" --minShapes=input:1x3x1088x1920 --optShapes=input:1x3x1088x1920 --maxShapes=input:1x3x1088x1920 --builderOptimizationLevel=5 \
  --saveEngine="$ENG" --skipInference > "${DIR}/trt_build.log" 2>&1
for round in 1 2 3; do
  echo "== tour $round : A = CUTLASS (acc f32)"
  "$GPU" "$A" "$DIR" | grep TOTAL
  if [ -n "$A16" ]; then
    echo "== tour $round : A' = CUTLASS (hors blocs acc f16)"
    "$GPU" "$A16" "$DIR" | grep TOTAL
  fi
  echo "== tour $round : B = TensorRT FP16"
  "$GPU" "$TRT" --loadEngine="$ENG" --shapes=input:1x3x1088x1920 --useCudaGraph --warmUp=800 --iterations=300 \
    --noDataTransfers 2>&1 | grep "GPU Compute Time:" | sed 's/percentile.*//'
done
