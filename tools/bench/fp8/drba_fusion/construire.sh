#!/bin/sh
# Compile le banc (nvcc sm_89, TensorRT du SDK local). Usage : construire.sh <binaire>
exec nice -n 19 /opt/cuda/bin/nvcc -arch=sm_89 -std=c++20 -O3 -lineinfo -Xptxas -v \
  -I"$HOME/ajp-sdk/tensorrt/include" -L"$HOME/ajp-sdk/tensorrt/lib" -lnvinfer \
  "$(dirname "$0")/drba_fusion_bench.cu" -o "$1"
