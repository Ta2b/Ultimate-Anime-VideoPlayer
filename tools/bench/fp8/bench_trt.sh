#!/usr/bin/env bash
# Banc TensorRT : SPAN FP16 contre FP8 Q/DQ, ms par image. Usage : bench_trt.sh span_fp16.onnx [HxW]
# Lance aussi make_fp8_qdq.py. Sortie : résumé GPU (médiane) de trtexec pour chaque variante.
set -euo pipefail
cd "$(dirname "$0")"; onnx="$1"; hw="${2:-1088x1920}"; h=${hw%x*}; w=${hw#*x}
trtexec=${TRTEXEC:-$(command -v trtexec || ls ~/ajp-sdk/*/bin/trtexec 2>/dev/null | head -1 || true)}
[ -x "${trtexec:-}" ] || { echo "trtexec introuvable (TRTEXEC=...)" >&2; exit 2; }
python3 make_fp8_qdq.py "$onnx" span_fp8_qdq.onnx
shape="--minShapes=input:1x3x${h}x${w} --optShapes=input:1x3x${h}x${w} --maxShapes=input:1x3x${h}x${w}"
for v in fp16 fp8; do
  f=$onnx; flags=""; [ $v = fp8 ] && f=span_fp8_qdq.onnx  # TensorRT 11 : fortement typé, ni --fp16 ni --fp8
  echo "== $v"; "$trtexec" --onnx="$f" $flags $shape --warmUp=500 --iterations=200 --useCudaGraph --noDataTransfers 2>&1 \
    | grep -E "GPU Compute Time.*median|Throughput|FP8|error|Error" | tee "result_$v.txt"
done
