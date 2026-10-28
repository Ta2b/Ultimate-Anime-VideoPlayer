# gpu_span_fp8 : le paquet de poids du SPAN entier en FP8 (noyaux CUTLASS, docs/span-fp8-cutlass.md), ses
# garde-fous de taille, et ce que toute dérivation de ce paquet partage (échelles, mesures PSNR et SSIM, verdict,
# reçu : derive.h). C++ pur, sans CUDA : le paquet se lit et se vérifie partout, les noyaux qui le consomment
# vivent dans le backend TensorRT (gpu/cuda/backend/span_fp8_net.cu).
ajp_add_module(gpu_span_fp8 SOURCES derive.cpp metrics.cpp package.cpp DEPS ajp::core_types)

# Les poids du SPAN lus dans son ONNX (onnx_weights.h) : le lecteur seulement, pas le backend, qui n'emporte pas gpu_ir.
ajp_add_module(gpu_span_fp8_onnx SOURCES onnx_weights.cpp DEPS ajp::gpu_span_fp8 ajp::gpu_ir ajp::kernels)
