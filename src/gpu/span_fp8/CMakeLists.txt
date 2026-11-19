# gpu_span_fp8 : le paquet de poids du SPAN entier en FP8 (noyaux CUTLASS, docs/span-fp8-cutlass.md), ses
# garde-fous de taille, et ce que toute dérivation de ce paquet partage (échelles, mesures PSNR et SSIM, verdict,
# reçu : derive.h). C++ pur, sans CUDA : le paquet se lit et se vérifie partout, les noyaux qui le consomment
# vivent dans le backend TensorRT (gpu/cuda/backend/span_fp8_net.cu).
kirari_add_module(gpu_span_fp8 SOURCES acc16_policy.cpp derive.cpp forward.cpp hard_cases.cpp metrics.cpp package.cpp quantize.cpp
  DEPS kirari::core_types PRIVATE_DEPS kirari::kernels)

# Les poids du SPAN lus dans son ONNX (onnx_weights.h), pour le lecteur, et le graphe ONNX rebâti d'un paquet
# (package_model.h), pour le moteur Vulkan : pas le backend TensorRT, qui n'emporte pas gpu_ir.
kirari_add_module(gpu_span_fp8_onnx SOURCES onnx_weights.cpp package_model.cpp DEPS kirari::gpu_span_fp8 kirari::gpu_ir kirari::kernels)
