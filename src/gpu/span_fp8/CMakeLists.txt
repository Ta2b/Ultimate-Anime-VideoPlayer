# gpu_span_fp8 : le paquet de poids du SPAN entier en FP8 (noyaux CUTLASS, docs/span-fp8-cutlass.md) et ses
# garde-fous de taille. C++ pur, sans CUDA : le paquet se lit et se vérifie partout, les noyaux qui le
# consomment vivent dans le backend TensorRT (gpu/cuda/backend/span_fp8_net.cu).
ajp_add_module(gpu_span_fp8 SOURCES package.cpp DEPS ajp::core_types)
