#!/usr/bin/env bash
# Compile la partie CUDA sans GPU : les noyaux .cu, le backend TensorRT et ses tests, que la CI ne
# construit pas (AJP_WITH_CUDA n'y est jamais ON : y installer le kit CUDA engage sa licence, une
# décision du propriétaire du dépôt). À lancer avant chaque poussée qui touche src/gpu/cuda,
# src/kernels ou les contrats du backend, là où le kit CUDA et TensorRT sont installés.
# Usage : tools/ci/check-cuda.sh [préréglage]   (linux-prefix-cuda par défaut ; windows-rtx sur le PC)
set -euo pipefail

preset=${1:-linux-prefix-cuda}
root=$(git rev-parse --show-toplevel)
cd "$root"
command -v nvcc >/dev/null || [[ -n "${CUDACXX:-}" ]] || { echo "nvcc absent : kit CUDA requis" >&2; exit 2; }

cmake --preset "$preset" >/dev/null
# Construction seule : les tests du backend demandent un GPU NVIDIA pour tourner.
cmake --build --preset "$preset" --target ajp_backend_tensorrt_kernels ajp_backend_tensorrt \
  ajp_backend_tensorrt_tests
echo "CUDA : noyaux, backend TensorRT et tests compilés ($preset)"
