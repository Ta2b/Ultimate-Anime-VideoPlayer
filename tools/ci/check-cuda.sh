#!/usr/bin/env bash
# Compile la partie CUDA sans GPU : les noyaux .cu, le backend TensorRT et ses tests, que la CI ne
# construit pas (AJP_WITH_CUDA n'y est ON que dans release-windows.yml, sur son entrée tensorrt :
# installer le kit CUDA engage sa licence, une décision du propriétaire du dépôt). À lancer avant
# chaque poussée qui touche src/gpu/cuda, src/kernels ou les contrats du backend, là où le kit CUDA
# et TensorRT sont installés.
# Usage : tools/ci/check-cuda.sh [préréglage]   (linux-prefix-cuda par défaut ; windows-rtx sur le PC)
set -euo pipefail

preset=${1:-linux-prefix-cuda}
root=$(git rev-parse --show-toplevel)
cd "$root"
# Les outils du poste de développement (cmake, ninja dans ~/ajp-tools) et TensorRT dans ~/ajp-sdk/tensorrt, là où
# tools/deps les installe : pris s'ils existent, sans rien changer ailleurs.
for dir in "$HOME/ajp-tools/bin" "$HOME/ajp-tools/venv/bin"; do
  [[ -d "$dir" ]] && PATH="$dir:$PATH"
done
export PATH
trt=()
if [[ -z "${AJP_TENSORRT_ROOT:-}" && -f "$HOME/ajp-sdk/tensorrt/include/NvInfer.h" ]]; then
  trt=(-DAJP_TENSORRT_ROOT="$HOME/ajp-sdk/tensorrt")
elif [[ -n "${AJP_TENSORRT_ROOT:-}" ]]; then
  trt=(-DAJP_TENSORRT_ROOT="$AJP_TENSORRT_ROOT")
fi
command -v nvcc >/dev/null || [[ -n "${CUDACXX:-}" ]] || { echo "nvcc absent : kit CUDA requis" >&2; exit 2; }

cmake --preset "$preset" "${trt[@]}" >/dev/null
# Construction seule : les tests du backend demandent un GPU NVIDIA pour tourner.
cmake --build --preset "$preset" --target ajp_backend_tensorrt_kernels ajp_backend_tensorrt \
  ajp_backend_tensorrt_tests
echo "CUDA : noyaux, backend TensorRT et tests compilés ($preset)"
