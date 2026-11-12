#!/usr/bin/env bash
# Compile la partie CUDA sans GPU, comme la tâche `cuda` de ci.yml : toutes les cibles (noyaux .cu, noyaux CUTLASS
# du SPAN FP8, backend TensorRT, leurs tests) et les architectures que la CI livre (sm_75 à sm_120), pas la seule
# carte locale. La CI ne peut pas les lancer faute de GPU NVIDIA. Ce script sert en local, avant une poussée qui
# touche src/gpu/cuda, src/kernels ou les contrats du backend, là où le kit CUDA et TensorRT sont installés ; il
# n'est vert que si la CI le sera. Sous Windows (windows-rtx), les architectures du préréglage restent celles de
# CMakeLists.txt : seule la liste de CI de Linux est passée.
# Usage : tools/ci/check-cuda.sh [préréglage] [architectures]
#         (linux-prefix-cuda par défaut ; windows-rtx sur le PC ; 89-real pour aller vite avec la seule carte locale)
set -euo pipefail

preset=${1:-linux-prefix-cuda}
archs=${2:-75-real;86-real;89-real;120-real;120-virtual}
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

arch=()
[[ "$preset" == windows-* ]] || arch=("-DAJP_CUDA_ARCHITECTURES=$archs")
cmake --preset "$preset" "${trt[@]}" "${arch[@]}" >/dev/null
# Construction seule : les tests du backend demandent un GPU NVIDIA pour tourner. Sans --target : tout ce que la
# tâche `cuda` de la CI construit (noyaux CUTLASS du SPAN FP8 et leurs tests compris).
cmake --build --preset "$preset"
echo "CUDA : tout compilé ($preset, architectures ${arch[*]:-du préréglage})"
