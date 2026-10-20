#!/usr/bin/env bash
# Chaque shader de src/kernels/glsl compile en SPIR-V et passe spirv-val, avec les options de
# cmake/AjpSpirv.cmake (vulkan1.2, -Werror), dans sa variante float16 (AJP_TENSOR_F16) quand il
# lit des tenseurs (il inclut tensor.glsl, la liste tensor_shaders de src/gpu/vulkan/compute).
# L'en-tête des constantes vient du même programme que la construction (src/kernels/glsl_main.cpp),
# compilé ici seul : la tâche « règles » de la CI n'a ni dépendance ni configuration CMake.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

"${CXX:-c++}" -std=c++23 -O1 -I "$root/src" -o "$work/ajp_kernels_glsl" \
  "$root/src/kernels/glsl.cpp" "$root/src/kernels/glsl_main.cpp"
"$work/ajp_kernels_glsl" "$work/ajp_kernels_spec.glsl"

status=0
count=0
for shader in "$root"/src/kernels/glsl/*.comp; do
  name="$(basename "$shader" .comp)"
  variants=("")
  if grep -q '"tensor.glsl"' "$shader"; then variants=("" "-DAJP_TENSOR_F16"); fi
  for define in "${variants[@]}"; do
    spv="$work/$name${define:+_f16}.spv"
    # shellcheck disable=SC2086  # un seul mot, vide ou -D
    if ! glslc --target-env=vulkan1.2 -Werror -I "$work" -I "$root/src/kernels/glsl" $define -o "$spv" "$shader"; then
      echo "::error file=src/kernels/glsl/$name.comp::ne compile pas${define:+ en float16}"
      status=1
      continue
    fi
    if ! spirv-val --target-env vulkan1.2 "$spv"; then
      echo "::error file=src/kernels/glsl/$name.comp::SPIR-V refusé par spirv-val${define:+ en float16}"
      status=1
    fi
    count=$((count + 1))
  done
done
echo "$count modules SPIR-V vérifiés"
exit $status
