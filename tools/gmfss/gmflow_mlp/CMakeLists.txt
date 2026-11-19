# Le MLP de GMFlow natif hors du lecteur (docs/gmflow-natif.md, « MLP natif (PR 3) »), la lecture et le décodeur
# ONNX compilés tels quels depuis src/ :
#  - gmflow_mlp_vide (sans GPU) : les poids et les emplois que lit prepare/gmflow_mlp.h, vidés pour
#    gmflow_mlp_emule.py, qui les rejoue avec les arrondis du noyau contre ONNX Runtime ;
#  - gmflow_mlp_banc (avec nvcc) : le noyau de gmflow_mlp_net.cu chronométré, et ses sorties comparées à celles de
#    l'émulation (gmflow_mlp_banc.sh le lance face à trtexec).
#
#   cmake -S tools/gmfss/gmflow_mlp -B build/gmflow_mlp -DCMAKE_BUILD_TYPE=Release [-DCMAKE_CUDA_ARCHITECTURES=89]
#   cmake --build build/gmflow_mlp
cmake_minimum_required(VERSION 3.24)
project(gmflow_mlp LANGUAGES CXX)
set(KIRARI_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../src")
set(gmflow_mlp_lecture "${KIRARI_SRC}/prepare/gmflow_mlp.cpp" "${KIRARI_SRC}/gpu/ir/onnx_reader.cpp"
    "${KIRARI_SRC}/gpu/ir/onnx_tensor.cpp" "${KIRARI_SRC}/gpu/ir/protobuf.cpp" "${KIRARI_SRC}/kernels/ref/half.cpp")

add_executable(gmflow_mlp_vide gmflow_mlp_vide.cpp ${gmflow_mlp_lecture})
target_compile_features(gmflow_mlp_vide PRIVATE cxx_std_23)
target_include_directories(gmflow_mlp_vide PRIVATE "${KIRARI_SRC}")

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_executable(gmflow_mlp_banc gmflow_mlp_banc.cpp "${KIRARI_SRC}/gpu/cuda/backend/gmflow_mlp_net.cu"
                 ${gmflow_mlp_lecture})
  target_compile_features(gmflow_mlp_banc PRIVATE cxx_std_23 cuda_std_20)
  target_include_directories(gmflow_mlp_banc PRIVATE "${KIRARI_SRC}")
  target_link_libraries(gmflow_mlp_banc PRIVATE CUDA::cudart)
  target_compile_options(gmflow_mlp_banc PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --ptxas-options=--warn-on-spills>)
else()
  message(STATUS "gmflow_mlp_banc : pas de nvcc, seul gmflow_mlp_vide est construit")
endif()
