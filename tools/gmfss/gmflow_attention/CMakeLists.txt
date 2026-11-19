# Les couches d'attention de GMFlow natif hors du lecteur (docs/gmflow-natif.md, « Attention native (PR 2) ») :
#  - gmflow_attention_vide : l'attention telle que la lit prepare/gmflow_attention_native.h (poids et plan), vidée
#    pour gmflow_attention_emule.py et gmflow_attention_onnx.py ; sans GPU ni dépendance, la lecture et le
#    décodeur ONNX compilés depuis src/ ;
#  - gmflow_attention_banc (avec -DKIRARI_BANC_CUDA=ON, nvcc et le runtime CUDA) : une couche native chronométrée
#    en graphe CUDA (QKV, puis l'attention fondue), ou ses sorties comparées à celles qu'écrit l'émulation.
#
#   cmake -S tools/gmfss/gmflow_attention -B build/gmflow_attention -DCMAKE_BUILD_TYPE=Release [-DKIRARI_BANC_CUDA=ON]
#   cmake --build build/gmflow_attention
cmake_minimum_required(VERSION 3.24)
project(gmflow_attention LANGUAGES CXX)
set(KIRARI_SRC "${CMAKE_CURRENT_SOURCE_DIR}/../../../src")
set(gmflow_attention_lecture "${KIRARI_SRC}/prepare/gmflow_attention_native.cpp"
    "${KIRARI_SRC}/prepare/gmflow_attention_check.cpp" "${KIRARI_SRC}/prepare/gmflow_attention_mask.cpp"
    "${KIRARI_SRC}/prepare/gmflow_onnx_view.cpp"
    "${KIRARI_SRC}/gpu/ir/onnx_reader.cpp" "${KIRARI_SRC}/gpu/ir/onnx_tensor.cpp" "${KIRARI_SRC}/gpu/ir/protobuf.cpp"
    "${KIRARI_SRC}/kernels/ref/half.cpp")
add_executable(gmflow_attention_vide gmflow_attention_vide.cpp ${gmflow_attention_lecture})
set_target_properties(gmflow_attention_vide PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
target_include_directories(gmflow_attention_vide PRIVATE "${KIRARI_SRC}")

option(KIRARI_BANC_CUDA "Construire le banc CUDA d'une couche d'attention" OFF)
if(KIRARI_BANC_CUDA)
  # Le PC de référence (RTX 4070 Ti) : sm 8.9, sauf -DCMAKE_CUDA_ARCHITECTURES=…
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89-real)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  # Le banc lit la paire en C++23 ; nvcc ne compile que le réseau, en C++20 (gmflow_attention_net.h).
  add_executable(gmflow_attention_banc gmflow_attention_banc.cpp
                 "${KIRARI_SRC}/gpu/cuda/backend/gmflow_attention_net.cu" ${gmflow_attention_lecture})
  set_target_properties(gmflow_attention_banc PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON CUDA_STANDARD 20
                                                         CUDA_STANDARD_REQUIRED ON)
  target_include_directories(gmflow_attention_banc PRIVATE "${KIRARI_SRC}")
  target_compile_options(gmflow_attention_banc PRIVATE
                         $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo --ptxas-options=--verbose,--warn-on-spills>)
  target_link_libraries(gmflow_attention_banc PRIVATE CUDA::cudart)
endif()
