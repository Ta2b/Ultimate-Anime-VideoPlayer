# nvof_dump : les flux NVOF (grille 1, 2 ou 4, coût, temps GPU) de paires de lumas, pour
# l'entraînement de FlowJaNai (precompute_nvof.py) et le banc de la porte 2.1 (nvof_scene.py). Hors du
# lecteur ; les en-têtes BSD de NVIDIA sont dans third_party/nvidia-optical-flow, la bibliothèque
# vient du pilote. Il faut le kit CUDA (cuda.h et libcuda).
#
#   cmake -S tools/nvof_interp/nvof_dump -B build/nvof_dump -DCMAKE_BUILD_TYPE=Release
#   cmake --build build/nvof_dump
cmake_minimum_required(VERSION 3.24)
project(nvof_dump LANGUAGES CXX)
set(NVOF_SDK_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/nvidia-optical-flow" CACHE PATH "Dossier de nvOpticalFlowCuda.h")
find_package(CUDAToolkit REQUIRED)
add_executable(nvof_dump nvof_dump.cpp)
target_compile_features(nvof_dump PRIVATE cxx_std_20)
target_include_directories(nvof_dump PRIVATE "${NVOF_SDK_INCLUDE}")
target_link_libraries(nvof_dump PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
