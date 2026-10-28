# nvof_dump : les flux NVOF (grille 4, sens avant et arrière) de paires de lumas, pour
# l'entraînement de FlowJaNai (precompute_nvof.py). Hors du lecteur ; il faut les en-têtes de
# l'Optical Flow SDK de NVIDIA (sa licence s'accepte au téléchargement) et le pilote NVIDIA.
#
#   cmake -S tools/nvof_interp/nvof_dump -B build/nvof_dump -DNVOF_SDK_INCLUDE=<sdk>/NvOFInterface
#   cmake --build build/nvof_dump
cmake_minimum_required(VERSION 3.24)
project(nvof_dump LANGUAGES CXX)
set(NVOF_SDK_INCLUDE "" CACHE PATH "Dossier de nvOpticalFlowCuda.h (Optical Flow SDK, NvOFInterface)")
find_package(CUDAToolkit REQUIRED)
add_executable(nvof_dump nvof_dump.cpp)
target_compile_features(nvof_dump PRIVATE cxx_std_20)
target_include_directories(nvof_dump PRIVATE "${NVOF_SDK_INCLUDE}")
target_link_libraries(nvof_dump PRIVATE CUDA::cuda_driver ${CMAKE_DL_LIBS})
