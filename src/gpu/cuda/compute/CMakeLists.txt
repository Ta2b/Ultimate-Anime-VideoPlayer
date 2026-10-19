# cuda_compute : port CUDA des noyaux de kernels/spec.h, contexte primaire de l'appareil choisi
# par UUID, mémoire bornée et import des images et sémaphores Vulkan exportés.
#
# Deux bibliothèques : les noyaux (.cu, C++20, nvcc) et les plans hôtes (.cpp, C++23) qui
# valident chaque appel, puis lancent les noyaux. Les .cu se compilent sans contraction FMA :
# chaque a × b + c s'arrondit en deux temps, dans l'ordre écrit de la spécification, comme la
# référence CPU (kernels/ref), et les sorties en sont exactes là où elle l'est.
add_library(ajp_cuda_compute_kernels STATIC
  kernels/detect.cu
  kernels/ingest.cu
  kernels/output.cu
  kernels/resample.cu
  kernels/rife.cu
  kernels/transfer.cu
  kernels/warp.cu)
target_include_directories(ajp_cuda_compute_kernels PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_cuda_compute_kernels PUBLIC ajp::dep::cudart)
# Avertissements de nvcc et du compilateur hôte qu'il pilote, en erreurs comme ailleurs.
if(MSVC)
  set(ajp_cuda_host_warnings -Xcompiler=/W4 $<$<BOOL:${AJP_WARNINGS_AS_ERRORS}>:-Xcompiler=/WX>)
else()
  set(ajp_cuda_host_warnings -Xcompiler=-Wall,-Wextra $<$<BOOL:${AJP_WARNINGS_AS_ERRORS}>:-Xcompiler=-Werror>)
endif()
target_compile_options(ajp_cuda_compute_kernels PRIVATE
  --fmad=false
  ${ajp_cuda_host_warnings}
  $<$<BOOL:${AJP_WARNINGS_AS_ERRORS}>:-Werror=all-warnings>)

ajp_add_module(cuda_compute
  SOURCES
    detect.cpp
    device.cpp
    interop.cpp
    memory.cpp
    refusal.cpp
    resize.cpp
    rife.cpp
    transfer.cpp
    image.cpp
    warp.cpp
    weights.cpp
    yuv.cpp
  DEPS ajp::core_types ajp::kernels ajp::dep::cudart
  PRIVATE_DEPS ajp_cuda_compute_kernels)
