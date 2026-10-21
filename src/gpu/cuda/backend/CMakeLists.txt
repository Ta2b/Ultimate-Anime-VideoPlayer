# ajp_backend_tensorrt.so : le backend TensorRT (gpu/backend.h), chargé à l'exécution.
# Il emporte prepare (constructeur trtexec) et sa propre copie de spdlog ; ses symboles
# restent cachés, sauf les trois points d'entrée.
add_library(ajp_backend_tensorrt_kernels STATIC export_kernels.cu gmfss_kernels.cu hdr_kernels.cu
  rife_kernels.cu slice_kernels.cu warp_kernels.cu)
target_include_directories(ajp_backend_tensorrt_kernels PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_backend_tensorrt_kernels PRIVATE ajp::dep::cudart)
# -lineinfo pour ncu ; ptxas donne les registres de chaque noyau au journal de build, et un
# débordement de registres en mémoire locale est une erreur (pixel_kernels.cuh).
target_compile_options(ajp_backend_tensorrt_kernels PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:-Werror=all-warnings -Xcompiler=-Wall,-Wextra -lineinfo
  --ptxas-options=--verbose,--warn-on-spills>)
set_target_properties(ajp_backend_tensorrt_kernels PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(ajp_backend_tensorrt MODULE
  entry.cpp gpu_timing.cpp ring_slot.cpp tensorrt_backend.cpp tensorrt_export.cpp tensorrt_gmfss.cpp
  tensorrt_kernels.cpp tensorrt_lanes.cpp tensorrt_rife.cpp tensorrt_textures.cpp tensorrt_warp.cpp tensorrt_true_hdr.cpp
  trt_engine.cpp)
target_include_directories(ajp_backend_tensorrt PRIVATE "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_backend_tensorrt PRIVATE
  ajp::gpu ajp::prepare ajp_backend_tensorrt_kernels ajp::dep::tensorrt CUDA::cuda_driver ${CMAKE_DL_LIBS})
target_compile_definitions(ajp_backend_tensorrt PRIVATE AJP_TRTEXEC_PATH="${AJP_TRTEXEC}")
set_target_properties(ajp_backend_tensorrt PROPERTIES
  PREFIX "" CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
target_link_options(ajp_backend_tensorrt PRIVATE LINKER:--exclude-libs,ALL LINKER:--no-undefined)
ajp_set_warnings(ajp_backend_tensorrt)

# Les bibliothèques statiques liées dans un objet partagé doivent être compilées en PIC. prepare
# et les modules qu'il tire sont déclarés après gpu : la propriété se pose à la fin du projet.
function(ajp_backend_tensorrt_pic)
  foreach(lib ajp_prepare ajp_core ajp_core_log ajp_kernels ajp_vendor_hdr)
    if(TARGET ${lib})
      set_property(TARGET ${lib} PROPERTY POSITION_INDEPENDENT_CODE ON)
    endif()
  endforeach()
endfunction()
cmake_language(DEFER DIRECTORY "${PROJECT_SOURCE_DIR}" CALL ajp_backend_tensorrt_pic)
