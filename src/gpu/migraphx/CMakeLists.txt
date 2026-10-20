# ajp_backend_migraphx : le backend Vulkan (chaîne sur l'appareil de rendu) dont les réseaux
# passent par MIGraphX (ROCm), sur les GPU AMD. AJP_WITH_MIGRAPHX ; ROCm installé.
#
# Le module ne voit que l'API C de MIGraphX (et son enveloppe migraphx.hpp) et le runtime HIP :
# leurs en-têtes et bibliothèques se cherchent seuls. migraphx-config.cmake et hip-config.cmake
# tireraient MIOpen, rocBLAS, hipBLASLt et le compilateur AMD (rocm-llvm, device-libs, comgr),
# dont ce module n'a pas besoin pour se construire.
set(ajp_rocm_hints "$ENV{ROCM_PATH}" /opt/rocm)
find_path(AJP_MIGRAPHX_INCLUDE migraphx/migraphx.hpp HINTS ${ajp_rocm_hints} PATH_SUFFIXES include REQUIRED)
find_library(AJP_MIGRAPHX_C migraphx_c HINTS ${ajp_rocm_hints} PATH_SUFFIXES lib REQUIRED)
find_path(AJP_HIP_INCLUDE hip/hip_runtime_api.h HINTS ${ajp_rocm_hints} PATH_SUFFIXES include REQUIRED)
find_library(AJP_HIP_RUNTIME amdhip64 HINTS ${ajp_rocm_hints} PATH_SUFFIXES lib REQUIRED)

add_library(ajp_backend_migraphx MODULE entry.cpp migraphx_device.cpp migraphx_runtime.cpp)
target_include_directories(ajp_backend_migraphx SYSTEM PRIVATE "${AJP_MIGRAPHX_INCLUDE}" "${AJP_HIP_INCLUDE}")
# Ce que hip::host pose : le runtime HIP d'AMD (et non la couche CUDA de HIP).
target_compile_definitions(ajp_backend_migraphx PRIVATE __HIP_PLATFORM_AMD__=1 __HIP_PLATFORM_HCC__=1)
target_link_libraries(ajp_backend_migraphx PRIVATE
  ajp::vk_backend ajp::core_log ajp::dep::spdlog "${AJP_MIGRAPHX_C}" "${AJP_HIP_RUNTIME}")
set_target_properties(ajp_backend_migraphx PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_migraphx PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_migraphx)
