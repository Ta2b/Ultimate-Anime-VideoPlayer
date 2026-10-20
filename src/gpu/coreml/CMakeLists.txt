# ajp_backend_coreml : le backend Vulkan (chaîne sur MoltenVK) dont les réseaux passent par Core ML,
# au travers d'ONNX Runtime (fournisseur CoreML), sur les Mac Apple Silicon. AJP_WITH_COREML ;
# AJP_ORT_ROOT : une version d'ONNX Runtime (include/, lib/), celle de onnxruntime-osx-arm64.
set(AJP_ORT_ROOT "" CACHE PATH "Racine d'ONNX Runtime (include/, lib/)")
find_path(AJP_ORT_INCLUDE onnxruntime_cxx_api.h
  HINTS "${AJP_ORT_ROOT}/include" "${AJP_ORT_ROOT}/include/onnxruntime" REQUIRED)
find_library(AJP_ORT_LIBRARY onnxruntime HINTS "${AJP_ORT_ROOT}/lib" REQUIRED)

add_library(ajp_backend_coreml MODULE entry.cpp coreml_runtime.cpp)
target_include_directories(ajp_backend_coreml SYSTEM PRIVATE "${AJP_ORT_INCLUDE}")
target_link_libraries(ajp_backend_coreml PRIVATE
  ajp::vk_backend ajp::core_log ajp::dep::spdlog "${AJP_ORT_LIBRARY}")
set_target_properties(ajp_backend_coreml PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_coreml PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_coreml)
