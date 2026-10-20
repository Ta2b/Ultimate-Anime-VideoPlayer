# ajp_backend_coreml : le backend Vulkan (chaîne sur MoltenVK) dont les réseaux passent par Core ML,
# au travers d'ONNX Runtime (fournisseur CoreML), sur les Mac Apple Silicon. AJP_WITH_COREML ;
# AJP_ORT_ROOT : une version d'ONNX Runtime (include/, lib/), celle de onnxruntime-osx-arm64.
set(AJP_ORT_ROOT "" CACHE PATH "Racine d'ONNX Runtime (include/, lib/)")
find_path(AJP_ORT_INCLUDE onnxruntime_cxx_api.h
  HINTS "${AJP_ORT_ROOT}/include" "${AJP_ORT_ROOT}/include/onnxruntime" REQUIRED)
find_library(AJP_ORT_LIBRARY onnxruntime HINTS "${AJP_ORT_ROOT}/lib" REQUIRED)

# Le runtime à part de ses points d'entrée : les tests du backend Vulkan le lient pour comparer
# ses sorties à l'interpréteur sur le vrai matériel (vendor_runtime_test.cpp).
ajp_add_module(coreml_rt SOURCES coreml_measure.cpp coreml_runtime.cpp
  DEPS ajp::vk_backend ajp::core_log PRIVATE_DEPS ajp::dep::spdlog "${AJP_ORT_LIBRARY}")
target_include_directories(ajp_coreml_rt SYSTEM PRIVATE "${AJP_ORT_INCLUDE}")
set_target_properties(ajp_coreml_rt PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

add_library(ajp_backend_coreml MODULE entry.cpp)
target_link_libraries(ajp_backend_coreml PRIVATE ajp::coreml_rt ajp::core_log ajp::dep::spdlog)
set_target_properties(ajp_backend_coreml PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_coreml PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_coreml)
