# kirari_backend_coreml : le backend Vulkan (chaîne sur MoltenVK) dont les réseaux passent par Core ML,
# au travers d'ONNX Runtime (fournisseur CoreML), sur les Mac Apple Silicon. KIRARI_WITH_COREML ;
# KIRARI_ORT_ROOT : une version d'ONNX Runtime (include/, lib/), celle de onnxruntime-osx-arm64.
set(KIRARI_ORT_ROOT "" CACHE PATH "Racine d'ONNX Runtime (include/, lib/)")
find_path(KIRARI_ORT_INCLUDE onnxruntime_cxx_api.h
  HINTS "${KIRARI_ORT_ROOT}/include" "${KIRARI_ORT_ROOT}/include/onnxruntime" REQUIRED)
find_library(KIRARI_ORT_LIBRARY onnxruntime HINTS "${KIRARI_ORT_ROOT}/lib" REQUIRED)

# L'interpolation d'Apple (coreml_vt.h, ses conversions sur le GPU par Metal : vt_gpu.h) et Core ML
# direct (coreml_direct.h) : VideoToolbox, Metal et Core ML, en Objective-C++ sur un Mac ; ailleurs,
# un refus (OBJCXX activé par le CMakeLists.txt de tête).
if(APPLE)
  set(kirari_coreml_objc coreml_direct.mm coreml_vt.mm vt_buffers.mm vt_gpu.mm)
  set_source_files_properties(${kirari_coreml_objc} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
else()
  set(kirari_coreml_objc coreml_direct_none.cpp coreml_vt_none.cpp)
endif()

# Le runtime à part de ses points d'entrée : les tests du backend Vulkan le lient pour comparer
# ses sorties à l'interpréteur sur le vrai matériel (vendor_runtime_test.cpp).
kirari_add_module(coreml_rt SOURCES coreml_measure.cpp coreml_runtime.cpp ${kirari_coreml_objc}
  DEPS kirari::vk_backend kirari::core_log kirari::coreml_pixels PRIVATE_DEPS kirari::dep::spdlog "${KIRARI_ORT_LIBRARY}")
target_include_directories(kirari_coreml_rt SYSTEM PRIVATE "${KIRARI_ORT_INCLUDE}")
if(APPLE)
  target_link_libraries(kirari_coreml_rt PRIVATE
    "-framework CoreML" "-framework VideoToolbox" "-framework CoreVideo" "-framework CoreMedia"
    "-framework Foundation" "-framework IOSurface" "-framework Metal")
endif()
set_target_properties(kirari_coreml_rt PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

add_library(kirari_backend_coreml MODULE entry.cpp)
target_link_libraries(kirari_backend_coreml PRIVATE kirari::coreml_rt kirari::core_log kirari::dep::spdlog)
set_target_properties(kirari_backend_coreml PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(kirari_backend_coreml PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
kirari_set_warnings(kirari_backend_coreml)
