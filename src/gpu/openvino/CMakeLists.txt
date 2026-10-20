# ajp_backend_openvino : le backend Vulkan (chaîne sur l'appareil de rendu) dont les réseaux
# passent par OpenVINO, sur les GPU Intel (Arc, intégrés Xe). AJP_WITH_OPENVINO.
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

# Le runtime à part de ses points d'entrée : les tests du backend Vulkan le lient pour comparer
# ses sorties à l'interpréteur sur le vrai matériel (vendor_runtime_test.cpp).
ajp_add_module(openvino_rt SOURCES openvino_runtime.cpp
  DEPS ajp::vk_backend ajp::core_log PRIVATE_DEPS ajp::dep::spdlog openvino::runtime)
set_target_properties(ajp_openvino_rt PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)

add_library(ajp_backend_openvino MODULE entry.cpp)
target_link_libraries(ajp_backend_openvino PRIVATE ajp::openvino_rt ajp::core_log ajp::dep::spdlog)
set_target_properties(ajp_backend_openvino PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_openvino PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_openvino)
