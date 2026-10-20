# ajp_backend_openvino : le backend Vulkan (chaîne sur l'appareil de rendu) dont les réseaux
# passent par OpenVINO, sur les GPU Intel (Arc, intégrés Xe). AJP_WITH_OPENVINO.
find_package(OpenVINO REQUIRED COMPONENTS Runtime)

add_library(ajp_backend_openvino MODULE entry.cpp openvino_runtime.cpp)
target_link_libraries(ajp_backend_openvino PRIVATE
  ajp::vk_backend ajp::core_log ajp::dep::spdlog openvino::runtime)
set_target_properties(ajp_backend_openvino PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_openvino PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_openvino)
