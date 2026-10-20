# ajp_backend_migraphx : le backend Vulkan (chaîne sur l'appareil de rendu) dont les réseaux
# passent par MIGraphX (ROCm), sur les GPU AMD. AJP_WITH_MIGRAPHX ; ROCm installé.
find_package(migraphx REQUIRED)
find_package(hip REQUIRED)

add_library(ajp_backend_migraphx MODULE entry.cpp migraphx_runtime.cpp)
target_link_libraries(ajp_backend_migraphx PRIVATE
  ajp::vk_backend ajp::core_log ajp::dep::spdlog migraphx::c hip::host)
set_target_properties(ajp_backend_migraphx PROPERTIES
  PREFIX ""
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  target_link_options(ajp_backend_migraphx PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_migraphx)
