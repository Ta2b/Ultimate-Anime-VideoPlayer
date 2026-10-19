# Banc des noyaux Vulkan du socle (option AJP_BUILD_BENCH) : ajp_vk_kernels_bench chronomètre
# chaque noyau de KernelRunner par requêtes timestamp, après échauffement, et rend médiane, p95,
# écart type et débit effectif. Statistiques, chronomètre et catalogue forment ajp_vk_bench,
# testée dans tests/unit/vk_bench.
add_library(ajp_vk_bench STATIC
  cases.cpp
  gpu_timer.cpp
  stats.cpp)
add_library(ajp::vk_bench ALIAS ajp_vk_bench)
target_include_directories(ajp_vk_bench PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ajp_vk_bench PUBLIC ajp::vk_compute ajp::kernels)
ajp_set_warnings(ajp_vk_bench)

add_executable(ajp_vk_kernels_bench main.cpp)
target_link_libraries(ajp_vk_kernels_bench PRIVATE ajp::vk_bench)
ajp_set_warnings(ajp_vk_kernels_bench)
