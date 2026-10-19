# Backend CPU de référence : Backend et KernelOps sur l'hôte, par kernels/ref et l'interpréteur
# de gpu_ir. Jamais proposé à l'utilisateur ni livré : il sert aux tests de bout en bout du
# module gpu et au débogage. ajp_backend_cpu_impl sert aussi aux tests sans chargement.
add_library(ajp_backend_cpu_impl STATIC cpu_backend.cpp cpu_kernels.cpp host_tensor.cpp)
target_include_directories(ajp_backend_cpu_impl PUBLIC "${PROJECT_SOURCE_DIR}/src")
target_link_libraries(ajp_backend_cpu_impl PUBLIC ajp::core_types ajp::kernels PRIVATE ajp::gpu_ir)
ajp_set_warnings(ajp_backend_cpu_impl)
add_library(ajp::backend_cpu_impl ALIAS ajp_backend_cpu_impl)

# Une bibliothèque partagée embarque ses dépendances statiques : elles doivent être PIC.
set_property(TARGET ajp_backend_cpu_impl ajp_gpu_ir ajp_kernels ajp_core_log
  PROPERTY POSITION_INDEPENDENT_CODE ON)

# ajp_backend_cpu.so (.dll), chargée par gpu/loader.h comme tout backend.
add_library(ajp_backend_cpu MODULE entry.cpp)
target_link_libraries(ajp_backend_cpu PRIVATE ajp_backend_cpu_impl)
set_target_properties(ajp_backend_cpu PROPERTIES PREFIX "" VISIBILITY_INLINES_HIDDEN ON)
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  # Seuls les points d'entrée sortent : les copies de kernels et gpu_ir restent privées.
  target_link_options(ajp_backend_cpu PRIVATE "LINKER:--exclude-libs,ALL" "LINKER:--no-undefined")
endif()
ajp_set_warnings(ajp_backend_cpu)
