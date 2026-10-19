# gpu : exécution du Plan, ring de sortie, chargement des backends. Thread gpu.
# Chaque backend (vulkan/, cuda/, metal/…) est une bibliothèque chargée à l'exécution,
# installée seulement si l'utilisateur la choisit ; cpu/ est la référence des tests.
# ir/ est la représentation interne commune aux moteurs Vulkan et Metal.
add_subdirectory(ir)

ajp_add_module(gpu
  SOURCES
    executor.cpp gpu_thread.cpp level_scheduler.cpp loader.cpp output_ring.cpp program.cpp worker.cpp
  DEPS ajp::core ajp::kernels ajp::vendor_hdr
  PRIVATE_DEPS ajp::core_log)

add_subdirectory(cpu)
