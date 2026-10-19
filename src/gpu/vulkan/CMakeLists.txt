# vulkan : moteur Vulkan. compute/ est le socle de calcul (contexte, mémoire, pipelines,
# exécution) et les noyaux GLSL de kernels/spec.h ; backend/ est le backend chargé à
# l'exécution (ajp_backend_vulkan), qui exécute aussi les réseaux importés par gpu_ir avec les
# shaders de shaders/.
add_subdirectory(compute)
add_subdirectory(backend)
