# gpu : exécution du Plan, ring de sortie, flux de calcul. Thread gpu.
# Chaque sous-dossier (cuda/, vulkan/, metal/...) deviendra une bibliothèque
# chargée à l'exécution, installée seulement si l'utilisateur la choisit.
ajp_add_module(gpu DEPS ajp::core ajp::kernels ajp::vendor_hdr)

# Socle de calcul Vulkan (vk_compute), sous vulkan/compute.
add_subdirectory(vulkan)
