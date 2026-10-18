# gpu : exécution du Plan, ring de sortie, flux de calcul. Thread gpu.
# Chaque sous-dossier (cuda/, vulkan/, metal/...) deviendra une bibliothèque
# chargée à l'exécution, installée seulement si l'utilisateur la choisit.
# ir/ est la représentation interne commune aux moteurs Vulkan et Metal.
add_subdirectory(ir)

ajp_add_module(gpu DEPS ajp::core ajp::kernels ajp::vendor_hdr)
