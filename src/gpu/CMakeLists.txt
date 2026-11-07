# gpu : exécution du Plan, ring de sortie, flux de calcul. Thread gpu.
# Chaque sous-dossier (cuda/, vulkan/, metal/...) deviendra une bibliothèque
# chargée à l'exécution, installée seulement si l'utilisateur la choisit.
# ir/ est la représentation interne commune aux moteurs Vulkan et Metal.
add_subdirectory(ir)
# Paquet de poids du SPAN entier en FP8 (C++ pur) ; ses noyaux CUTLASS sont dans cuda/backend.
add_subdirectory(span_fp8)
if(AJP_WITH_CUDA)
  add_subdirectory(cuda)
endif()

ajp_add_module(gpu DEPS ajp::core ajp::core_log ajp::kernels)

# L'empreinte des en-têtes de la frontière (cmake/BoundaryHash.cmake), recalculée quand l'un d'eux change.
include(BoundaryHash)
list(TRANSFORM AJP_BOUNDARY_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/" OUTPUT_VARIABLE ajp_boundary_files)
ajp_boundary_hash(ajp_boundary ${ajp_boundary_files})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${ajp_boundary_files})
target_compile_definitions(ajp_gpu INTERFACE AJP_BOUNDARY_HASH=0x${ajp_boundary}ULL)

# Socle de calcul Vulkan (vk_compute), sous vulkan/compute.
add_subdirectory(vulkan)
# Les parties portables du backend Core ML, vérifiées partout : les images de l'interpolation
# d'Apple (coreml/vt_pixels.h), la place et les noms du modèle compilé (coreml/coreml_compiled.h).
ajp_add_module(coreml_pixels SOURCES coreml/coreml_compiled.cpp coreml/vt_pixels.cpp DEPS ajp::gpu ajp::kernels)
if(AJP_WITH_COREML)
  add_subdirectory(coreml)
endif()
