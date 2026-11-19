# gpu : exécution du Plan, ring de sortie, flux de calcul. Thread gpu.
# Chaque sous-dossier (cuda/, vulkan/, metal/...) deviendra une bibliothèque
# chargée à l'exécution, installée seulement si l'utilisateur la choisit.
# ir/ est la représentation interne commune aux moteurs Vulkan et Metal.
add_subdirectory(ir)
# Paquet de poids du SPAN entier en FP8 (C++ pur) ; ses noyaux CUTLASS sont dans cuda/backend.
add_subdirectory(span_fp8)
if(KIRARI_WITH_CUDA)
  add_subdirectory(cuda)
endif()

kirari_add_module(gpu DEPS kirari::core kirari::core_log kirari::kernels)

# L'empreinte des en-têtes de la frontière (cmake/BoundaryHash.cmake), recalculée quand l'un d'eux change.
include(BoundaryHash)
list(TRANSFORM KIRARI_BOUNDARY_HEADERS PREPEND "${PROJECT_SOURCE_DIR}/" OUTPUT_VARIABLE kirari_boundary_files)
kirari_boundary_hash(kirari_boundary ${kirari_boundary_files})
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${kirari_boundary_files})
target_compile_definitions(kirari_gpu INTERFACE KIRARI_BOUNDARY_HASH=0x${kirari_boundary}ULL)

# Socle de calcul Vulkan (vk_compute), sous vulkan/compute.
add_subdirectory(vulkan)
# Les parties portables du backend Core ML, vérifiées partout : les images de l'interpolation
# d'Apple (coreml/vt_pixels.h), la place et les noms du modèle compilé (coreml/coreml_compiled.h).
# Les clés d'Artifact des moteurs fusionnés du backend TensorRT, en C++ pur : testées sans CUDA (B03-09).
kirari_add_module(cuda_fused_keys SOURCES cuda/backend/fused_keys.cpp DEPS kirari::kernels)
# Le choix d'un mélange de morceaux natifs (RIFE natif, GMFSS natif), en C++ pur : testé sans CUDA.
kirari_add_module(cuda_native_mix SOURCES cuda/backend/native_mix.cpp cuda/backend/gmfss_mix.cpp
  cuda/backend/rife_mix.cpp)
kirari_add_module(coreml_pixels SOURCES coreml/coreml_compiled.cpp coreml/vt_pixels.cpp DEPS kirari::gpu kirari::kernels)
if(KIRARI_WITH_COREML)
  add_subdirectory(coreml)
endif()
