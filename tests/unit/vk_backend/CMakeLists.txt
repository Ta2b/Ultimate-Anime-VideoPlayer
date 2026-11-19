# Tests du backend Vulkan sur l'appareil Vulkan du système (la RTX sur le PC de référence,
# lavapipe en CI) : chaque opération du moteur de réseaux contre l'interpréteur de gpu_ir,
# graphes façon SPAN et RIFE, artefacts, enregistrements rejoués, KernelOps contre kernels/ref,
# familles de GPU, autotuner, chargement de la bibliothèque avec la bonne et une mauvaise ABI.
# Sans Vulkan, les tests qui en ont besoin sont sautés avec leur raison.

# Une bibliothèque de backend d'une autre ABI, construite ici et jamais rangée dans le dépôt.
add_library(kirari_vk_backend_fake MODULE fake_backend.cpp)
set_target_properties(kirari_vk_backend_fake PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
kirari_set_warnings(kirari_vk_backend_fake)

kirari_add_unit_tests(vk_backend
  SOURCES
    backend_rig.cpp
    backend_test.cpp
    deband_test.cpp
    deferred_steps_test.cpp
    entry_test.cpp
    family_test.cpp
    host_import_test.cpp
    kernel_ops_test.cpp
    pan_warp_test.cpp
    kernel_planes_test.cpp
    net_alias_test.cpp
    net_artcnn_test.cpp
    net_conv_m8_test.cpp
    net_conv_test.cpp
    net_int8_test.cpp
    net_gate_test.cpp
    net_hoist_test.cpp
    net_measure_test.cpp
    net_ops_test.cpp
    net_passes_test.cpp
    net_profile_test.cpp
    net_rig.cpp
    net_runtime_test.cpp
    span8_test.cpp
    net_sums_test.cpp
    drba_fused_chain_test.cpp
    rife_fused_chain_test.cpp
    rife_fused_test.cpp
    ring_warp_test.cpp
    runtime_queue_test.cpp
    slice_ops_test.cpp
    spirv_shared.cpp
    stage_clock_test.cpp
    test_models.cpp
    vendor_runtime_test.cpp
  # core : le SHA-256 du vrai modèle des références (core/model_index.h).
  DEPS kirari::gpu_ir kirari::gpu_span_fp8_onnx kirari::kernels kirari::core kirari::prepare ${CMAKE_DL_LIBS}
  MAIN kirari_test_vulkan)

# Sorties de référence par modèle (vendor_runtime_test.cpp) : enregistrées sous le dossier du
# build, puis comparées à chaque runtime des réseaux que ce build a (Core ML)
# quand son matériel est là. KIRARI_SPAN_MODEL : le vrai SPAN d'AnimeJaNai installé sur le poste,
# lu en place ; absent, seuls les modèles synthétiques servent.
set(KIRARI_SPAN_MODEL "${KIRARI_ANIMEJANAI_ONNX_DIR}/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
    CACHE FILEPATH "Modèle SPAN ×2 d'AnimeJaNai pour les tests des runtimes")
# Le dossier des RIFE du lecteur (rife_v4.22_lite.onnx, rife_v4.25.onnx), lus en place par le test caché
# [.rife_fused] : la chaîne fusionnée contre le RIFE entier, tous deux sur le backend Vulkan.
set(KIRARI_RIFE_MODEL_DIR "${KIRARI_MODELS_DIR}" CACHE PATH "Dossier des RIFE ONNX du lecteur (test caché [.rife_fused])")
# Les ArtCNN livrés sont rangés dans le dépôt (190 et 51 Ko) : lus en place par net_artcnn_test.cpp.
target_compile_definitions(kirari_vk_backend_tests PRIVATE
  KIRARI_ARTCNN_MODEL_DIR="${PROJECT_SOURCE_DIR}/tools/upscale/models"
  KIRARI_RIFE_MODEL_DIR="${KIRARI_RIFE_MODEL_DIR}"
  KIRARI_TEST_BACKEND="$<TARGET_FILE:kirari_backend_vulkan>"
  KIRARI_TEST_FAKE_BACKEND="$<TARGET_FILE:kirari_vk_backend_fake>"
  KIRARI_TEST_REFERENCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/reference"
  KIRARI_SPAN_MODEL="${KIRARI_SPAN_MODEL}")
add_dependencies(kirari_vk_backend_tests kirari_backend_vulkan kirari_vk_backend_fake)
foreach(vendor IN ITEMS coreml)
  if(TARGET kirari_${vendor}_rt)
    string(TOUPPER "${vendor}" upper)
    target_link_libraries(kirari_vk_backend_tests PRIVATE kirari::${vendor}_rt)
    target_compile_definitions(kirari_vk_backend_tests PRIVATE KIRARI_TEST_WITH_${upper}=1)
  endif()
endforeach()

# Les tests cachés du repli Vulkan sur la vraie carte (B40-02) : RIFE et DistilDRBA fusionnés contre le moteur entier, le
# profil de RIFE et les sommes des convolutions (`ctest -L gpu`, voir tests/CMakeLists.txt).
kirari_add_hardware_tests(kirari_vk_backend_tests gpu 3600 rife_fused drba_fused rife_profil sommes)
