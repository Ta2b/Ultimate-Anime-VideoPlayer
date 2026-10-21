# Tests du backend Vulkan sur l'appareil Vulkan du système (la RTX sur le PC de référence,
# lavapipe en CI) : chaque opération du moteur de réseaux contre l'interpréteur de gpu_ir,
# graphes façon SPAN et RIFE, artefacts, enregistrements rejoués, KernelOps contre kernels/ref,
# familles de GPU, autotuner, chargement de la bibliothèque avec la bonne et une mauvaise ABI.
# Sans Vulkan, les tests qui en ont besoin sont sautés avec leur raison.

# Une bibliothèque de backend d'une autre ABI, construite ici et jamais rangée dans le dépôt.
add_library(ajp_vk_backend_fake MODULE fake_backend.cpp)
set_target_properties(ajp_vk_backend_fake PROPERTIES PREFIX "" CXX_VISIBILITY_PRESET hidden)
ajp_set_warnings(ajp_vk_backend_fake)

ajp_add_unit_tests(vk_backend
  SOURCES
    backend_rig.cpp
    backend_test.cpp
    deferred_steps_test.cpp
    entry_test.cpp
    family_test.cpp
    gmfss_ops_test.cpp
    kernel_ops_test.cpp
    kernel_planes_test.cpp
    net_alias_test.cpp
    net_conv_test.cpp
    net_measure_test.cpp
    net_ops_test.cpp
    net_rig.cpp
    net_runtime_test.cpp
    runtime_queue_test.cpp
    slice_ops_test.cpp
    test_models.cpp
    vendor_runtime_test.cpp
  # core : le SHA-256 du vrai modèle des références (core/model_index.h).
  DEPS ajp::gpu_ir ajp::kernels ajp::core ${CMAKE_DL_LIBS}
  MAIN ajp_test_vulkan)

# Sorties de référence par modèle (vendor_runtime_test.cpp) : enregistrées sous le dossier du
# build, puis comparées à chaque runtime des réseaux que ce build a (MIGraphX, OpenVINO, Core ML)
# quand son matériel est là. AJP_SPAN_MODEL : le vrai SPAN d'AnimeJaNai installé sur le poste,
# lu en place ; absent, seuls les modèles synthétiques servent.
set(AJP_SPAN_MODEL "$ENV{HOME}/.local/share/AnimeJaNai-HDR-linux/animejanai/onnx/2x_AnimeJaNai_HD_V3.1Sharp1_Performance_SPANF3_b5f48_unshuffle_fp16.onnx"
    CACHE FILEPATH "Modèle SPAN ×2 d'AnimeJaNai pour les tests des runtimes")
target_compile_definitions(ajp_vk_backend_tests PRIVATE
  AJP_TEST_BACKEND="$<TARGET_FILE:ajp_backend_vulkan>"
  AJP_TEST_FAKE_BACKEND="$<TARGET_FILE:ajp_vk_backend_fake>"
  AJP_TEST_REFERENCE_DIR="${CMAKE_CURRENT_BINARY_DIR}/reference"
  AJP_SPAN_MODEL="${AJP_SPAN_MODEL}")
add_dependencies(ajp_vk_backend_tests ajp_backend_vulkan ajp_vk_backend_fake)
foreach(vendor IN ITEMS migraphx openvino coreml winml)
  if(TARGET ajp_${vendor}_rt)
    string(TOUPPER "${vendor}" upper)
    target_link_libraries(ajp_vk_backend_tests PRIVATE ajp::${vendor}_rt)
    target_compile_definitions(ajp_vk_backend_tests PRIVATE AJP_TEST_WITH_${upper}=1)
  endif()
endforeach()
