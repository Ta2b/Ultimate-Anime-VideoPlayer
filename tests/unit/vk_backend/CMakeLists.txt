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
    entry_test.cpp
    family_test.cpp
    gmfss_ops_test.cpp
    kernel_ops_test.cpp
    kernel_planes_test.cpp
    net_alias_test.cpp
    net_conv_test.cpp
    net_ops_test.cpp
    net_rig.cpp
    net_runtime_test.cpp
    slice_ops_test.cpp
  DEPS ajp::gpu_ir ajp::kernels ${CMAKE_DL_LIBS}
  MAIN ajp_test_vulkan)

target_compile_definitions(ajp_vk_backend_tests PRIVATE
  AJP_TEST_BACKEND="$<TARGET_FILE:ajp_backend_vulkan>"
  AJP_TEST_FAKE_BACKEND="$<TARGET_FILE:ajp_vk_backend_fake>")
add_dependencies(ajp_vk_backend_tests ajp_backend_vulkan ajp_vk_backend_fake)
