# Tests de vk_compute sur l'appareil Vulkan du système (lavapipe en CI) : chaque noyau contre
# kernels/ref, l'exécution rejouée, les sémaphores timeline, l'allocateur et l'adoption.
# Sans Vulkan, les tests qui en ont besoin sont sautés avec leur raison.

# Un chargeur factice, construit ici et jamais rangé dans le dépôt : il s'ouvre mais n'exporte
# pas vkGetInstanceProcAddr.
add_library(kirari_vk_compute_fake_loader SHARED fake_loader.cpp)
kirari_set_warnings(kirari_vk_compute_fake_loader)

kirari_add_unit_tests(vk_compute
  SOURCES
    caps_test.cpp
    color_test.cpp
    context_test.cpp
    detect_test.cpp
    execution_test.cpp
    memory_test.cpp
    resample_test.cpp
    rife_test.cpp
    rig.cpp
    transfer_test.cpp
    warp_test.cpp
  DEPS kirari::kernels
  MAIN kirari_test_vulkan)

target_compile_definitions(kirari_vk_compute_tests PRIVATE
  KIRARI_TEST_FAKE_VULKAN="$<TARGET_FILE:kirari_vk_compute_fake_loader>")
add_dependencies(kirari_vk_compute_tests kirari_vk_compute_fake_loader)
