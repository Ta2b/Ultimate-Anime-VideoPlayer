# Tests de vk_compute sur l'appareil Vulkan du système (lavapipe en CI) : chaque noyau contre
# kernels/ref, l'exécution rejouée, les sémaphores timeline, l'allocateur et l'adoption.
# Sans Vulkan, les tests qui en ont besoin sont sautés avec leur raison.

# Un chargeur factice, construit ici et jamais rangé dans le dépôt : il s'ouvre mais n'exporte
# pas vkGetInstanceProcAddr.
add_library(ajp_vk_compute_fake_loader SHARED fake_loader.cpp)
ajp_set_warnings(ajp_vk_compute_fake_loader)

ajp_add_unit_tests(vk_compute
  SOURCES
    color_test.cpp
    context_test.cpp
    detect_test.cpp
    execution_test.cpp
    host_allocations.cpp
    memory_test.cpp
    resample_test.cpp
    rife_test.cpp
    rig.cpp
    spirv_reflect_test.cpp
    transfer_test.cpp
    warp_test.cpp
  DEPS ajp::kernels)

target_compile_definitions(ajp_vk_compute_tests PRIVATE
  AJP_TEST_FAKE_VULKAN="$<TARGET_FILE:ajp_vk_compute_fake_loader>")
add_dependencies(ajp_vk_compute_tests ajp_vk_compute_fake_loader)
