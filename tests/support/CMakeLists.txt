# Support des tests qui créent un appareil Vulkan (vk_compute, vk_backend, render) : la couche de
# validation selon KIRARI_TEST_VULKAN_VALIDATION, chaque erreur notée sous le nom du test, et le main
# de Catch2 qui fait échouer l'exécutable s'il en reste une. Bibliothèque d'objets : le main entre
# dans l'exécutable qui la lie (kirari_add_unit_tests ... MAIN kirari_test_vulkan).
add_library(kirari_test_vulkan OBJECT vulkan_validation.cpp vulkan_test_main.cpp)
target_include_directories(kirari_test_vulkan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(kirari_test_vulkan PUBLIC kirari::core_types Catch2::Catch2)
kirari_set_warnings(kirari_test_vulkan)
