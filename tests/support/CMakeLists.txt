# Support des tests qui créent un appareil Vulkan (vk_compute, vk_backend, render) : la couche de
# validation selon AJP_TEST_VULKAN_VALIDATION, chaque erreur notée sous le nom du test, et le main
# de Catch2 qui fait échouer l'exécutable s'il en reste une. Bibliothèque d'objets : le main entre
# dans l'exécutable qui la lie (ajp_add_unit_tests ... MAIN ajp_test_vulkan).
add_library(ajp_test_vulkan OBJECT vulkan_validation.cpp vulkan_test_main.cpp)
target_include_directories(ajp_test_vulkan PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(ajp_test_vulkan PUBLIC ajp::core_types Catch2::Catch2)
ajp_set_warnings(ajp_test_vulkan)
